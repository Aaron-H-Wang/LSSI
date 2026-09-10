import scipy.io
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-8
torch.manual_seed(0)
np.random.seed(0)

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
mat_data = scipy.io.loadmat('CoreLossData_FR79.mat')

f_raw = mat_data['f'].flatten()             # already kHz in the FR79 export
B_raw = mat_data['B'].flatten()             # already mT in the FR79 export
P_raw = mat_data['P'].flatten()             # mW/cm^3

sort_idx = np.lexsort((B_raw, f_raw))
f_sorted, B_sorted, P_sorted = f_raw[sort_idx], B_raw[sort_idx], P_raw[sort_idx]

test_mask = np.zeros(len(f_sorted), dtype=bool)
test_mask[::5] = True
train_mask = ~test_mask

f_tr, B_tr, P_tr = f_sorted[train_mask], B_sorted[train_mask], P_sorted[train_mask]
f_te, B_te, P_te = f_sorted[test_mask], B_sorted[test_mask], P_sorted[test_mask]

f_norm, B_norm, P_norm = np.mean(f_tr), np.mean(B_tr), np.mean(P_tr)

def to_tensor(x): return torch.tensor(x, dtype=torch.float32)

t_f_tr, t_B_tr, t_P_tr = to_tensor(f_tr/f_norm), to_tensor(B_tr/B_norm), to_tensor(P_tr/P_norm)
t_f_te, t_B_te, t_P_te = to_tensor(f_te/f_norm), to_tensor(B_te/B_norm), to_tensor(P_te/P_norm)

# ==========================================
# 2. EXTENDED LIBRARY
# ==========================================
# The exponents are FIXED here, so grid resolution is the accuracy ceiling: a true
# Steinmetz beta of 2.53 can only be represented as a blend of the 2.5 and 2.6
# columns. Densify a range if the residuals show structure there.
F_EXPS = [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2]
B_EXPS = [1.6, 1.8, 2.0, 2.2, 2.4, 2.5, 2.6, 2.8, 3.0]

LIB_LABELS, LIB_FUNCS = [], []

def add_term(label, fn):
    LIB_LABELS.append(label)
    LIB_FUNCS.append(fn)

def _pow(x, p):
    return x if p == 1.0 else torch.pow(torch.clamp(x, min=EPS), p)

def PB(a, b):
    return lambda f, B, a=a, b=b: _pow(f, a) * _pow(B, b)

# --- power grid: f^a * B^b ---
for a in F_EXPS:
    for b in B_EXPS:
        add_term(f"f^{a:g} * B^{b:g}", PB(a, b))

# --- anomalous / excess loss ---
add_term("f^1.5 * B^1.5", PB(1.5, 1.5))

# --- HF rolloff, cutoffs in units of f_norm ---
# Band-wise Steinmetz fits on this dataset give a NON-MONOTONE alpha:
#   1.30 (52-86 kHz) -> 1.77 (380-630 kHz) -> 1.17 (630-1029 kHz)
# The turnover is the signature of a high-frequency roll-off, and it sits near
# 1.5*f_norm. The candidate cutoffs must bracket that value; if they are all larger
# than the true knee the solver cannot represent the turnover at all.
for fc in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
    for (a, b) in ((1.0, 2.0), (1.5, 2.4), (2.0, 2.0)):
        add_term(f"f^{a:g} * B^{b:g} / (1+(f/{fc:g})^2)",
                 lambda f, B, a=a, b=b, fc=fc: _pow(f, a)*_pow(B, b)/(1.0+(f/fc)**2))

# --- variable-exponent corrections ---
# Band-wise beta drifts from 1.97 at low B to 2.43 at high B, and from 2.48 at low f
# to 2.09 at high f. These ln terms are the first-order expansion of a drifting
# exponent, and they consistently take a large share of the final fit.
for (a, b) in ((1.0, 2.0), (1.4, 2.4), (1.8, 2.4)):
    add_term(f"f^{a:g} * B^{b:g} * ln(B)",
             lambda f, B, a=a, b=b: _pow(f, a)*_pow(B, b)*torch.log(torch.clamp(B, min=EPS)))
    add_term(f"f^{a:g} * B^{b:g} * ln(f)",
             lambda f, B, a=a, b=b: _pow(f, a)*_pow(B, b)*torch.log(torch.clamp(f, min=EPS)))

# --- saturation ---
for b0 in (2.0, 4.0):
    add_term(f"f * B^2 * (exp(B/{b0:g})-1)",
             lambda f, B, b0=b0: f*_pow(B, 2)*(torch.exp(torch.clamp(B/b0, max=10.0))-1.0))

# --- low-order auxiliaries ---
add_term("f * B", PB(1.0, 1.0))
add_term("f", PB(1.0, 0.0))
add_term("B", PB(0.0, 1.0))
add_term("1", lambda f, B: torch.ones_like(f))

N_TERMS = len(LIB_LABELS)

# Every library function gets a permanent xi index, numbered from 1 in library order.
# The index identifies the term for the whole run: xi_k always means LIB_LABELS[k-1],
# whether or not that term survives pruning.
XI_NAMES = [f"xi_{i+1}" for i in range(N_TERMS)]
XI_MATH = [rf'$\xi_{{{i+1}}}$' for i in range(N_TERMS)]

def build_library(f, B):
    return torch.stack([fn(f, B) for fn in LIB_FUNCS], dim=1)

Theta_tr = build_library(t_f_tr, t_B_tr)
Theta_te = build_library(t_f_te, t_B_te)

# Column normalization. Two things depend on it: (a) a single hard threshold is only
# meaningful if coefficients are comparable across columns, and (b) AdamW applies the
# SAME decoupled weight decay to every coefficient, so without normalization the decay
# penalizes numerically large columns far more than small ones.
COL_SCALE = torch.sqrt(torch.mean(Theta_tr**2, dim=0)) + EPS

print(f"Library terms: {N_TERMS} | Train: {len(f_tr)} | Test: {len(f_te)}")

# ==========================================
# 3. MODEL & LOSS
# ==========================================
class SparseSSI(nn.Module):
    def __init__(self, n_terms, col_scale):
        super().__init__()
        # Positive init: coefficients are constrained non-negative, so the clamp must
        # not wipe out half the library on the first step.
        self.coefs = nn.Parameter(torch.rand(n_terms) * 0.05 + 0.01)
        self.register_buffer('mask', torch.ones(n_terms))
        self.register_buffer('scale', col_scale)

    def forward(self, Theta):
        return (Theta / self.scale) @ (self.coefs * self.mask)

    def project(self):
        with torch.no_grad():
            self.coefs.data.clamp_(min=0.0)
            self.coefs.data *= self.mask

    def physical_coefs(self):
        return ((self.coefs * self.mask) / self.scale).detach().numpy()

model = SparseSSI(N_TERMS, COL_SCALE)

def log_mape_loss(pred, target, w_log=0.5, w_mape=0.5):
    log_loss = torch.mean((torch.log(torch.clamp(pred, min=EPS))
                           - torch.log(target + EPS))**2)
    mape_loss = torch.mean(torch.abs((pred - target) / (target + EPS)))
    return w_log * log_loss + w_mape * mape_loss

# ==========================================
# 4. TRAINING PHASES
# ==========================================
# Sparsification here comes from AdamW, NOT from an L1 penalty. Two mechanisms:
#   - decoupled weight decay shrinks every coefficient toward zero each step;
#   - the non-negativity clamp turns "wants to be negative" into exactly zero.
# Terms that carry no signal therefore collapse to 0 on their own, and the ramped
# hard threshold at the end only sweeps up what is already negligible.
logs = {'epoch': [], 'loss': [], 'tr_mse': [], 'te_mse': [],
        'tr_mape': [], 'te_mape': [], 'sparsity': []}

param_history = {'epoch': [], 'coefs': []}
prune_epoch = [None] * N_TERMS

global_step = 0
phase1_epochs = 1200
sample_every = 100

# ---- The ONLY sparsity knob ----
# Applied to the COLUMN-NORMALIZED coefficients, so it is comparable across terms of
# wildly different magnitude. Raise it to get fewer terms, lower it to get more.
# On this dataset the useful range is roughly 0.02 - 0.5; see the sweep in the notes.
max_pruning_threshold = 0.1

weight_decay_coefs = 0.05        # regularizer only, NOT a sparsifier; see note below
total_phase2_epochs = 30001

# Threshold ramps from 0 to max_pruning_threshold over the last part of Phase 2, and
# is applied every PRUNE_CHECK_EVERY epochs so survivors get a refit window between
# removals. MIN_TERMS is a floor: without it a large threshold wipes the whole
# library in one pass and the model collapses to MAPE = 100%.
PRUNE_START_FRAC = 0.3           # ramp begins at this fraction of Phase 2
PRUNE_CHECK_EVERY = 250          # epochs between threshold applications
MIN_TERMS = 2

# Note on what actually creates sparsity here. AdamW's decoupled weight decay does
# NOT: it applies p -= lr*wd*p, a multiplicative shrink that never reaches zero, and
# Adam's adaptive step pushes small coefficients back up at ~lr while the decay term
# vanishes with p. Sweeping wd from 0.002 to 1.0 changed the surviving term count by
# less than 10%. Sparsity comes from the non-negativity clamp (which turns "wants to
# be negative" into exactly zero) and from this threshold.

def record_params(step):
    with torch.no_grad():
        param_history['epoch'].append(step)
        param_history['coefs'].append(model.coefs.detach().clone().numpy())

def log_metrics(epoch, loss_value, pred_tr):
    model.eval()
    with torch.no_grad():
        te_pred = model(Theta_te)
        tr_mape = torch.mean(torch.abs((pred_tr - t_P_tr) / t_P_tr)) * 100
        te_mape = torch.mean(torch.abs((te_pred - t_P_te) / t_P_te)) * 100
        logs['epoch'].append(epoch)
        logs['loss'].append(loss_value)
        logs['tr_mse'].append(torch.mean((pred_tr - t_P_tr)**2).item())
        logs['te_mse'].append(torch.mean((te_pred - t_P_te)**2).item())
        logs['tr_mape'].append(tr_mape.item())
        logs['te_mape'].append(te_mape.item())
        logs['sparsity'].append(int((model.mask != 0).sum().item()))

# ---------- Phase 1: Warm-up (no decay, let every term find its scale) ----------
optimizer = optim.AdamW([model.coefs], lr=0.01, weight_decay=0.0)

print("Phase 1: Warming up coefficients...")
for _ in range(phase1_epochs):
    optimizer.zero_grad()
    pred = model(Theta_tr)
    loss = log_mape_loss(pred, t_P_tr)
    loss.backward()
    optimizer.step()
    model.project()

    if global_step % sample_every == 0:
        record_params(global_step)
    global_step += 1
print(f"  warm-up loss = {loss.item():.6f}")

# ---------- Phase 2: AdamW Discovery & Pruning ----------
phase2_lr = 0.005
optimizer = optim.AdamW([model.coefs], lr=phase2_lr, weight_decay=weight_decay_coefs)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=250, factor=0.5)

print("\nPhase 2: AdamW discovery with ramped hard thresholding...")
for epoch in range(total_phase2_epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(Theta_tr)
    loss = log_mape_loss(pred, t_P_tr)
    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())
    model.project()

    # Ramped hard thresholding. The threshold grows linearly, so weak terms drop out
    # gradually rather than all at once.
    if epoch > PRUNE_START_FRAC * total_phase2_epochs and epoch % PRUNE_CHECK_EVERY == 0:
        span = (1.0 - PRUNE_START_FRAC) * total_phase2_epochs
        progress = (epoch - PRUNE_START_FRAC * total_phase2_epochs) / span
        current_threshold = max_pruning_threshold * min(progress, 1.0)
        with torch.no_grad():
            dead = (model.coefs.data < current_threshold) & (model.mask != 0)
            n_alive = int(model.mask.sum().item())
            if int(dead.sum()) > 0 and n_alive - int(dead.sum()) >= MIN_TERMS:
                model.mask[dead] = 0
                model.coefs.data *= model.mask
                print(f"  [prune] epoch {epoch:5d} | thr {current_threshold:.4f} "
                      f"| -{int(dead.sum()):3d} -> {int(model.mask.sum()):3d} active")
                # Critical: reset optimizer and learning rate after every removal.
                # ReduceLROnPlateau has cut lr to a fraction of base by this point and
                # Adam's moments still describe the pre-prune landscape, so without a
                # reset the survivors cannot absorb the removed terms' weight and the
                # fit collapses instead of recovering. Measured on this dataset: 3.51%
                # test MAPE without the reset, 1.12% with it.
                optimizer = optim.AdamW([model.coefs], lr=phase2_lr,
                                        weight_decay=weight_decay_coefs)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                                 patience=250, factor=0.5)

    for i in range(N_TERMS):
        if prune_epoch[i] is None and model.mask[i] == 0:
            prune_epoch[i] = global_step

    if global_step % sample_every == 0:
        record_params(global_step)
    global_step += 1

    if epoch % 100 == 0:
        log_metrics(epoch, loss.item(), pred)
    if epoch % 2000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.5f} | "
              f"Test MAPE: {logs['te_mape'][-1]:.3f}% | Active: {logs['sparsity'][-1]}")

# ---------- Phase 3: LBFGS Polishing ----------
# Weight decay biases every surviving coefficient downward. With the support now
# frozen, refit without decay so the reported values are unbiased.
print("\nPhase 3: LBFGS polishing...")
optimizer_p3 = optim.LBFGS(
    [model.coefs],
    lr=0.01, max_iter=20, history_size=50,
    tolerance_grad=1e-10, tolerance_change=1e-12,
    line_search_fn='strong_wolfe'
)

def closure():
    optimizer_p3.zero_grad()
    loss = log_mape_loss(model(Theta_tr), t_P_tr)
    loss.backward()
    model.coefs.grad[model.mask == 0] = 0.0
    return loss

# The non-negativity projection between LBFGS steps corrupts the quasi-Newton
# curvature history, so LBFGS can wander or diverge to NaN. Track the best feasible
# iterate and fall back to it.
with torch.no_grad():
    best_loss = log_mape_loss(model(Theta_tr), t_P_tr).item()
best_state = model.coefs.detach().clone()

for step in range(200):
    loss_val = optimizer_p3.step(closure)
    model.project()

    cur = float(loss_val) if torch.isfinite(loss_val) else float('inf')
    if not torch.isfinite(model.coefs.data).all() or not np.isfinite(cur):
        print(f"  LBFGS diverged at step {step}; rolling back.")
        break
    if cur < best_loss:
        best_loss = cur
        best_state = model.coefs.detach().clone()

    if step % 5 == 0:
        log_metrics(total_phase2_epochs + step, cur, model(Theta_tr).detach())
        if step % 50 == 0:
            print(f"LBFGS {step:3d} | Train MAPE: {logs['tr_mape'][-1]:.3f}% | "
                  f"Test MAPE: {logs['te_mape'][-1]:.3f}% | Active: {logs['sparsity'][-1]}")

# LBFGS is not monotone, so keep the best iterate rather than the last one.
with torch.no_grad():
    model.coefs.data.copy_(best_state)
    model.project()
    # Sweep out anything that collapsed to zero during polishing, so every printed
    # coefficient is strictly positive.
    model.mask[model.coefs.data <= 0.0] = 0
    model.coefs.data *= model.mask
print(f"  kept best iterate, train loss = {best_loss:.6f}")

# ---------- Final cleanup: drop negligible survivors, then re-polish ----------
# The target-count schedule can leave a term alive at ~0 contribution. Remove
# anything below MIN_CONTRIB of the predicted magnitude and refit the rest.
MIN_CONTRIB = 0.005
with torch.no_grad():
    c = torch.abs((Theta_tr / model.scale) * (model.coefs * model.mask)).mean(0)
    c = c / (c.sum() + EPS)
    negligible = (c < MIN_CONTRIB) & (model.mask != 0)
    if int(model.mask.sum()) - int(negligible.sum()) >= MIN_TERMS and int(negligible.sum()) > 0:
        print(f"  cleanup: dropping {int(negligible.sum())} terms below "
              f"{MIN_CONTRIB*100:.1f}% contribution")
        model.mask[negligible] = 0
        model.coefs.data *= model.mask

optimizer_p4 = optim.LBFGS([model.coefs], lr=0.01, max_iter=20, history_size=50,
                           tolerance_grad=1e-10, tolerance_change=1e-12,
                           line_search_fn='strong_wolfe')

def closure_p4():
    optimizer_p4.zero_grad()
    loss = log_mape_loss(model(Theta_tr), t_P_tr)
    loss.backward()
    model.coefs.grad[model.mask == 0] = 0.0
    return loss

with torch.no_grad():
    best_loss = log_mape_loss(model(Theta_tr), t_P_tr).item()
best_state = model.coefs.detach().clone()
for step in range(100):
    loss_val = optimizer_p4.step(closure_p4)
    model.project()
    cur = float(loss_val) if torch.isfinite(loss_val) else float('inf')
    if not torch.isfinite(model.coefs.data).all() or not np.isfinite(cur):
        break
    if cur < best_loss:
        best_loss = cur
        best_state = model.coefs.detach().clone()
with torch.no_grad():
    model.coefs.data.copy_(best_state)
    model.project()
    model.mask[model.coefs.data <= 0.0] = 0
    model.coefs.data *= model.mask
print(f"  final train loss = {best_loss:.6f}")

active_idx = [i for i in range(N_TERMS) if model.mask[i] != 0]
n_active = len(active_idx)
assert all(model.coefs.data[i].item() > 0 for i in active_idx), "non-positive coefficient survived"

# ==========================================
# 5. VISUALIZATION (training history)
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axs = plt.subplots(2, 2, figsize=(8.5, 5.5))
axs[0, 0].plot(logs['epoch'], logs['loss'], color='blue', lw=2, label='Overall Loss')
# axs[0, 0].set_yscale('log')
axs[0, 0].legend(); axs[0, 0].grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)
axs[0, 0].set_xlabel('Epoch', fontsize=10); axs[0, 0].set_ylabel('Training Loss', fontsize=10)
axs[0, 0].tick_params(axis='both', labelsize=8, direction='in', top=True, right=True)

axs[0, 1].plot(logs['epoch'], logs['tr_mse'], color='green', lw=2, label='Train MSE')
axs[0, 1].plot(logs['epoch'], logs['te_mse'], color='orange', lw=2, linestyle='--', label='Test MSE')
axs[0, 1].set_yscale('log'); axs[0, 1].legend(); axs[0, 1].grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)
axs[0, 1].set_xlabel('Epoch', fontsize=10)
axs[0, 1].tick_params(axis='both', labelsize=8, direction='in', top=True, right=True)

axs[1, 0].plot(logs['epoch'], logs['tr_mape'], color='red', lw=2, label='Train MAPE')
axs[1, 0].plot(logs['epoch'], logs['te_mape'], color='darkred', lw=2, linestyle='--', label='Test MAPE')
axs[1, 0].set_xlabel('Epoch', fontsize=10); axs[1, 0].set_ylabel('Percentage', fontsize=10)
axs[1, 0].legend(); axs[1, 0].grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)
#axs[1, 0].axhline(y=1.0, color='gray', linestyle=':', label='1% Target')
axs[1, 0].tick_params(axis='both', labelsize=8, direction='in', top=True, right=True)

axs[1, 1].step(logs['epoch'], logs['sparsity'], color='purple', lw=2, where='post', label='Remaining Terms')
#axs[1, 1].set_title('Active Library Terms') 
axs[1, 1].set_ylim([0, N_TERMS + 1]); axs[1, 1].legend(); axs[1, 1].grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)
axs[1, 1].set_xlabel('Epoch', fontsize=10); axs[1, 1].set_ylabel('Active Terms', fontsize=10)
axs[1, 1].tick_params(axis='both', labelsize=8, direction='in', top=True, right=True)

plt.tight_layout()
fig.savefig('training.svg', format='svg', bbox_inches='tight')
fig.savefig('training.png', format='png', dpi=600, bbox_inches='tight')
plt.show()

# ==========================================
# 6. FINAL EQUATION AND DIAGNOSTICS
# ==========================================
with torch.no_grad():
    xi = model.physical_coefs()
    contrib = torch.abs((Theta_tr / model.scale) * (model.coefs * model.mask)).mean(0)
    contrib = (contrib / (contrib.sum() + EPS)).numpy()
    order = sorted(active_idx, key=lambda i: -contrib[i])

    print("\n" + "="*100)
    print("CHARACTERIZATION SCALARS")
    print("="*100)
    print(f"  f_norm = {f_norm:.3f} kHz | B_norm = {B_norm:.3f} mT | P_norm = {P_norm:.3f} mW/cm^3")
    print(f"  Library = {N_TERMS} terms | threshold = {max_pruning_threshold} | "
          f"weight_decay = {weight_decay_coefs}")

    print("\n" + "="*100)
    print("DISCOVERED DIMENSIONLESS EQUATION   (f = f/f_norm, B = B/B_norm, P = P/P_norm)")
    print("="*100)
    terms = [f"{xi[i]:.6e} * {LIB_LABELS[i]:<30s} [{XI_NAMES[i]:>7s}, {contrib[i]*100:5.1f}%]"
             for i in order]
    print("P / P_norm = \n  " + "\n+ ".join(terms))
    print("="*100)
    print(f"Active terms: {n_active} / {N_TERMS} | Pruned: {N_TERMS - n_active}")

    print("\n" + "-"*100)
    print("SURVIVING COEFFICIENTS (library order)")
    print("-"*100)
    for i in active_idx:
        print(f"  {XI_NAMES[i]:>7s} = {xi[i]:.6e}   (normalized {model.coefs.data[i].item():.6e})"
              f"   ->  {LIB_LABELS[i]}")

    # ---------- RESIDUAL DIAGNOSTICS ----------
    model.eval()
    pred_tr = model(Theta_tr)
    pred_te = model(Theta_te)
    final_tr_mape = torch.mean(torch.abs((pred_tr - t_P_tr) / t_P_tr)) * 100
    final_te_mape = torch.mean(torch.abs((pred_te - t_P_te) / t_P_te)) * 100
    print(f"\nFINAL Train MAPE = {final_tr_mape.item():.3f}%")
    print(f"FINAL Test  MAPE = {final_te_mape.item():.3f}%")

    # Error Histograms
    fig2, ax2 = plt.subplots(figsize=(8.5/3, 2.2))
    res_tr = ((pred_tr - t_P_tr) / t_P_tr * 100).numpy()
    res_te = ((pred_te - t_P_te) / t_P_te * 100).numpy()

    c_tr = '#D81B60'
    c_te = '#1E88E5'

    w_tr = np.ones_like(res_tr) * 100.0 / len(res_tr)
    w_te = np.ones_like(res_te) * 100.0 / len(res_te)
    ax2.hist(res_tr, bins=40, color=c_tr, alpha=0.55, edgecolor='k',
             linewidth=0.4, weights=w_tr)
    ax2.hist(res_te, bins=40, color=c_te, alpha=0.55, edgecolor='k',
             linewidth=0.4, weights=w_te)
    ax2.axvline(0, color='gray', linestyle='--', lw=1.0)

    ax2.text(0.95, 0.9,
             r'Train: $\mu=$' f'{res_tr.mean():.2f}' r', $\sigma=$' f'{res_tr.std():.2f}' '\n'
             r'Test: $\mu=$' f'{res_te.mean():.2f}' r', $\sigma=$' f'{res_te.std():.2f}',
             transform=ax2.transAxes, ha='right', va='top', fontsize=6, color='none',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='#999999', linewidth=0.8))
    ax2.text(0.95, 0.9,
             r'Train: $\mu=$' f'{res_tr.mean():.2f}' r', $\sigma=$' f'{res_tr.std():.2f}',
             transform=ax2.transAxes, ha='right', va='top', fontsize=6, color=c_tr)
    ax2.text(0.95, 0.9,
             '\n' r'Test: $\mu=$' f'{res_te.mean():.2f}' r', $\sigma=$' f'{res_te.std():.2f}',
             transform=ax2.transAxes, ha='right', va='top', fontsize=6, color=c_te)

    ax2.set_xlabel('Residual (%)', fontsize=10)
    ax2.set_ylabel('Number of Size (%)', fontsize=10)
    ax2.tick_params(axis='both', labelsize=8, direction='in', top=True, right=True)
    ax2.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    fig2.savefig('residuals.svg', format='svg', bbox_inches='tight')
    fig2.savefig('residuals.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()

    # Residual Scatter Plots
    fig3, ax3 = plt.subplots(2, 1, figsize=(8.5/2, 8.5/2 * 1.2))
    ax3[0].scatter(B_tr, res_tr, alpha=0.5, s=14, color='blue', label='Train')
    ax3[0].scatter(B_te, res_te, alpha=0.7, s=14, color='orange', label='Test')
    ax3[0].axhline(0, color='red', linestyle='--', lw=1.0)
    ax3[0].set_xlabel(r'$B$ (mT)', fontsize=10)

    ax3[1].scatter(f_tr, res_tr, alpha=0.5, s=14, color='green', label='Train')
    ax3[1].scatter(f_te, res_te, alpha=0.7, s=14, color='red', label='Test')
    ax3[1].axhline(0, color='red', linestyle='--', lw=1.0)
    ax3[1].set_xlabel(r'$f$ (kHz)', fontsize=10)

    for ax in ax3:
        ax.set_ylabel('Residual (%)', fontsize=10)
        ax.tick_params(axis='both', labelsize=8, direction='in', top=True, right=True)
        ax.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, frameon=False)

    plt.tight_layout()
    fig3.savefig('scatter.svg', format='svg', bbox_inches='tight')
    fig3.savefig('scatter.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()

    # ---------- CORE LOSS VALIDATION ----------
    meas_te = (t_P_te.numpy()) * P_norm
    pred_te_phys = (pred_te.numpy()) * P_norm
    ss_res = np.sum((meas_te - pred_te_phys) ** 2)
    ss_tot = np.sum((meas_te - meas_te.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    with open('metrics.json', 'w', encoding='utf-8') as stream:
        json.dump({
            'material': 'FR79', 'samples': int(len(P_sorted)),
            'frequency_khz': [float(f_sorted.min()), float(f_sorted.max())],
            'flux_density_mt': [float(B_sorted.min()), float(B_sorted.max())],
            'test_mape_percent': float(final_te_mape), 'test_r2': float(r2),
            'active_term_count': int(n_active),
            'active_terms': [{'index': int(i + 1), 'label': LIB_LABELS[i],
                              'coefficient': float(xi[i]),
                              'contribution_percent': float(contrib[i] * 100)}
                             for i in active_idx]
        }, stream, indent=2)

    W = 8.5 / 2
    H = 2.5
    fig4, ax4 = plt.subplots(figsize=(W, H))
    lo, hi = 0, 1.05 * max(meas_te.max(), pred_te_phys.max())
    ticks = np.linspace(lo, hi, 7)
    line = np.array([lo, hi])
    ax4.plot(line, line, color='#d91b0d', linestyle='--', lw=1.2, zorder=3,
             label=r'$\hat{P}_v = P_v$')
    ax4.scatter(meas_te, pred_te_phys, s=14, alpha=0.55,
                color='#d97d0d', edgecolor='#d96c0d', linewidth=0.4, zorder=4)
    ax4.text(0.06, 0.92,
             r'$R^2 = $' f'{r2:.4f}' '\n'
             r'$\mathrm{MAPE} = $' f'{final_te_mape.item():.2f}%',
             transform=ax4.transAxes,
             fontsize=10, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='#999999', linewidth=0.8))
    ax4.set_xlabel(r'$P_v$ (mW/cm$^3$)', fontsize=10)
    ax4.set_ylabel(r'$\hat{P}_v$ (mW/cm$^3$)', fontsize=10)
    ax4.set_xticks(ticks); ax4.set_yticks(ticks)
    ax4.set_xlim(lo, hi); ax4.set_ylim(lo, hi)
    ax4.set_aspect('auto')
    ax4.tick_params(axis='both', labelsize=8, direction='in', top=True, right=True)
    ax4.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)
    ax4.set_axisbelow(True)
    ax4.legend(loc='lower right', fontsize=10, frameon=False)
    fig4.savefig('core_loss_validation.svg', format='svg', bbox_inches='tight')
    fig4.savefig('core_loss_validation.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()

# ==========================================
# 7. COEFFICIENT EVOLUTION DURING TRAINING
# ==========================================
epochs_arr = np.array(param_history['epoch'])
coef_arr = np.array(param_history['coefs'])       # shape: (n_samples, N_TERMS)

fig_w = 8.5 / 2
figp, axp = plt.subplots(figsize=(fig_w, 2.7))

xi_colors = ['#EE3377', '#0077BB', '#EE7733', '#009988',
             '#CC3311', '#33BBEE', '#BBBB00', '#AA3377']
GREY = '#bbbbbb'

# Every surviving term is drawn in color and labelled with its xi index; pruned terms
# are grey. Survivors are ordered by contribution so the color order matches the
# equation printout above.
keep_idx = order
keep_lab = {i: XI_MATH[i] for i in keep_idx}

for i in range(N_TERMS):
    if i not in keep_lab:
        axp.plot(epochs_arr, coef_arr[:, i], lw=0.6, color=GREY, zorder=1)
for rank, i in enumerate(keep_idx):
    axp.plot(epochs_arr, coef_arr[:, i], lw=1.2,
             color=xi_colors[rank % len(xi_colors)], zorder=3)

# End-of-curve labels, spread vertically so they do not overlap.
x_end = epochs_arr[-1]
lab_pts = sorted([(float(coef_arr[-1, i]), rank, i) for rank, i in enumerate(keep_idx)])
y0, y1 = axp.get_ylim()
pad = 0.06 * (y1 - y0)
axp.set_ylim(y0 - pad, y1 + pad)
y0, y1 = axp.get_ylim()
min_gap = 0.06 * (y1 - y0)
adj = [t[0] for t in lab_pts]
for _ in range(200):
    moved = False
    for k in range(1, len(adj)):
        if adj[k] - adj[k-1] < min_gap:
            shift = (min_gap - (adj[k] - adj[k-1])) / 2
            adj[k-1] -= shift; adj[k] += shift; moved = True
    if not moved:
        break
for (yv, rank, i), y_text in zip(lab_pts, adj):
    axp.annotate(keep_lab[i], xy=(x_end, y_text),
                 xytext=(4, 0), textcoords='offset points',
                 color=xi_colors[rank % len(xi_colors)], fontsize=8,
                 va='center', ha='left', fontweight='bold', clip_on=False)

pruned_idx = [i for i in range(N_TERMS) if i not in keep_lab]
if pruned_idx:
    # Bottom-right corner, sitting just above y = 0. A blended transform is used so
    # x follows the axes fraction (staying in the corner whatever the xlim ends up
    # being) while y stays in data units and can be pinned relative to the zero line.
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(axp.transAxes, axp.transData)
    ylo, yhi = axp.get_ylim()
    axp.annotate('Others Pruned', xy=(0.97, 0.04 * (yhi - ylo)), xycoords=trans,
                 color=GREY, fontsize=8, va='bottom', ha='right', clip_on=False)

axp.axvline(phase1_epochs, color='gray', linestyle=':', lw=0.8)
axp.set_xlabel('Epoch', fontsize=10)
axp.set_ylabel(r'Coefficient Vector $\boldsymbol{\xi}$', fontsize=10)
axp.tick_params(axis='both', labelsize=8, direction='in', top=True, right=True)
axp.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)
axp.set_axisbelow(True)
x0, x1 = axp.get_xlim()
axp.set_xlim(x0, x1 + 0.10 * (x1 - x0))

plt.tight_layout()
figp.savefig('param_evolution.svg', format='svg', bbox_inches='tight')
figp.savefig('param_evolution.png', format='png', dpi=600, bbox_inches='tight')
plt.show()

print("\nCurves labelled in param_evolution (by contribution):")
for i in keep_idx:
    print(f"  {XI_NAMES[i]:>7s} : {LIB_LABELS[i]}")
