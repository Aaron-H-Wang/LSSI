import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
mat_data = scipy.io.loadmat('CoreLossData_FR95.mat')

f_raw = mat_data['f'].flatten() / 1000.0
B_raw = mat_data['B'].flatten() / 10.0
P_raw = mat_data['P'].flatten()

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

EPS = 1e-8

# ==========================================
# 2. MODEL
# ==========================================
class ConstrainedSINDy(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_h = nn.Parameter(torch.zeros(2))
        self.raw_e = nn.Parameter(torch.zeros(2))
        self.raw_a = nn.Parameter(torch.zeros(2))
        self.raw_s = nn.Parameter(torch.zeros(2))
        self.gamma_h = nn.Parameter(torch.tensor(0.0))
        self.sat_g  = nn.Parameter(torch.tensor(-0.5))
        self.raw_fc = nn.Parameter(torch.tensor(0.0))
        self.raw_pc = nn.Parameter(torch.tensor(0.0))
        self.raw_fc2 = nn.Parameter(torch.tensor(1.0))
        self.raw_pc2 = nn.Parameter(torch.tensor(0.0))
        self.coefs = nn.Parameter(torch.rand(10) * 0.1)
        self.register_buffer('mask', torch.ones(10))
        self.bounds_h   = (0.8, 1.3, 2.0, 2.8)
        self.bounds_e   = (1.8, 2.2, 1.8, 2.2)
        self.bounds_a   = (1.3, 1.7, 1.3, 1.7)
        self.bounds_s   = (0.9, 1.2, 2.8, 3.5)
        self.bounds_fc  = (2.0, 6.0)
        self.bounds_pc  = (1.0, 4.0)
        self.bounds_fc2 = (3.0, 10.0)
        self.bounds_pc2 = (1.0, 4.0)

    def _map(self, raw, bounds):
        s = torch.sigmoid(raw)
        f_exp = bounds[0] + (bounds[1] - bounds[0]) * s[0]
        B_exp = bounds[2] + (bounds[3] - bounds[2]) * s[1]
        return f_exp, B_exp

    def _map_scalar(self, raw, bounds):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.sigmoid(raw)

    def get_exponents(self):
        return {
            'h':   self._map(self.raw_h, self.bounds_h),
            'e':   self._map(self.raw_e, self.bounds_e),
            'a':   self._map(self.raw_a, self.bounds_a),
            's':   self._map(self.raw_s, self.bounds_s),
            'fc':  self._map_scalar(self.raw_fc, self.bounds_fc),
            'pc':  self._map_scalar(self.raw_pc, self.bounds_pc),
            'fc2': self._map_scalar(self.raw_fc2, self.bounds_fc2),
            'pc2': self._map_scalar(self.raw_pc2, self.bounds_pc2),
        }

    def forward(self, f, B):
        exps = self.get_exponents()
        fh, bh = exps['h']; fe, be = exps['e']; fa, ba = exps['a']; fs, bs = exps['s']
        fc, pc = exps['fc'], exps['pc']; fc2, pc2 = exps['fc2'], exps['pc2']
        rolloff_e = 1.0 / (1.0 + torch.pow(f / fc,  pc))
        rolloff_h = 1.0 / (1.0 + torch.pow(f / fc2, pc2))
        logB = torch.log(B + EPS)
        bh_mod = bh + self.gamma_h * logB
        theta = [
            torch.pow(f, fh) * torch.pow(B, bh_mod) * rolloff_h,
            torch.pow(f, fe) * torch.pow(B, be) * rolloff_e,
            torch.pow(f, fa) * torch.pow(B, ba) * rolloff_e,
            torch.pow(f, fs) * torch.pow(B, bs),
            f * torch.exp(self.sat_g * B),
            f * B,
            f * torch.pow(B, 2),
            f,
            B,
            torch.ones_like(f)
        ]
        active_coefs = self.coefs * self.mask
        return sum(c * t for c, t in zip(active_coefs, theta))

model = ConstrainedSINDy()

# ==========================================
# 3. LOSS
# ==========================================
def log_mape_loss(pred, target, w_log=0.5, w_mape=0.5):
    log_loss = torch.mean((torch.log(torch.clamp(pred, min=EPS))
                           - torch.log(target + EPS))**2)
    mape_loss = torch.mean(torch.abs((pred - target) / (target + EPS)))
    return w_log * log_loss + w_mape * mape_loss

# ==========================================
# 4. TRAINING PHASES
# ==========================================
logs = {'epoch': [], 'loss': [], 'tr_mse': [], 'te_mse': [], 'tr_mape': [], 'te_mape': [], 'sparsity': []}

# --- 参数演化记录 ---
param_history = {'epoch': []}
coef_names = [f'coef[{i}]' for i in range(10)]
exp_names = ['raw_h_f', 'raw_h_B', 'raw_e_f', 'raw_e_B', 'raw_a_f', 'raw_a_B',
             'raw_s_f', 'raw_s_B', 'gamma_h', 'sat_g',
             'raw_fc', 'raw_pc', 'raw_fc2', 'raw_pc2']
for name in coef_names + exp_names:
    param_history[name] = []
prune_epoch = [None] * 10

global_step = 0          # 贯穿 Phase 1 + Phase 2 的连续计数器
phase1_epochs = 1200
sample_every = 100

def record_params(step):
    """采样当前所有被追踪参数到 param_history。"""
    with torch.no_grad():
        param_history['epoch'].append(step)
        for i in range(10):
            param_history[f'coef[{i}]'].append(model.coefs[i].item())
        flat = [model.raw_h[0], model.raw_h[1], model.raw_e[0], model.raw_e[1],
                model.raw_a[0], model.raw_a[1], model.raw_s[0], model.raw_s[1],
                model.gamma_h, model.sat_g,
                model.raw_fc, model.raw_pc, model.raw_fc2, model.raw_pc2]
        for name, val in zip(exp_names, flat):
            param_history[name].append(val.item())

max_pruning_threshold = 0.001
total_phase2_epochs = 20001

exp_params = [model.raw_h, model.raw_e, model.raw_a, model.raw_s,
              model.gamma_h, model.sat_g,
              model.raw_fc, model.raw_pc, model.raw_fc2, model.raw_pc2]

# ---------- Phase 1: Warm-up ----------
for p in exp_params: p.requires_grad = False
optimizer = optim.AdamW([model.coefs], lr=0.01, weight_decay=0.0)

print("Phase 1: Warming up coefficients...")
for _ in range(phase1_epochs):
    optimizer.zero_grad()
    pred = model(t_f_tr, t_B_tr)
    loss = log_mape_loss(pred, t_P_tr)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        model.coefs.data.clamp_(min=0.0)

    if global_step % sample_every == 0:
        record_params(global_step)
    global_step += 1

# ---------- Phase 2: Joint Discovery & Pruning ----------
for p in exp_params: p.requires_grad = True

optimizer = optim.AdamW([
    {'params': [model.coefs], 'weight_decay': 0.002},
    {'params': exp_params, 'weight_decay': 0.0}
], lr=0.001)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=250, factor=0.5)

print("\nPhase 2: Joint discovery with dual HF rolloff...")
for epoch in range(total_phase2_epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(t_f_tr, t_B_tr)
    loss = log_mape_loss(pred, t_P_tr)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    with torch.no_grad():
        model.coefs.data.clamp_(min=0.0)

    if epoch > 0.75 * total_phase2_epochs:
        progress = (epoch - 0.75 * total_phase2_epochs) / (0.25 * total_phase2_epochs)
        current_threshold = max_pruning_threshold * progress
        with torch.no_grad():
            model.mask[torch.abs(model.coefs.data) < current_threshold] = 0
            model.coefs.data *= model.mask

    # 记录剪枝时刻（全局口径）
    for i in range(10):
        if prune_epoch[i] is None and model.mask[i] == 0:
            prune_epoch[i] = global_step

    # 参数演化采样（接续 Phase 1）
    if global_step % sample_every == 0:
        record_params(global_step)
    global_step += 1

    # 指标日志（本地 epoch 口径）
    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            tr_mape = torch.mean(torch.abs((pred - t_P_tr) / t_P_tr)) * 100
            te_pred = model(t_f_te, t_B_te)
            te_mape = torch.mean(torch.abs((te_pred - t_P_te) / t_P_te)) * 100
            tr_mse = torch.mean((pred - t_P_tr)**2)
            te_mse = torch.mean((te_pred - t_P_te)**2)
            logs['epoch'].append(epoch)
            logs['loss'].append(loss.item())
            logs['tr_mse'].append(tr_mse.item())
            logs['te_mse'].append(te_mse.item())
            logs['tr_mape'].append(tr_mape.item())
            logs['te_mape'].append(te_mape.item())
            logs['sparsity'].append((model.mask != 0).sum().item())

    if epoch % 1000 == 0:
        with torch.no_grad():
            ex = model.get_exponents()
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.5f} | Test MAPE: {te_mape.item():.3f}% | "
              f"Active: {logs['sparsity'][-1]} | fc_e: {ex['fc'].item():.2f} | fc_h: {ex['fc2'].item():.2f}")

# ---------- Phase 3: LBFGS Polishing ----------
print("\nPhase 3: LBFGS polishing...")
optimizer_p3 = optim.LBFGS(
    list(model.parameters()),
    lr=0.01, max_iter=20, history_size=50,
    tolerance_grad=1e-10, tolerance_change=1e-12,
    line_search_fn='strong_wolfe'
)

def closure():
    optimizer_p3.zero_grad()
    pred = model(t_f_tr, t_B_tr)
    loss = log_mape_loss(pred, t_P_tr)
    loss.backward()
    return loss

for step in range(200):
    optimizer_p3.step(closure)
    with torch.no_grad():
        model.coefs.data.clamp_(min=0.0)
        model.coefs.data *= model.mask

    if step % 5 == 0:
        model.eval()
        with torch.no_grad():
            pred = model(t_f_tr, t_B_tr)
            tr_mape = torch.mean(torch.abs((pred - t_P_tr) / t_P_tr)) * 100
            te_pred = model(t_f_te, t_B_te)
            te_mape = torch.mean(torch.abs((te_pred - t_P_te) / t_P_te)) * 100
            tr_mse = torch.mean((pred - t_P_tr)**2)
            te_mse = torch.mean((te_pred - t_P_te)**2)
            global_epoch = total_phase2_epochs + step
            logs['epoch'].append(global_epoch)
            logs['loss'].append(tr_mse.item())
            logs['tr_mse'].append(tr_mse.item())
            logs['te_mse'].append(te_mse.item())
            logs['tr_mape'].append(tr_mape.item())
            logs['te_mape'].append(te_mape.item())
            logs['sparsity'].append((model.mask != 0).sum().item())
            print(f"LBFGS {step:3d} | Train MAPE: {tr_mape:.3f}% | Test MAPE: {te_mape:.3f}% | Active: {logs['sparsity'][-1]}")

# ==========================================
# 5. VISUALIZATION (training history)
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs[0, 0].plot(logs['epoch'], logs['loss'], color='blue', lw=2); axs[0, 0].set_yscale('log')
axs[0, 0].set_title('Training Objective History'); axs[0, 0].grid(True)
axs[0, 1].plot(logs['epoch'], logs['tr_mse'], color='green', lw=2, label='Train MSE')
axs[0, 1].plot(logs['epoch'], logs['te_mse'], color='orange', lw=2, linestyle='--', label='Test MSE')
axs[0, 1].set_yscale('log'); axs[0, 1].set_title('MSE'); axs[0, 1].legend(); axs[0, 1].grid(True)
axs[1, 0].plot(logs['epoch'], logs['tr_mape'], color='red', lw=2, label='Train MAPE')
axs[1, 0].plot(logs['epoch'], logs['te_mape'], color='darkred', lw=2, linestyle='--', label='Test MAPE')
axs[1, 0].axhline(y=1.0, color='gray', linestyle=':', label='1% Target')
axs[1, 0].set_title('MAPE (%)'); axs[1, 0].legend(); axs[1, 0].grid(True)
axs[1, 1].step(logs['epoch'], logs['sparsity'], color='purple', lw=2, where='post')
axs[1, 1].set_title('Active Library Terms'); axs[1, 1].set_ylim([0, 11]); axs[1, 1].grid(True)
plt.tight_layout(); plt.show()

# ==========================================
# 6. FINAL EQUATION AND DIAGNOSTICS
# ==========================================
with torch.no_grad():
    xi = (model.coefs * model.mask).numpy()
    exps = model.get_exponents()
    fh, bh = exps['h'][0].item(), exps['h'][1].item()
    fe, be = exps['e'][0].item(), exps['e'][1].item()
    fa, ba = exps['a'][0].item(), exps['a'][1].item()
    fs, bs = exps['s'][0].item(), exps['s'][1].item()
    fc, pc = exps['fc'].item(), exps['pc'].item()
    fc2, pc2 = exps['fc2'].item(), exps['pc2'].item()
    gamma = model.gamma_h.item()
    delta = model.sat_g.item()

    labels = [
        f"(f/f_n)^{fh:.3f} * (B/B_n)^({bh:.3f} {gamma:+.3f}*log(B/B_n)) * R_h(f)       [hysteresis*]",
        f"(f/f_n)^{fe:.3f} * (B/B_n)^{be:.3f} * R_e(f)                                 [eddy]",
        f"(f/f_n)^{fa:.3f} * (B/B_n)^{ba:.3f} * R_e(f)                                 [anomalous]",
        f"(f/f_n)^{fs:.3f} * (B/B_n)^{bs:.3f}                                          [saturation]",
        f"(f/f_n) * exp({delta:.3f}*(B/B_n))                                           [exp sat]",
        "(f/f_n) * (B/B_n)                                                             [cross]",
        "(f/f_n) * (B/B_n)^2                                                           [aux]",
        "(f/f_n)                                                                       [aux]",
        "(B/B_n)                                                                       [aux]",
        "1                                                                             [bias]"
    ]

    print("\n" + "="*100)
    print("CHARACTERIZATION SCALARS")
    print("="*100)
    print(f"  f_norm = {f_norm:.3f} kHz | B_norm = {B_norm:.3f} mT | P_norm = {P_norm:.3f} mw/cm^3")
    print(f"  Eddy/anom rolloff R_e(f) = 1/(1+(f/fc)^p)    fc  = {fc*f_norm:.1f} kHz, p  = {pc:.3f}")
    print(f"  Hysteresis rolloff R_h(f) = 1/(1+(f/fc2)^p2) fc2 = {fc2*f_norm:.1f} kHz, p2 = {pc2:.3f}")

    print("\n" + "="*100)
    print("DISCOVERED DIMENSIONLESS EQUATION")
    print("="*100)
    active_terms = [f"{xi[i]:.4f} * {labels[i]}" for i in range(10) if model.mask[i] != 0]
    if active_terms:
        print("P / P_norm = \n  " + "\n+ ".join(active_terms))
    print("="*100)

    # ---------- RESIDUAL DIAGNOSTICS ----------
    model.eval()
    pred_tr = model(t_f_tr, t_B_tr)
    pred_te = model(t_f_te, t_B_te)
    final_tr_mape = torch.mean(torch.abs((pred_tr - t_P_tr) / t_P_tr)) * 100
    final_te_mape = torch.mean(torch.abs((pred_te - t_P_te) / t_P_te)) * 100
    print(f"\nFINAL Train MAPE = {final_tr_mape.item():.3f}%")
    print(f"FINAL Test  MAPE = {final_te_mape.item():.3f}%")

    # Error Histograms
    fig2, ax2 = plt.subplots(figsize=(8.5/3, 2.2))
    res_tr = ((pred_tr - t_P_tr) / t_P_tr * 100).numpy()
    res_te = ((pred_te - t_P_te) / t_P_te * 100).numpy()

    # 强对比、色盲友好的两色
    c_tr = '#D81B60'   
    c_te = '#1E88E5'   

    # 以各自样本数归一化为百分比
    w_tr = np.ones_like(res_tr) * 100.0 / len(res_tr)
    w_te = np.ones_like(res_te) * 100.0 / len(res_te)
    ax2.hist(res_tr, bins=40, color=c_tr, alpha=0.55, edgecolor='k',
             linewidth=0.4, weights=w_tr)
    ax2.hist(res_te, bins=40, color=c_te, alpha=0.55, edgecolor='k',
             linewidth=0.4, weights=w_te)
    ax2.axvline(0, color='gray', linestyle='--', lw=1.0)

    # 共享背景框（透明文字，仅提供边框和白底）
    ax2.text(0.95, 0.9,
             r'Train: $\mu=$' f'{res_tr.mean():.2f}' r', $\sigma=$' f'{res_tr.std():.2f}' '\n'
             r'Test: $\mu=$' f'{res_te.mean():.2f}' r', $\sigma=$' f'{res_te.std():.2f}',
             transform=ax2.transAxes, ha='right', va='top', fontsize=6, color='none',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='#999999', linewidth=0.8))
    # Train 行
    ax2.text(0.95, 0.9,
             r'Train: $\mu=$' f'{res_tr.mean():.2f}' r', $\sigma=$' f'{res_tr.std():.2f}',
             transform=ax2.transAxes, ha='right', va='top', fontsize=6, color=c_tr)
    # Test 行（向下偏移一行）
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
    fig3, ax3 = plt.subplots(2, 1, figsize=(8.5/3, 8.5/3 * 1.5))
    ax3[0].scatter(B_tr, res_tr, alpha=0.5, s=14, color='green', label='Train')
    ax3[0].scatter(B_te, res_te, alpha=0.7, s=14, color='orange', label='Test')
    ax3[0].axhline(0, color='red', linestyle='--', lw=1.0)
    ax3[0].set_xlabel(r'$B$ (mT)', fontsize=10)

    ax3[1].scatter(f_tr, res_tr, alpha=0.5, s=14, color='#ef4444', label='Train')
    ax3[1].scatter(f_te, res_te, alpha=0.7, s=14, color='#3b82f6', label='Test')
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
    W = 8.5 / 3        # 固定宽度
    H = 2              # 高度可调，按需修改这个值
    fig4, ax4 = plt.subplots(figsize=(W, H))
    lo, hi = 0, 1500
    ticks = np.arange(0, 1501, 250)
    line = np.array([lo, hi])
    ax4.plot(line, line, color='#c0392b', linestyle='--', lw=1.2, zorder=3,
             label=r'$\hat{P}_v = P_v$')
    ax4.scatter(meas_te, pred_te_phys, s=14, alpha=0.55,
                color='#3b6fc4', edgecolor='#2c4f8a', linewidth=0.4, zorder=4)
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
    # 7. PARAMETER EVOLUTION DURING TRAINING (Phase 1 + Phase 2)
    # ==========================================
    epochs_arr = np.array(param_history['epoch'])

    # === 整张图宽度改为 Letter 的 2/3 (8.5 * 2/3 ≈ 5.67 inch) ===
    fig_w = 8.5 * 2 / 3
    figp, axp = plt.subplots(1, 2, figsize=(fig_w, fig_w / 2 * 0.7))
    
    # IEEE / 色盲友好调色板
    ieee_colors = [
        '#000000',  # black
        '#E69F00',  # orange
        '#56B4E9',  # sky blue
        '#009E73',  # bluish green
        '#0072B2',  # blue
        '#D55E00',  # vermillion
        '#CC79A7',  # reddish purple
        '#882255',  # wine
        '#44AA99',  # teal
        '#117733',  # dark green
    ]
    
    # ---------- Left: 线性系数 ξ ----------
    # 只保留 xi_1, xi_2, xi_4 的颜色，其余灰色 ("Pruned")
    xi_colors = [
        '#EE3377',  # magenta
        '#0077BB',  # blue
        '#EE7733',  # orange
        '#009988',  # teal
        '#CC3311',  # red
        '#33BBEE',  # cyan
        '#BBBB00',  # olive/yellow-green
        '#AA3377',  # purple
        '#004488',  # dark blue
        '#000000',  # black
    ]
    GREY = '#bbbbbb'
#    keep_xi = {0: r'$\xi_1$', 1: r'$\xi_2$', 3: r'$\xi_4$'}  # 0-based 索引
    keep_xi = {0: r'$\xi_1$', 1: r'$\xi_2$', 3: r'$\xi_4$', 8: r'$\xi_9$'}  # 0-based 索引
    
    for i in range(10):
        vals = np.array(param_history[f'coef[{i}]'], dtype=float)
        if i in keep_xi:
            axp[0].plot(epochs_arr, vals, lw=1.2,
                        color=xi_colors[i], linestyle='-', zorder=3)
        else:
            axp[0].plot(epochs_arr, vals, lw=0.8,
                        color=GREY, linestyle='-', zorder=1)
    
    # 在保留曲线末端标注变量名（同色），替代图例
    x_end = epochs_arr[-1]
    for i, lab in keep_xi.items():
        y_end = np.array(param_history[f'coef[{i}]'], dtype=float)[-1]
        axp[0].annotate(lab, xy=(x_end, y_end),
                        xytext=(4, 0), textcoords='offset points',
                        color=xi_colors[i], fontsize=8, va='center', ha='left',
                        fontweight='bold', clip_on=False)
    
    # 灰色被剪枝曲线统一标一个 "Pruned"——放在灰线中间区域的上方、居中
    pruned_idx = [i for i in range(10) if i not in keep_xi]
    x_mid = epochs_arr[len(epochs_arr) // 2]                      # epoch 轴中部
    pruned_at_mid = [np.array(param_history[f'coef[{i}]'], dtype=float)[len(epochs_arr) // 2]
                     for i in pruned_idx]
    y_mid_top = float(np.max(pruned_at_mid))                      # 中部处灰线的最高点
    axp[0].annotate('Others Pruned', xy=(x_mid, y_mid_top),
                    xytext=(0, 5), textcoords='offset points',
                    color=GREY, fontsize=8, va='bottom', ha='center', clip_on=False)
    
    axp[0].axvline(phase1_epochs, color='gray', linestyle=':', lw=0.8)
    axp[0].set_xlabel('Epoch', fontsize=10)
    axp[0].set_ylabel(r'Sparse Coefficients $\mathbf{\Xi}$', fontsize=10)
    # 不再使用图例
    
    # ---------- Right: 自适应参数 Λ ----------
    # 机制 → 颜色
    mech_color = {
        'h':  ieee_colors[1],   # hysteresis
        'e':  ieee_colors[2],   # eddy
        'a':  ieee_colors[3],   # anomalous
        's':  ieee_colors[4],   # saturation
        'r1': ieee_colors[5],   # rolloff 1 (fc, pc)
        'r2': ieee_colors[6],   # rolloff 2 (fc2, pc2)
        'g':  ieee_colors[7],   # gamma_h (单独)
        'd':  ieee_colors[8],   # sat_g / delta (单独)
    }
    # 分量 → 线型
    comp_ls = {
        'f':   '-',             # f 指数 / 截止频率
        'B':   '--',            # B 指数 / 极点阶数
        'solo': '-.',           # 单独标量项
    }
    
    # 每个参数的 (机制, 分量, 图例标签)
    param_style = [
        ('h',  'f', r'$\alpha_h$'),
        ('h',  'B', r'$\beta_h$'),
        ('e',  'f', r'$\alpha_e$'),
        ('e',  'B', r'$\beta_e$'),
        ('a',  'f', r'$\alpha_a$'),
        ('a',  'B', r'$\beta_a$'),
        ('s',  'f', r'$\alpha_s$'),
        ('s',  'B', r'$\beta_s$'),
        ('h',  'solo', r'$\gamma_h$'),
        ('d',  'solo', r'$\delta$'),
        ('r1', 'f', r'$f_{c1}$'),
        ('r1', 'B', r'$p_{c1}$'),
        ('r2', 'f', r'$f_{c2}$'),
        ('r2', 'B', r'$p_{c2}$'),
    ]
    
    # 只保留这些参数（按标签匹配），其余删除
    keep_lambda = {
        r'$\alpha_h$', r'$\beta_h$',
        r'$\alpha_e$', r'$\beta_e$',
        r'$\alpha_s$', r'$\beta_s$',
        r'$\gamma_h$',
        r'$f_{c1}$', r'$f_{c2}$',
        r'$p_{c1}$', r'$p_{c2}$',
    }
    
    # 先画线，收集保留参数的末端坐标
    lam_labels = []
    for j, name in enumerate(exp_names):
        mech, comp, lab = param_style[j]
        if lab not in keep_lambda:
            continue  # 删除未保留的参数
        vals = np.array(param_history[name], dtype=float)
        col = mech_color[mech]
        axp[1].plot(epochs_arr, vals, lw=1.2,
                    color=col, linestyle=comp_ls[comp], zorder=3)
        lam_labels.append([vals[-1], col, lab])
    
    # 末端标注：均匀散开 + 防溢出
    # 先给 y 轴上下各留一点额外空间，避免标签溢出
    y0, y1 = axp[1].get_ylim()
    pad = 0.06 * (y1 - y0)
    axp[1].set_ylim(y0 - pad, y1 + pad)
    y0, y1 = axp[1].get_ylim()

    lam_labels.sort(key=lambda t: t[0])          # 按真实末端值排序
    n = len(lam_labels)
    min_gap = 0.07 * (y1 - y0)                    # 最小垂直间距（越大越散）

    adj = [t[0] for t in lam_labels]             # 标签 y 位置，从真实末端值出发
    lo = y0 + 0.03 * (y1 - y0)
    hi = y1 - 0.03 * (y1 - y0)
    for _ in range(200):
        moved = False
        for k in range(1, n):
            if adj[k] - adj[k - 1] < min_gap:
                shift = (min_gap - (adj[k] - adj[k - 1])) / 2
                adj[k - 1] -= shift
                adj[k] += shift
                moved = True
        adj[0] = max(adj[0], lo)
        adj[-1] = min(adj[-1], hi)
        for k in range(1, n):                    # 夹紧后再保证一次最小间距
            if adj[k] - adj[k - 1] < min_gap:
                adj[k] = adj[k - 1] + min_gap
        if adj[-1] > hi:                         # 顶部溢出则整体下移
            adj = [a - (adj[-1] - hi) for a in adj]
        if not moved:
            break

    for (yt_true, col, lab), y_text in zip(lam_labels, adj):
        axp[1].annotate(lab, xy=(epochs_arr[-1], y_text),
                        xytext=(4, 0), textcoords='offset points',
                        color=col, fontsize=8, va='center', ha='left',
                        clip_on=False)
    
    axp[1].axvline(phase1_epochs, color='gray', linestyle=':', lw=0.8)
    axp[1].set_xlabel('Epoch', fontsize=10)
    axp[1].set_ylabel(r'Raw Parameters $\mathbf{\Lambda}_{\text{raw}}$', fontsize=10)
    # 不再使用图例
    
    for ax in axp:
        ax.tick_params(axis='both', labelsize=8, direction='in', top=True, right=True)
        ax.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)
        ax.set_axisbelow(True)
        # 右侧留出标注空间
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0, x1 + 0.08 * (x1 - x0))
    
    plt.tight_layout()
    figp.savefig('param_evolution.svg', format='svg', bbox_inches='tight')
    figp.savefig('param_evolution.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()