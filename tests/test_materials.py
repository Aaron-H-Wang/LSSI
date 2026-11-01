"""Verify cross-material units, splits, and the shared identification protocol."""
from dataclasses import asdict
from pathlib import Path
import pytest
from lssi.config import load_config
from lssi.data import load_core_loss

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('material,counts', [
    ('fr78',(1056,264)),('fr79',(992,248)),('fr95',(1056,264))])
def test_material_units_and_split(material,counts):
    config = load_config(ROOT/'configs'/f'{material}.yaml')
    dataset = load_core_loss(ROOT/config.data.mat_path,
        freq_scale=config.data.f_scale,flux_scale=config.data.b_scale)
    assert (dataset.n_train,dataset.n_test) == counts
    assert 300 < dataset.f_norm < 400  # working frequency is kHz
    assert 40 < dataset.B_norm < 110   # working flux density is mT


def test_identical_training_schedule_across_materials():
    schedules=[]
    for material in ('fr78','fr79','fr95'):
        schedule=asdict(load_config(ROOT/'configs'/f'{material}.yaml').train)
        schedule.pop('output_dir')
        schedules.append(schedule)
    assert schedules[0] == schedules[1] == schedules[2]
