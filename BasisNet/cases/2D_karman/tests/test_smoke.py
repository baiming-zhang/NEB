import numpy as np
import torch

from eigenfluid_sss.basisnet.karman2d.postprocess import generalized_symmetric_eig
from eigenfluid_sss.basisnet.karman2d.training import (
    Config,
    OfficialPeelingFields,
    fluid_points,
)


def test_final_defaults():
    cfg = Config()
    assert cfg.modes == 96
    assert cfg.envelope_mode == "localized"
    assert cfg.sampling_mode == "mixed"
    assert cfg.learning_rate == 8.0e-4


def test_forward_shape_and_sampling():
    cfg = Config(modes=3, hidden=8, hidden_layers=2, frequencies=2)
    model = OfficialPeelingFields(cfg)
    points = fluid_points(cfg, 16, 0.35, cfg.seed)
    xy = torch.tensor(points)
    alpha = torch.full((len(points), 1), 0.35)
    assert model(xy, alpha).shape == (16, 3)
    assert np.all((points[:, 0] >= 0) & (points[:, 0] <= 10))


def test_generalized_eigenproblem():
    mass = np.diag([1.0, 2.0, 3.0])
    stiffness = np.diag([3.0, 4.0, 5.0])
    values, vectors = generalized_symmetric_eig(stiffness, mass, 3)
    assert np.all(np.diff(values) >= 0)
    assert np.allclose(vectors.T @ mass @ vectors, np.eye(3), atol=1.0e-10)
