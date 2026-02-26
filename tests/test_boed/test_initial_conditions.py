import numpy as np
import pytest

from boed.core.initial_conditions import (
    U0Spec,
    available_u0,
    get_u0_spec,
    make_u0,
    make_u0_bank,
    u0_registry,
)


def test_available_u0_nonempty():
    assert len(list(available_u0())) > 0


def test_make_u0_shapes_and_finite():
    x = np.linspace(0, 1, 128)
    for kind in available_u0():
        u = make_u0(x, kind)
        assert u.shape == x.shape
        assert np.all(np.isfinite(u))


def test_random_fourier_deterministic():
    x = np.linspace(0, 1, 64)
    u1 = make_u0(x, "random_fourier", seed=1)
    u2 = make_u0(x, "random_fourier", seed=1)
    u3 = make_u0(x, "random_fourier", seed=2)
    assert np.allclose(u1, u2)
    assert not np.allclose(u1, u3)


def test_make_u0_unknown_kind_raises():
    x = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        make_u0(x, "not_a_kind")


def test_registry_metadata_and_bank():
    x = np.linspace(0, 1, 50)
    reg = u0_registry()
    assert "gaussian" in reg
    for name, spec in reg.items():
        assert isinstance(spec, U0Spec)
        assert isinstance(spec.description, str) and spec.description
        assert isinstance(spec.defaults, dict)
        u = spec.build(x)
        assert u.shape == x.shape

    bank = make_u0_bank(x, kinds=["gaussian", "sine"])
    assert set(bank.keys()) == {"gaussian", "sine"}


def test_get_u0_spec():
    spec = get_u0_spec("gaussian")
    assert isinstance(spec, U0Spec)
    assert spec.name == "gaussian"
