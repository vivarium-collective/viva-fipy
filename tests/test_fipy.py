"""Smoke tests for the FiPy FVM engine primitives + general transport process."""
import pytest

pytest.importorskip("fipy")
from viva_fipy import fipy_classes, grid2d, lu_solver, solve_sweep, FipyTransportProcess  # noqa: E402


def test_fipy_classes_bundle():
    F = fipy_classes()
    for k in ("CellVariable", "Grid2D", "TransientTerm", "DiffusionTerm", "PowerLawConvectionTerm"):
        assert k in F


def test_grid_and_real_solve_diffuses():
    F = fipy_classes()
    mesh = grid2d(nx=10, ny=10, dx=0.1, dy=0.1)
    c = F["CellVariable"](name="c", mesh=mesh, value=0.0, hasOld=True)
    c.constrain(1.0, mesh.facesLeft)        # Dirichlet source on the left
    c.constrain(0.0, mesh.facesRight)
    eq = F["TransientTerm"]() == F["DiffusionTerm"](coeff=1.0)
    solver = lu_solver()
    resid = 0.0
    for _ in range(20):
        resid = solve_sweep(eq, c, dt=0.5, solver=solver)
    import numpy as np
    v = np.asarray(c.value)
    assert float(v.max()) > 0.1            # the left boundary diffused inward
    assert float(resid) >= 0.0             # a finite residual was returned


def test_process_is_configured():
    assert "diffusivity" in FipyTransportProcess.config_schema
