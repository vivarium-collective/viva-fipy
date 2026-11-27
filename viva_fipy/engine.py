"""The genuine FiPy finite-volume engine as reusable primitives.

Build a mesh, get the direct LU solver, and run the standard
``TransientTerm == DiffusionTerm − PowerLawConvectionTerm (+ source)`` sweep —
so a workspace reuses one FVM engine instead of re-importing FiPy and
re-implementing the solve. A *specialized* solver (coupled fields, buoyancy-driven
velocity, bespoke boundary conditions) keeps that domain logic but assembles its
equations from these primitives; a *generic* single-field reaction-advection-diffusion
problem can use :class:`viva_fipy.processes.FipyTransportProcess` directly.

All FiPy imports are lazy so importing this module never drags in the heavy solver.
"""
from __future__ import annotations


def fipy_classes() -> dict:
    """Lazy bundle of the FiPy classes a solver needs (the heavy import)."""
    from fipy import (CellVariable, FaceVariable, Variable, Grid1D, Grid2D,
                      TransientTerm, DiffusionTerm, PowerLawConvectionTerm)
    return {
        "CellVariable": CellVariable, "FaceVariable": FaceVariable, "Variable": Variable,
        "Grid1D": Grid1D, "Grid2D": Grid2D, "TransientTerm": TransientTerm,
        "DiffusionTerm": DiffusionTerm, "PowerLawConvectionTerm": PowerLawConvectionTerm,
    }


def lu_solver():
    """The direct scipy LU solver (exact to ~machine epsilon), or None if unavailable."""
    try:
        from fipy.solvers.scipy.linearLUSolver import LinearLUSolver
        return LinearLUSolver()
    except Exception:  # pragma: no cover
        return None


def grid2d(nx: int, ny: int, dx: float, dy: float):
    from fipy import Grid2D
    return Grid2D(nx=int(nx), ny=int(ny), dx=float(dx), dy=float(dy))


def solve_sweep(eq, field, dt: float, solver=None, n_sweeps: int = 2) -> float:
    """Sweep ``eq`` for ``field`` over ``dt`` and return the max scaled residual.

    Two sweeps by default (the non-linear PowerLawConvection coupling settles in one);
    uses the direct LU solver when given.
    """
    import numpy as np
    field.updateOld()
    for _ in range(int(n_sweeps)):
        if solver is not None:
            eq.sweep(var=field, dt=float(dt), solver=solver)
        else:
            eq.sweep(var=field, dt=float(dt))
    try:
        rv = eq.justResidualVector(var=field, dt=float(dt))
        scale = max(float(np.max(np.abs(np.asarray(field.value)))), 1e-30)
        return float(np.max(np.abs(np.asarray(rv)))) / scale
    except Exception:
        return 0.0
