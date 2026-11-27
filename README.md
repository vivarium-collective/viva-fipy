# viva-fipy

The genuine **FiPy** finite-volume engine as reusable
[process-bigraph](https://github.com/vivarium-collective/process-bigraph) primitives +
a general reaction-advection-diffusion process — one FVM engine every workspace reuses
instead of re-importing FiPy and re-implementing the solve.

## Two ways to use it

**1. The general process** — a generic 2-D `dc/dt = D∇²c − u·∇c + source`:
```python
from viva_fipy import FipyTransportProcess
p = FipyTransportProcess(config={"nx": 40, "ny": 40, "diffusivity": 1e-9,
                                 "vy": 1e-3, "left": 1.0}, core=core)   # source via the input port
```

**2. The engine primitives** — for a *specialized* solver (coupled fields, buoyancy-driven
velocity, bespoke BCs) that keeps its domain logic:
```python
from viva_fipy import fipy_classes, grid2d, lu_solver, solve_sweep
F = fipy_classes()                       # the FiPy term/variable classes (lazy heavy import)
mesh = grid2d(nx, ny, dx, dy)
solver = lu_solver()                     # direct scipy LU (exact to ~machine epsilon)
eq = F["TransientTerm"]() == F["DiffusionTerm"](coeff=D) - F["PowerLawConvectionTerm"](coeff=u)
residual = solve_sweep(eq, field, dt, solver=solver)
```

Reproducibility: pin `viva-fipy` by tag; a consumer's run-env capture records the exact
version. Plain-version tags (`0.1.0`).
