# UAV Wing Multidisciplinary Design Tool

This project implements a conceptual UAV wing sizing framework integrating:

- XFOIL-based airfoil aerodynamics
- Finite-wing induced drag correction
- Multi-flight-condition constraints
- Structural stress and deflection limits
- Tapered wing geometry
- Mass minimization under constraints

## How to Use

Open the notebook `wing_design_tool.ipynb` and run:

```python
optimize_wing(3.0, 18.0, 14.0, make_plots=True)
