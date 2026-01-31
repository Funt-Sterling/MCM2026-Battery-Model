# MCM 2026 Problem A: Modeling Smartphone Battery Drain

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Mathematical Contest in Modeling (MCM) 2026**  
> Problem A: Smartphone Battery Drain Modeling using Equivalent Circuit Model (ECM)

## 📁 Project Structure

```
Problem_A/
├── src/                          # Source code
│   ├── ecm_model.py              # Equivalent Circuit Model (main model)
│   ├── battery_model.py          # KiBaM + Extended thermal model
│   ├── ecm_visualizations.py     # ECM figure generation
│   ├── visualizations.py         # KiBaM figure generation
│   ├── sensitivity_analysis.py   # OAT & Monte Carlo analysis
│   ├── recommendations.py        # User/OS recommendations
│   └── run_analysis.py           # Main execution script
│
├── figures/                      # Generated visualizations
│   ├── ecm_circuit_diagram.png   # ECM schematic
│   ├── ecm_rc_dynamics.png       # RC transient response
│   ├── ecm_scenario_comparison.png
│   ├── ecm_thermal_feedback.png
│   ├── ecm_aging_effects.png
│   ├── ecm_daily_simulation.png
│   └── figure*.png               # Additional figures
│
├── docs/                         # Documentation
│   ├── ECM_EQUATIONS.md          # All equations (LaTeX)
│   └── README.md                 # Detailed model documentation
│
├── data/                         # Input data
│   ├── 2026_MCM_Problem_A.pdf    # Problem statement
│   └── Formulas.png              # Reference formulas
│
└── .gitignore
```

## 🔬 Model Overview

This project implements an **Equivalent Circuit Model (ECM)** for smartphone battery drain, based on the industry-standard MathWorks Simscape Battery framework.

### Key Equations

**Terminal Voltage (Kirchhoff's Voltage Law):**
```
U = OCV(SOC, T) - I·R₀(SOC, T) - ΔU_RC1 - ΔU_RC2
```

**RC Dynamics:**
```
τₖ · d(ΔU_RCk)/dt + ΔU_RCk = I · Rₖ
```

**State of Charge (Coulomb Counting):**
```
dSOC/dt = -I / (C_aged · 3600)
```

**Thermal Model:**
```
M_th · dT/dt = Q_gen - Q_diss
```

## 🚀 Quick Start

```bash
# Navigate to source directory
cd src

# Run the ECM model
python3 ecm_model.py

# Generate all ECM figures
python3 ecm_visualizations.py

# Run sensitivity analysis
python3 sensitivity_analysis.py

# Generate recommendations
python3 recommendations.py
```

## 📊 Results Summary

| Scenario | Avg Current | Battery Life |
|----------|-------------|--------------|
| Idle | 148 mA | >12 hours |
| Light | 488 mA | 8.11 hours |
| Moderate | 1014 mA | 3.90 hours |
| Heavy | 1522 mA | 2.60 hours |
| Gaming | 1920 mA | 2.06 hours |
| Navigation | 1511 mA | 2.62 hours |

## 📚 References

1. MathWorks Simscape Battery Documentation
2. Tremblay, O. et al. "A Generic Battery Model for Hybrid Electric Vehicles" (2007)
3. Chen, M. & Rincón-Mora, G.A. "Accurate Electrical Battery Model" (2006)

## 👥 Team

MCM 2026 Submission

---
*This project satisfies MCM requirements for continuous-time differential equations, physics-based modeling, and practical recommendations.*
