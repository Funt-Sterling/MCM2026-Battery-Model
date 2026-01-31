# 2026 MCM Problem A: Smartphone Battery Model

## Project Structure

```
Problem_A/
├── battery_model.py          # Core model (KiBaM + Extended with thermal)
├── visualizations.py         # O-Prize style figures
├── sensitivity_analysis.py   # Parameter sensitivity & uncertainty
├── recommendations.py        # User & OS recommendations
├── run_analysis.py           # Main execution script
└── README.md                 # This file
```

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib seaborn

# Run complete analysis
python run_analysis.py
```

## Model Overview

### Basic Model: Kinetic Battery Model (KiBaM)
The battery is modeled as two charge reservoirs:
- **q₁**: Available charge (directly powers the device)
- **q₂**: Bound charge (chemical reservoir, slowly replenishes q₁)

$$\frac{dq_1}{dt} = -I(t) + k\left(q_2 - q_1\frac{1-c}{c}\right)$$

$$\frac{dq_2}{dt} = -k\left(q_2 - q_1\frac{1-c}{c}\right)$$

### Extended Model: KiBaM + Thermal + Peukert + Aging

State vector: `[q₁, q₂, T]`

Additional physics:
1. **Arrhenius temperature correction** for effective capacity
2. **Peukert's Law** for rate-dependent losses
3. **Thermal dynamics** with self-heating feedback
4. **Aging model** for capacity fade over cycles

## Generated Figures

| Figure | Description | Paper Section |
|--------|-------------|---------------|
| figure1_kibam_schematic | Two-tank KiBaM diagram | Section 3 |
| figure2_thermal_feedback | Thermal feedback loop | Section 4 |
| figure3_model_comparison | Linear vs Our Model | Section 5 |
| figure4_temperature_effects | Temperature impact | Section 5 |
| figure5_sensitivity_heatmap | 2D sensitivity map | Section 6 |
| figure6_usage_scenarios | Real-world scenarios | Section 5 |
| figure7_recovery_effect | Recovery demonstration | Section 5 |
| figure8_aging_effect | Battery aging | Section 6 |
| figure_tornado | Parameter sensitivity | Section 6 |
| figure_recommendations | User recommendations | Section 7 |

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Q_nom | 4000 mAh | Nominal battery capacity |
| c | 0.625 | KiBaM capacity ratio |
| k | 0.002 /s | KiBaM recovery rate constant |
| n_peukert | 1.05 | Peukert exponent |
| E_a | 0.35 eV | Activation energy |

## References

1. Manwell & McGowan (1993) - Original KiBaM formulation
2. Peukert (1897) - Capacity-rate relationship  
3. Battery University - Li-ion specifications
4. Arrhenius equation for temperature dependence
