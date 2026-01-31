# Equivalent Circuit Model (ECM) - Mathematical Framework

## From MathWorks Simscape Battery Documentation

This document provides the **exact physics equations** from the industry-standard MathWorks Simscape Battery documentation that validate and enhance our smartphone battery drain model.

---

## 1. Terminal Voltage Equation (Kirchhoff's Voltage Law)

The terminal voltage of a Li-ion cell is given by:

$$U = V_{OC}(SOC, T) - \eta_{inst} - \eta_{dyn}$$

Where:
- $V_{OC}(SOC, T)$ = Open Circuit Voltage (lookup table function)
- $\eta_{inst} = I \cdot R_0(SOC, T)$ = Instantaneous overpotential (ohmic)
- $\eta_{dyn} = \sum_{k=1}^{n} \Delta U_{RC,k}$ = Dynamic overpotential (RC pairs)

**Expanded form with two RC pairs:**
$$U = V_{OC}(SOC, T) - I \cdot R_0(SOC, T) - \Delta U_{RC,1} - \Delta U_{RC,2}$$

---

## 2. RC Dynamics (First-Order ODEs)

Each RC pair models transient polarization effects:

$$\tau_k(SOC, T) \frac{d(\Delta U_{RC,k})}{dt} + \Delta U_{RC,k} = I \cdot R_k(SOC, T)$$

Where $\tau_k = R_k \cdot C_k$ is the time constant.

**Rearranged for ODE solver:**
$$\frac{d(\Delta U_{RC,k})}{dt} = \frac{I \cdot R_k - \Delta U_{RC,k}}{\tau_k}$$

### Time Constants (from MathWorks defaults):
| RC Pair | Physical Meaning | Time Constant |
|---------|------------------|---------------|
| RC1 | Activation polarization | τ₁ ≈ 30-100 s |
| RC2 | Concentration polarization | τ₂ ≈ 200-500 s |

---

## 3. State of Charge (Coulomb Counting)

SOC is governed by integral calculus via Coulomb counting:

$$\frac{dSOC}{dt} = -\frac{I}{C_{aged} \cdot 3600}$$

Where:
- $I$ = Current [A], positive for discharge
- $C_{aged}$ = Aged battery capacity [Ah]
- 3600 = conversion from Ah to As (Coulombs)

**Integrated form:**
$$SOC(t) = SOC_0 - \frac{1}{C_{aged}} \int_0^t I(\tau) d\tau$$

---

## 4. Thermal Model (Lumped Mass Heat Equation)

The battery temperature is governed by:

$$M_{th} \frac{dT}{dt} = Q_{gen} - Q_{diss}$$

### Heat Generation:
$$Q_{gen} = \underbrace{I^2 R_0}_{Ohmic} + \underbrace{\sum_k |\Delta U_{RC,k} \cdot I|}_{RC} + \underbrace{I \cdot T \cdot \frac{dV_{OC}}{dT}}_{Reversible}$$

### Heat Dissipation (Newton's Cooling):
$$Q_{diss} = \frac{T - T_{amb}}{R_{th}}$$

Where:
- $M_{th}$ = Thermal mass [J/K]
- $R_{th}$ = Thermal resistance to ambient [K/W]
- $T_{amb}$ = Ambient temperature [K]

---

## 5. Temperature-Dependent Parameters

### Arrhenius Relationship for Resistance:
$$R(T) = R_{ref} \cdot \exp\left[\alpha \left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right]$$

Simplified:
$$R(T) \approx R_{ref} \cdot \exp\left[\beta (T_{ref} - T)\right]$$

Where $\beta \approx 0.02$ K⁻¹ for typical Li-ion.

### Temperature Effect on Capacity:
$$C(T) = C_{ref} \cdot \left[1 - \gamma \cdot (T_{ref} - T)\right]$$

Where $\gamma \approx 0.005$ K⁻¹.

---

## 6. Aging Model (√N Relationship)

### Cycling Aging (Capacity Fade):
$$C_{faded} = C \cdot \left(1 + \frac{\delta_C}{100} \sqrt{\frac{n}{N_{ref}}}\right)$$

Where:
- $\delta_C \approx -20\%$ (capacity decreases)
- $N_{ref} = 500$ cycles (reference)
- $n$ = current cycle count

### Cycling Aging (Resistance Growth):
$$R_{faded} = R \cdot \left(1 + \frac{\delta_R}{100} \sqrt{\frac{n}{N_{ref}}}\right)$$

Where $\delta_R \approx +30\%$ (resistance increases).

### Interpretation:
- After 500 cycles: ~80% capacity, ~130% resistance
- After 1000 cycles: ~72% capacity, ~142% resistance

---

## 7. Complete State-Space Formulation

The ECM can be written in standard ODE form:

**State vector:** $\mathbf{y} = [SOC, \Delta U_{RC,1}, \Delta U_{RC,2}, T]^T$

**Governing equations:**
$$\frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}, I(t), t)$$

Where:
$$\frac{d}{dt}\begin{bmatrix} SOC \\ \Delta U_{RC,1} \\ \Delta U_{RC,2} \\ T \end{bmatrix} = \begin{bmatrix} -\frac{I}{C_{aged} \cdot 3600} \\ \frac{I \cdot R_1 - \Delta U_{RC,1}}{\tau_1} \\ \frac{I \cdot R_2 - \Delta U_{RC,2}}{\tau_2} \\ \frac{Q_{gen} - Q_{diss}}{M_{th}} \end{bmatrix}$$

---

## 8. Bridge Model: User Activity → Current I(t)

The current function connects user behavior to the physics:

$$I(t) = I_{base} + I_{screen}(b, r) + I_{CPU}(u) + I_{network}(m, s) + I_{GPS} + I_{sensors}$$

Where:
- $b$ = brightness [0,1]
- $r$ = refresh rate [Hz]
- $u$ = CPU utilization [0,1]
- $m$ = network mode (WiFi, 4G, 5G)
- $s$ = signal strength [0,1]

### Component Models:
$$I_{screen} = I_{min} + (I_{max} - I_{min}) \cdot b \cdot \left(\frac{r}{60}\right)^{0.3}$$
$$I_{CPU} = I_{idle} + (I_{max,CPU} - I_{idle}) \cdot u$$
$$I_{network} = f(m, s) \quad \text{(lookup table)}$$

---

## 9. Key Parameter Values (MathWorks + Smartphone Adaptation)

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Nominal Capacity | $C_{nom}$ | 4.0 | Ah |
| Nominal Voltage | $V_{nom}$ | 3.7 | V |
| Cutoff Voltage | $V_{min}$ | 3.0 | V |
| Max Voltage | $V_{max}$ | 4.2 | V |
| Internal Resistance | $R_0$ | 85 | mΩ |
| RC1 Resistance | $R_1$ | 29 | mΩ |
| RC1 Time Constant | $\tau_1$ | 36 | s |
| RC2 Resistance | $R_2$ | 20 | mΩ |
| RC2 Time Constant | $\tau_2$ | 300 | s |
| Thermal Mass | $M_{th}$ | 45 | J/K |
| Thermal Resistance | $R_{th}$ | 8 | K/W |
| Ref Cycles | $N_{ref}$ | 500 | - |
| Capacity Fade | $\delta_C$ | -20 | % |
| Resistance Growth | $\delta_R$ | +30 | % |

---

## 10. References (from MathWorks Documentation)

1. Tremblay, O., L.-A. Dessaint, and A.-I. Dekkiche. "A Generic Battery Model for the Dynamic Simulation of Hybrid Electric Vehicles." *IEEE Vehicle Power and Propulsion Conference*, 2007.

2. Nikolian, A., et al. "Classification of Electric Model Based Battery Characterization Methods." *2014 IEEE Vehicle Power and Propulsion Conference*.

3. Einhorn, M., et al. "Comparison, Selection, and Parameterization of Electrical Battery Models for Automotive Applications." *IEEE Transactions on Power Electronics*, 28(3):1429-1437, 2013.

4. Chen, M., and G.A. Rincón-Mora. "Accurate Electrical Battery Model Capable of Predicting Runtime and I-V Performance." *IEEE Transactions on Energy Conversion*, 21(2):504-511, 2006.

5. Safari, M., M. Morcrette, A. Teyssot, and C. Delacourt. "Multimodal Physics-Based Aging Model for Life Prediction of Li-Ion Batteries." *Journal of The Electrochemical Society*, 156(3):A145-A153, 2009.

---

## 11. Implementation Notes

### Numerical Solver:
- Use **RK45** (Runge-Kutta 4th/5th order) with adaptive step size
- Max step: 30 seconds (for accuracy)
- Events: SOC < 1% or V < V_min (terminal conditions)

### Lookup Tables:
- OCV(SOC): Use linear interpolation between breakpoints
- R(SOC, T): Bilinear interpolation
- τ(SOC): Linear interpolation

### Validation:
- Steady-state: TTE ≈ C/I for constant current
- RC dynamics: 63.2% response at t = τ
- Thermal: ΔT_steady = Q_gen × R_th

---

*This framework satisfies MCM requirements for:*
- ✅ **Continuous-time model** (ODEs)
- ✅ **Differential equations** (4 coupled ODEs)
- ✅ **Integral calculus** (Coulomb counting for SOC)
- ✅ **Physics-based** (not curve fitting)
- ✅ **Temperature effects** (Arrhenius)
- ✅ **Aging effects** (√N relationship)
- ✅ **Industry-standard** (MathWorks documentation)
