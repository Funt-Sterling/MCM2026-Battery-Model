# Visualization Action Plan: O-Prize Quality Polish

This document outlines the specific visualization tasks required to meet the "Judge Ultra Pro Max" criteria for the MCM paper. These visualizations are designed to demonstrate model robustness, data rigor, and dynamic stability.

## 1. The "Must-Have" 3D Surface Plot: Sensitivity Analysis

**Goal:** Visually prove coupled non-linear interactions (Arrhenius Degradation). A 2D plot is insufficient.
**File:** `src/visualizations.py` (New function: `figure9_sensitivity_3d_surface`)

### Specification
-   **Type:** 3D Surface Plot (`ax.plot_surface`)
-   **X-axis:** Ambient Temperature ($0^{\circ}$C to $45^{\circ}$C)
-   **Y-axis:** User Load / C-rate (Idle to Gaming)
-   **Z-axis:** Battery Cycle Life (or Capacity Fade %)
-   **Visual Impact:** Flat surface at low temp/load, sharp "drop off" (cliff) at high temp/load.
-   **Winning Precedent:** Team 2407038 (Submersibles) - Figures 14, 15.

### Implementation Logic
1.  **Grid Generation:** Create meshgrids for Temperature (0-45) and Load (0.1C - 2.0C).
2.  **Z-Data Calculation:** Use `kibam_model` or `ecm_model` to simulate degradation rate. If full simulation is too slow, use the analytical Arrhenius factor combined with C-rate impact:
    $$ \text{Degradation} \propto e^{\frac{-E_a}{RT}} \times (1 + k \cdot I_{load}) $$
3.  **Styling:** Use `cmap='viridis'` or `plasma`. Add lighting effects/shading for depth.

---

## 2. The "Correlation Heatmap": Input Parameter Logic

**Goal:** Prove "Data Cleaning" and understanding of input significance.
**File:** `src/visualizations.py` (New function: `figure10_input_correlation`)

### Specification
-   **Type:** Correlation Matrix Heatmap (`ax.imshow` or `sns.heatmap`)
-   **Variables:**
    1.  Screen Brightness
    2.  CPU Frequency
    3.  5G Signal Strength
    4.  Ambient Temperature
    5.  **Target:** Battery Life (Time-to-Empty)
-   **Key Insight:** Show strong negative correlation between **5G Signal Strength** and **Battery Life** (validating "Tail Energy" theory).
-   **Winning Precedent:** Team 2418251 (Tennis) - Figure 8; Team 2406324 ($10 \times 10$ matrix).

### Implementation Logic
1.  **Data Generation:** Run a Monte Carlo simulation (e.g., N=100 samples) varying the input parameters randomly.
2.  **Compute Correlation:** Calculate Pearson correlation coefficients between inputs and the output (Battery Life).
3.  **Visualization:** Plot the matrix. Annotate cells with correlation values. Use a diverging colormap (e.g., `coolwarm`) where Red = Negative Correlation (bad for battery), Blue = Positive.

---

## 3. The "Phase Plane" Trajectory: Visualizing Stability

**Goal:** Show dynamic stability and thermal feedback loops better than time-series.
**File:** `src/visualizations.py` (New function: `figure11_thermal_phase_plane`)

### Specification
-   **Type:** Phase Portrait (2D Trajectory)
-   **X-axis:** Temperature ($T$)
-   **Y-axis:** Internal Resistance ($R_0$)
-   **Look:** A spiral or loop.
    -   *Stable:* Spirals inward (cooling works).
    -   *Unstable:* Spirals outward (thermal runaway).
-   **Winning Precedent:** Team 2425397 (Lampreys) - Figure 11.

### Implementation Logic
1.  **Simulation:** Run the `ExtendedBatteryModel` under a high-stress scenario (e.g., Gaming) that pushes thermal limits.
2.  **Plotting:** Plot $R_0(t)$ vs $T(t)$ as a continuous line.
3.  **Annotations:** Add arrows to the line indicating time direction. Mark the "Stable Equilibrium" point.

---

## 4. The "System Dynamics" Causal Loop Diagram

**Goal:** Visualizing "Electrical Dynamics" vs "Thermal Dynamics".
**Tool:** PowerPoint / Visio / Draw.io (Manual Creation recommended for best aesthetics).
**Output:** `figures/system_dynamics_loop.png`

### Specification
-   **Style:** Block diagram (Reference Team 2322687).
-   **Flow:**
    1.  Current (Start)
    2.  $\rightarrow$ Heat Generation ($I^2R$)
    3.  $\rightarrow$ Temperature Rise
    4.  $\rightarrow$ Reaction Rate ($k$) increase
    5.  $\rightarrow$ SEI Growth / Capacity Fade
    6.  Feedback: Temperature Rise $\rightarrow$ Resistance Increase $\rightarrow$ Heat Generation.
-   **Visuals:** Use distinct shapes/colors for Electrical (Blue) vs Thermal (Red) vs Aging (Grey) components.

---

## 5. Paper Polishing Guidelines

**Requirement:** Refine layout and styling.

1.  **No "Page Bridges":** Do not start a new page for every chapter. Flow text continuously. Use `\section` and `\subsection` to separate content, not `\newpage`.
2.  **Image Sizing:**
    -   Avoid full-page images unless absolutely necessary (e.g., massive system diagram).
    -   Wrap text around smaller figures if possible, or use standard top/bottom floats (`[t]`, `[b]`).
    -   "Visual Density": Images should convey complex info compactly.
3.  **Verification:**
    -   Prepare to verify every number and image against the model (Phase 2).
