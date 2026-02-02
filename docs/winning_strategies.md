Here is the consolidated, **Master Prompt** optimized for the O-Prize execution. It strips away repetition and organizes the requirements into the four strategic pillars derived from your "Mastering the O-Prize" and "Winning Strategies" documents.

***

### **The O-Prize Master Prompt: MCM 2026 Problem A**

**Role:** Act as a Senior Mathematical Modeling Lead targeting an **Outstanding Winner** award. Your objective is to execute the **"Pulse of Power"** solution, ensuring every section moves beyond "standard student modeling" to "rigorous engineering simulation."

**Instruction:** Construct the complete LaTeX framework and Python simulation logic for the Smartphone Battery Dynamics problem by strictly adhering to the following four strategic pillars.

#### **1. The Physics Engine (The Mathematical Core)**
*   **Iterative Complexity:** Implement a two-stage modeling approach to demonstrate depth.
    *   *Model I (Conceptual):* The **Kinetic Battery Model (KiBaM)** (Two-Tank ODE) to explain charge recovery intuition.
    *   *Model II (Rigorous):* The **Electro-Thermal Equivalent Circuit Model (ECM)**.
*   **Governing Equations:** Derive a coupled system of continuous-time Differential Equations:
    *   **Electrical:** Use Kirchhoff’s Voltage Law with a Thevenin Equivalent circuit (2-RC pairs) to capture transient voltage response.
    *   **Thermal:** Use the Heat Equation ($m c_p \frac{dT}{dt}$) to model Joule heating ($I^2R$).
    *   **The Feedback Loop:** Explicitly model **Coupling** using the **Arrhenius Equation**, where Temperature ($T$) dynamically updates Internal Resistance ($R_{int}$) and Capacity ($Q$) at every time step.
    *   **Aging:** Integrate the "Square Root of Time" law ($\sqrt{N}$) to model capacity fade based on cycle history.

#### **2. The User Engine (Stochastic Input Modeling)**
*   **Activity-to-Current Mapping:** Do not use static current values. Construct a "Bridge Model" to convert behavior into Amperes ($I(t)$):
    *   **Display:** Use the Non-linear Gamma Correction model ($P \propto RGB^{2.2}$) for OLED screens.
    *   **Processor:** Use the DVFS equation ($P = C \cdot V^2 \cdot f$) to model cubic power scaling.
    *   **Network (The "Wow" Factor):** Implement the **5G RRC State Machine** (Idle $\to$ Connected $\to$ Tail). Hardcode the specific "Tail Energy" values from *Narayanan et al. (2021)* (~1092 mW active, ~600 mW tail for 12s).
*   **Scenario Simulation:** Simulate 5 distinct **Stochastic Personas** (Gamer, Influencer, Commuter, Chatter, Streamer).
    *   *Requirement:* Mathematically demonstrate the **"Chatty vs. Streaming Paradox"** (proving that low-bandwidth messaging drains more energy than high-bandwidth streaming due to RRC tail accumulation).

#### **3. Validation & Parameter Estimation (The Data Pillar)**
*   **Data Source:** Use **NASA Prognostics** and **CALCE** datasets for Li-Ion discharge curves.
*   **Methodology:** Reject manual tuning. Use **Genetic Algorithms (GA)** to solve the inverse problem: optimize parameters ($R_0, R_1, C_1$) to minimize RMSE against the experimental datasets.
*   **TTE Definition:** Define "Time-to-Empty" as the **First Hitting Time (FHT)** where terminal voltage $V(t)$ crosses the cutoff threshold (3.4V), rather than simply SOC=0.

#### **4. The Scientific Narrative (Paper Structure)**
*   **Visuals:** Generate **System Architecture Diagrams** (not simple flowcharts) showing feedback loops. Use **Sensitivity Heatmaps** (Temperature vs. Load) and **Tornado Plots** for robustness checks.
*   **Structure:** Follow the "O-Prize Template":
    *   *Assumptions:* Use a **Justification Table** citing physics sources for every assumption.
    *   *Sensitivity:* Calculate quantitative sensitivity indices ($S = \frac{\partial Y/Y}{\partial X/X}$).
    *   *Recommendations:* Provide specific, quantified advice (e.g., "Implement 'Eco-5G' to force 4G for transfers < 50MB").
    *   *Memo:* Write a non-technical memo to the manufacturer focusing on thermal management logic.

**Action:** Based on these pillars, generate the full **LaTeX code for the Main Body** and the **Python code for the ECM Simulation**, ensuring all equations are rigorous, all assumptions are justified by physics, and the narrative flow connects user behavior directly to electrochemical dynamics.