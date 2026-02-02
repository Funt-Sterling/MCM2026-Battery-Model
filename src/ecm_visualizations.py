"""
=============================================================================
ECM VISUALIZATIONS - Publication-Quality Figures for MCM Paper
=============================================================================
Generates O-Prize caliber figures showing:
1. Equivalent Circuit Diagram with RC pairs
2. RC transient dynamics
3. OCV-SOC curve
4. Model validation comparisons
5. Thermal coupling feedback
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ecm_model import (
    EquivalentCircuitModel, ECMParameters, SmartphoneLoadModel,
    create_scenario_current_func, get_ocv, get_aged_capacity,
    SOC_BREAKPOINTS, OCV_25C, realistic_day_profile
)
import warnings
warnings.filterwarnings('ignore')

# Style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def draw_ecm_circuit_diagram():
    """
    Draw the Equivalent Circuit Model schematic.
    
    Shows: OCV ─┬─ R0 ─┬─ R1║C1 ─┬─ R2║C2 ─┬─ Terminal
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 14)
    ax.set_ylim(-1, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Color scheme
    colors = {
        'ocv': '#2E86AB',
        'R0': '#E94F37',
        'RC1': '#44AF69',
        'RC2': '#F18F01',
        'wire': '#333333',
        'terminal': '#8B4513',
        'ground': '#666666'
    }
    
    # === OCV Source ===
    # Circle for voltage source
    ocv_circle = Circle((1.5, 5), 0.6, fill=False, color=colors['ocv'], linewidth=2.5)
    ax.add_patch(ocv_circle)
    ax.plot([1.5, 1.5], [4.4, 4.7], color=colors['ocv'], linewidth=2)  # minus
    ax.plot([1.3, 1.7], [5.3, 5.3], color=colors['ocv'], linewidth=2)  # plus top
    ax.plot([1.5, 1.5], [5.15, 5.45], color=colors['ocv'], linewidth=2)  # plus vertical
    ax.text(1.5, 3.5, 'OCV(SOC, T)', ha='center', va='top', fontsize=11, 
            color=colors['ocv'], fontweight='bold')
    
    # Wire from OCV to R0
    ax.plot([2.1, 3], [5, 5], color=colors['wire'], linewidth=2)
    
    # === R0 (Instantaneous Resistance) ===
    # Draw resistor (zigzag)
    r0_x = np.array([3, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5, 4.7, 5])
    r0_y = np.array([5, 5, 5.3, 4.7, 5.3, 4.7, 5.3, 4.7, 5, 5])
    ax.plot(r0_x, r0_y, color=colors['R0'], linewidth=2.5)
    ax.text(4, 4.2, r'$R_0$(SOC, T)', ha='center', va='top', fontsize=11,
            color=colors['R0'], fontweight='bold')
    ax.text(4, 3.8, 'Instantaneous\nResistance', ha='center', va='top', fontsize=9,
            color=colors['R0'], style='italic')
    
    # Wire from R0 to RC1
    ax.plot([5, 5.5], [5, 5], color=colors['wire'], linewidth=2)
    
    # === RC1 (First RC pair - fast dynamics) ===
    # R1 on top
    r1_x = np.array([5.5, 5.8, 6, 6.2, 6.4, 6.6, 6.8, 7, 7.2, 7.5])
    r1_y = np.array([6.5, 6.5, 6.8, 6.2, 6.8, 6.2, 6.8, 6.2, 6.5, 6.5])
    ax.plot(r1_x, r1_y, color=colors['RC1'], linewidth=2.5)
    ax.text(6.5, 7.3, r'$R_1$', ha='center', va='bottom', fontsize=11,
            color=colors['RC1'], fontweight='bold')
    
    # C1 on bottom (parallel plates)
    ax.plot([6.2, 6.2], [3.3, 3.9], color=colors['RC1'], linewidth=3)
    ax.plot([6.8, 6.8], [3.3, 3.9], color=colors['RC1'], linewidth=3)
    ax.text(6.5, 2.8, r'$C_1$', ha='center', va='top', fontsize=11,
            color=colors['RC1'], fontweight='bold')
    
    # Connect RC1 parallel
    ax.plot([5.5, 5.5], [5, 6.5], color=colors['RC1'], linewidth=2)  # Left vertical to R1
    ax.plot([5.5, 5.5], [5, 3.6], color=colors['RC1'], linewidth=2)  # Left vertical to C1
    ax.plot([5.5, 6.2], [3.6, 3.6], color=colors['RC1'], linewidth=2)  # Bottom to C1
    ax.plot([6.8, 7.5], [3.6, 3.6], color=colors['RC1'], linewidth=2)  # C1 to right
    ax.plot([7.5, 7.5], [3.6, 6.5], color=colors['RC1'], linewidth=2)  # Right vertical
    ax.plot([7.5, 7.5], [5, 6.5], color=colors['RC1'], linewidth=2)
    
    # Junction dot
    ax.plot(5.5, 5, 'o', color=colors['RC1'], markersize=6)
    ax.plot(7.5, 5, 'o', color=colors['RC1'], markersize=6)
    
    # RC1 label
    ax.text(6.5, 1.8, r'$\tau_1 = R_1 C_1 \approx 36$s', ha='center', fontsize=10,
            color=colors['RC1'], style='italic')
    ax.text(6.5, 1.3, '(Activation)', ha='center', fontsize=9, color=colors['RC1'])
    
    # Wire from RC1 to RC2
    ax.plot([7.5, 8.5], [5, 5], color=colors['wire'], linewidth=2)
    
    # === RC2 (Second RC pair - slow dynamics) ===
    # R2 on top
    r2_x = np.array([8.5, 8.8, 9, 9.2, 9.4, 9.6, 9.8, 10, 10.2, 10.5])
    r2_y = np.array([6.5, 6.5, 6.8, 6.2, 6.8, 6.2, 6.8, 6.2, 6.5, 6.5])
    ax.plot(r2_x, r2_y, color=colors['RC2'], linewidth=2.5)
    ax.text(9.5, 7.3, r'$R_2$', ha='center', va='bottom', fontsize=11,
            color=colors['RC2'], fontweight='bold')
    
    # C2 on bottom
    ax.plot([9.2, 9.2], [3.3, 3.9], color=colors['RC2'], linewidth=3)
    ax.plot([9.8, 9.8], [3.3, 3.9], color=colors['RC2'], linewidth=3)
    ax.text(9.5, 2.8, r'$C_2$', ha='center', va='top', fontsize=11,
            color=colors['RC2'], fontweight='bold')
    
    # Connect RC2 parallel
    ax.plot([8.5, 8.5], [5, 6.5], color=colors['RC2'], linewidth=2)
    ax.plot([8.5, 8.5], [5, 3.6], color=colors['RC2'], linewidth=2)
    ax.plot([8.5, 9.2], [3.6, 3.6], color=colors['RC2'], linewidth=2)
    ax.plot([9.8, 10.5], [3.6, 3.6], color=colors['RC2'], linewidth=2)
    ax.plot([10.5, 10.5], [3.6, 6.5], color=colors['RC2'], linewidth=2)
    
    # Junction dots
    ax.plot(8.5, 5, 'o', color=colors['RC2'], markersize=6)
    ax.plot(10.5, 5, 'o', color=colors['RC2'], markersize=6)
    
    # RC2 label
    ax.text(9.5, 1.8, r'$\tau_2 = R_2 C_2 \approx 300$s', ha='center', fontsize=10,
            color=colors['RC2'], style='italic')
    ax.text(9.5, 1.3, '(Concentration)', ha='center', fontsize=9, color=colors['RC2'])
    
    # === Terminal ===
    ax.plot([10.5, 12], [5, 5], color=colors['wire'], linewidth=2)
    ax.plot(12, 5, 'o', color=colors['terminal'], markersize=10)
    ax.plot(12.5, 5, 's', color=colors['terminal'], markersize=12)
    ax.text(12.8, 5, ' U', ha='left', va='center', fontsize=14, 
            color=colors['terminal'], fontweight='bold')
    ax.text(12, 4.2, 'Terminal\nVoltage', ha='center', va='top', fontsize=9,
            color=colors['terminal'])
    
    # === Ground connections ===
    for x in [1.5, 4, 6.5, 9.5]:
        ax.plot([x, x], [0, 0.3], color=colors['ground'], linewidth=2)
        ax.plot([x-0.3, x+0.3], [0.15, 0.15], color=colors['ground'], linewidth=2)
        ax.plot([x-0.2, x+0.2], [0.05, 0.05], color=colors['ground'], linewidth=1.5)
        ax.plot([x-0.1, x+0.1], [-0.05, -0.05], color=colors['ground'], linewidth=1)
    
    # Ground wires
    ax.plot([1.5, 1.5], [0.3, 4.4], color=colors['ground'], linewidth=1.5, linestyle='--', alpha=0.5)
    
    # === Current arrow ===
    ax.annotate('', xy=(11.5, 5.5), xytext=(11, 5.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(11.25, 6, 'I', ha='center', fontsize=12, color='red', fontweight='bold')
    
    # === Title and equation ===
    ax.text(6.5, 8.5, 'Equivalent Circuit Model (ECM) with RC Dynamics', 
            ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Main equation box
    eq_box = FancyBboxPatch((1, -0.7), 12, 1.2, boxstyle="round,pad=0.05",
                            facecolor='#f0f0f0', edgecolor='#333333', linewidth=1.5)
    ax.add_patch(eq_box)
    ax.text(7, -0.1, r'$U = \mathrm{OCV}(\mathrm{SOC}, T) - I \cdot R_0(\mathrm{SOC}, T) - \Delta U_{RC1} - \Delta U_{RC2}$',
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def draw_rc_dynamics():
    """
    Show RC transient dynamics with step response.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    t = np.linspace(0, 600, 1000)  # 10 minutes
    
    # RC time constants
    tau1, tau2 = 36, 300  # seconds
    R1, R2 = 0.029, 0.020  # Ohms
    I = 1.0  # 1A step
    
    # RC dynamics: ΔU = I·R·(1 - e^(-t/τ))
    dU_RC1 = I * R1 * (1 - np.exp(-t / tau1))
    dU_RC2 = I * R2 * (1 - np.exp(-t / tau2))
    dU_total = dU_RC1 + dU_RC2
    
    # === Plot 1: Individual RC responses ===
    ax1 = axes[0, 0]
    ax1.plot(t, dU_RC1 * 1000, 'g-', linewidth=2.5, label=f'RC1 (τ₁={tau1}s)')
    ax1.plot(t, dU_RC2 * 1000, 'orange', linewidth=2.5, label=f'RC2 (τ₂={tau2}s)')
    ax1.axvline(tau1, color='g', linestyle='--', alpha=0.5, label=f'63.2% at τ₁')
    ax1.axvline(tau2, color='orange', linestyle='--', alpha=0.5, label=f'63.2% at τ₂')
    ax1.axhline(R1*1000*0.632, color='g', linestyle=':', alpha=0.5)
    ax1.axhline(R2*1000*0.632, color='orange', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Overpotential [mV]')
    ax1.set_title('RC Dynamics: Step Response', fontweight='bold')
    ax1.legend(loc='center right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 600)
    
    # === Plot 2: Combined response ===
    ax2 = axes[0, 1]
    ax2.fill_between(t, 0, dU_RC1*1000, alpha=0.4, color='green', label='RC1 (fast)')
    ax2.fill_between(t, dU_RC1*1000, dU_total*1000, alpha=0.4, color='orange', label='RC2 (slow)')
    ax2.plot(t, dU_total*1000, 'k-', linewidth=2, label='Total ΔU_dyn')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Dynamic Overpotential [mV]')
    ax2.set_title('Combined Dynamic Overpotential', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 600)
    
    # === Plot 3: OCV vs SOC curve ===
    ax3 = axes[1, 0]
    soc = np.linspace(0, 1, 100)
    ocv_25 = [get_ocv(s, 298.15) for s in soc]
    ocv_0 = [get_ocv(s, 273.15) for s in soc]
    ocv_40 = [get_ocv(s, 313.15) for s in soc]
    
    ax3.fill_between(soc*100, ocv_0, ocv_40, alpha=0.2, color='blue', label='T range')
    ax3.plot(soc*100, ocv_0, 'b--', linewidth=1.5, label='T = 0°C')
    ax3.plot(soc*100, ocv_25, 'k-', linewidth=2.5, label='T = 25°C')
    ax3.plot(soc*100, ocv_40, 'r--', linewidth=1.5, label='T = 40°C')
    
    # Mark key points
    ax3.axhline(3.7, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(3.0, color='red', linestyle='--', alpha=0.5, label='Cutoff (3.0V)')
    ax3.axhline(4.2, color='green', linestyle='--', alpha=0.5, label='Max (4.2V)')
    
    ax3.set_xlabel('State of Charge [%]')
    ax3.set_ylabel('Open Circuit Voltage [V]')
    ax3.set_title('OCV-SOC Characteristic Curve', fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)
    ax3.set_ylim(2.9, 4.3)
    
    # === Plot 4: Complete voltage breakdown ===
    ax4 = axes[1, 1]
    
    # Simulate a realistic discharge
    params = ECMParameters()
    model = EquivalentCircuitModel(params)
    result = model.simulate(
        I_func=lambda t: 800,  # 800mA constant
        t_span=(0, 4*3600),    # 4 hours
        T_amb=298.15
    )
    
    t_h = result['t_hours']
    soc = result['SOC']
    V = result['voltage']
    dU1 = result['dU_RC1']
    dU2 = result['dU_RC2']
    
    # Calculate OCV at each point
    OCV = np.array([get_ocv(s, 298.15) for s in soc])
    R0_drop = OCV - V - dU1 - dU2  # I·R0
    
    ax4.stackplot(t_h, R0_drop*1000, dU1*1000, dU2*1000,
                  labels=['IR₀ (ohmic)', 'ΔU_RC1', 'ΔU_RC2'],
                  colors=['#E94F37', '#44AF69', '#F18F01'], alpha=0.7)
    ax4.set_xlabel('Time [hours]')
    ax4.set_ylabel('Voltage Drop [mV]')
    ax4.set_title('Voltage Loss Breakdown During Discharge', fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def draw_model_comparison():
    """
    Compare ECM predictions across scenarios.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    params = ECMParameters()
    model = EquivalentCircuitModel(params)
    load_model = SmartphoneLoadModel()
    
    # Use distinct, professional colors for each scenario
    scenarios = ['idle', 'light', 'moderate', 'heavy', 'gaming', 'navigation']
    scenario_colors = {
        'idle': '#27AE60',      # Green
        'light': '#3498DB',     # Blue
        'moderate': '#9B59B6',  # Purple
        'heavy': '#E67E22',     # Orange
        'gaming': '#E74C3C',    # Red
        'navigation': '#1ABC9C' # Teal
    }
    
    # === Plot 1: SOC vs Time ===
    ax1 = axes[0, 0]
    for scenario in scenarios:
        color = scenario_colors[scenario]
        I_func = create_scenario_current_func(scenario, load_model)
        result = model.simulate(I_func, t_span=(0, 12*3600), T_amb=298.15)
        ax1.plot(result['t_hours'], result['SOC_percent'], 
                linewidth=2.5, label=scenario.capitalize(), color=color)
    
    ax1.axhline(10, color='#C0392B', linestyle='--', linewidth=2, alpha=0.8, label='Low battery (10%)')
    ax1.axhline(20, color='#F39C12', linestyle=':', linewidth=1.5, alpha=0.6, label='Warning (20%)')
    ax1.set_xlabel('Time [hours]', fontsize=11)
    ax1.set_ylabel('State of Charge [%]', fontsize=11)
    ax1.set_title('Battery Drain Comparison', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', ncol=2, fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 100)
    
    # === Plot 2: Terminal Voltage ===
    ax2 = axes[0, 1]
    for scenario in scenarios:
        color = scenario_colors[scenario]
        I_func = create_scenario_current_func(scenario, load_model)
        result = model.simulate(I_func, t_span=(0, 8*3600), T_amb=298.15)
        ax2.plot(result['t_hours'], result['voltage'], 
                linewidth=2.5, label=scenario.capitalize(), color=color)
    
    ax2.axhline(3.0, color='#C0392B', linestyle='--', linewidth=2, label='Cutoff (3.0V)')
    ax2.axhline(3.3, color='#F39C12', linestyle=':', linewidth=1.5, alpha=0.7, label='Low voltage (3.3V)')
    ax2.set_xlabel('Time [hours]', fontsize=11)
    ax2.set_ylabel('Terminal Voltage [V]', fontsize=11)
    ax2.set_title('Voltage Profile Comparison', fontweight='bold', fontsize=12)
    ax2.legend(loc='lower left', ncol=2, fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 8)
    ax2.set_ylim(2.9, 4.3)
    
    # === Plot 3: Temperature rise ===
    ax3 = axes[1, 0]
    temp_scenarios = ['light', 'moderate', 'heavy', 'gaming']
    for scenario in temp_scenarios:
        color = scenario_colors[scenario]
        I_func = create_scenario_current_func(scenario, load_model)
        result = model.simulate(I_func, t_span=(0, 4*3600), T_amb=298.15)
        ax3.plot(result['t_hours'], result['T_celsius'], 
                linewidth=2.5, label=scenario.capitalize(), color=color)
    
    ax3.axhline(45, color='#C0392B', linestyle='--', linewidth=2, label='Thermal limit (45°C)')
    ax3.axhline(35, color='#E67E22', linestyle=':', linewidth=1.5, alpha=0.7, label='Warm (35°C)')
    ax3.axhline(25, color='#7F8C8D', linestyle=':', linewidth=1.5, alpha=0.5, label='Ambient (25°C)')
    ax3.fill_between([0, 4], 45, 55, alpha=0.15, color='red', label='_nolegend_')
    ax3.set_xlabel('Time [hours]', fontsize=11)
    ax3.set_ylabel('Battery Temperature [°C]', fontsize=11)
    ax3.set_title('Thermal Response', fontweight='bold', fontsize=12)
    ax3.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 4)
    ax3.set_ylim(24, 50)
    
    # === Plot 4: Time-to-Empty bar chart ===
    ax4 = axes[1, 1]
    
    tte_list = []
    for scenario in scenarios:
        I_func = create_scenario_current_func(scenario, load_model)
        result = model.simulate(I_func, t_span=(0, 24*3600), T_amb=298.15)
        tte = result['time_to_empty'] if result['time_to_empty'] else 24
        tte_list.append(tte)
    
    # Use consistent colors for bars
    bar_colors = [scenario_colors[s] for s in scenarios]
    scenario_labels = [s.capitalize() for s in scenarios]
    bars = ax4.barh(scenario_labels, tte_list, color=bar_colors, edgecolor='#333333', linewidth=1.2, height=0.7)
    ax4.set_xlabel('Time to Empty [hours]', fontsize=11)
    ax4.set_title('Battery Life by Usage Scenario', fontweight='bold', fontsize=12)
    ax4.set_xlim(0, max(tte_list) * 1.15)
    
    # Add value labels with background
    for bar, tte in zip(bars, tte_list):
        ax4.text(tte + 0.2, bar.get_y() + bar.get_height()/2,
                f'{tte:.1f}h', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_ylim(-0.5, len(scenarios) - 0.5)
    
    plt.tight_layout()
    return fig


def draw_thermal_feedback_ecm():
    """
    Show thermal feedback in ECM model.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    params = ECMParameters()
    
    # === Plot 1: Temperature effect on R0 ===
    ax1 = axes[0]
    temps = np.linspace(-10, 50, 100)
    T_ref = 25
    
    R0_factor = np.exp(params.R0_temp_coeff * ((273.15 + T_ref) - (273.15 + temps)))
    
    ax1.plot(temps, R0_factor, 'b-', linewidth=2.5)
    ax1.axvline(25, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(temps, R0_factor, 1.0, where=temps<25, alpha=0.3, color='blue')
    ax1.fill_between(temps, R0_factor, 1.0, where=temps>25, alpha=0.3, color='red')
    
    ax1.set_xlabel('Temperature [°C]')
    ax1.set_ylabel('Resistance Factor (vs 25°C)')
    ax1.set_title('Temperature Effect on Internal Resistance', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax1.annotate('Cold: Higher R\n→ More heat\n→ Faster drain', 
                xy=(-5, 1.5), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # === Plot 2: Heat generation sources ===
    ax2 = axes[1]
    
    currents = np.linspace(100, 2000, 100)  # mA
    I_A = currents / 1000
    
    R0 = params.R0_ref
    R1 = params.R1_ref
    R2 = params.R2_ref
    
    # Heat components (steady state approximation)
    Q_ohmic = I_A**2 * R0 * 1000  # mW
    Q_RC1 = I_A**2 * R1 * 1000
    Q_RC2 = I_A**2 * R2 * 1000
    Q_total = Q_ohmic + Q_RC1 + Q_RC2
    
    ax2.stackplot(currents, Q_ohmic, Q_RC1, Q_RC2,
                  labels=['Ohmic (R₀)', 'RC1', 'RC2'],
                  colors=['#E94F37', '#44AF69', '#F18F01'], alpha=0.7)
    ax2.plot(currents, Q_total, 'k-', linewidth=2, label='Total')
    
    ax2.set_xlabel('Current [mA]')
    ax2.set_ylabel('Heat Generation [mW]')
    ax2.set_title('Heat Generation vs Current', fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Mark typical usage points
    usage_currents = [200, 500, 1000, 1500]
    usage_labels = ['Light', 'Moderate', 'Heavy', 'Gaming']
    for I, label in zip(usage_currents, usage_labels):
        Q = (I/1000)**2 * (R0+R1+R2) * 1000
        ax2.plot(I, Q, 'ko', markersize=8)
        ax2.annotate(label, (I, Q+20), ha='center', fontsize=9)
    
    # === Plot 3: Feedback loop diagram ===
    ax3 = axes[2]
    ax3.set_xlim(-1, 5)
    ax3.set_ylim(-1, 5)
    ax3.axis('off')
    ax3.set_title('Thermal Feedback Loop', fontweight='bold')
    
    # Draw boxes
    boxes = [
        (0.5, 3.5, 'Current\nI(t)', '#3498db'),
        (3.0, 3.5, 'Heat Gen\nQ = I²R', '#e74c3c'),
        (3.0, 1.0, 'Temp Rise\ndT/dt', '#f39c12'),
        (0.5, 1.0, 'R₀(T) ↑', '#2ecc71'),
    ]
    
    for x, y, text, color in boxes:
        box = FancyBboxPatch((x-0.6, y-0.5), 1.2, 1.0, boxstyle="round,pad=0.05",
                            facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax3.add_patch(box)
        ax3.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrows = [
        ((1.3, 3.5), (2.2, 3.5)),  # I → Q
        ((3.0, 3.0), (3.0, 1.8)),  # Q → T
        ((2.2, 1.0), (1.3, 1.0)),  # T → R
        ((0.5, 1.8), (0.5, 3.0)),  # R → I (feedback)
    ]
    
    for (x1, y1), (x2, y2) in arrows:
        ax3.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Feedback label
    ax3.text(-0.2, 2.4, 'Feedback', fontsize=9, rotation=90, va='center', 
             style='italic', color='red')
    
    plt.tight_layout()
    return fig


def draw_aging_effects():
    """
    Show aging effects on battery capacity and resistance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # === Plot 1: Capacity fade (√N relationship) ===
    ax1 = axes[0, 0]
    cycles = np.linspace(0, 1500, 100)
    
    params = ECMParameters()
    capacity = []
    for n in cycles:
        p_temp = ECMParameters(n_cycles=n)
        capacity.append(get_aged_capacity(p_temp))
    
    capacity = np.array(capacity)
    cap_percent = (capacity / params.Q_nom) * 100
    
    ax1.plot(cycles, cap_percent, '#2980B9', linewidth=3)
    ax1.axhline(80, color='#C0392B', linestyle='--', linewidth=2, label='80% (EOL)')
    ax1.axhline(70, color='#E67E22', linestyle='--', linewidth=1.5, alpha=0.8, label='70%')
    ax1.axvline(500, color='#7F8C8D', linestyle=':', linewidth=1.5, alpha=0.7, label='500 cycles')
    
    # Shade degraded zone
    ax1.fill_between(cycles, 80, cap_percent, where=cap_percent<80, 
                     alpha=0.25, color='#E74C3C', label='_nolegend_')
    ax1.fill_between(cycles, cap_percent, 105, alpha=0.15, color='#3498DB', label='_nolegend_')
    
    ax1.set_xlabel('Cycle Count', fontsize=11)
    ax1.set_ylabel('Capacity [% of original]', fontsize=11)
    ax1.set_title('Capacity Fade: √N Relationship', fontweight='bold', fontsize=12)
    ax1.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1500)
    ax1.set_ylim(50, 105)
    
    # Add equation with better positioning
    ax1.text(1100, 95, r'$C = C_0 (1 - \delta_C \sqrt{n/N_{ref}})$',
             fontsize=10, ha='center', 
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9E6', alpha=0.9, edgecolor='#F39C12'))
    
    # === Plot 2: Resistance increase ===
    ax2 = axes[0, 1]
    
    R_factor = []
    for n in cycles:
        p_temp = ECMParameters(n_cycles=n)
        from ecm_model import get_aged_resistance_factor
        R_factor.append(get_aged_resistance_factor(p_temp))
    
    R_factor = np.array(R_factor)
    R_percent = R_factor * 100
    
    ax2.plot(cycles, R_percent, '#C0392B', linewidth=3)
    ax2.axhline(100, color='#7F8C8D', linestyle=':', linewidth=1.5, alpha=0.7, label='Baseline (100%)')
    ax2.axhline(130, color='#E67E22', linestyle='--', linewidth=2, label='30% increase')
    ax2.axhline(150, color='#C0392B', linestyle='--', linewidth=2, label='50% increase')
    
    # Shade growth zone
    ax2.fill_between(cycles, 100, R_percent, alpha=0.2, color='#E74C3C', label='_nolegend_')
    
    ax2.set_xlabel('Cycle Count', fontsize=11)
    ax2.set_ylabel('Resistance [% of original]', fontsize=11)
    ax2.set_title('Resistance Growth: √N Relationship', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1500)
    ax2.set_ylim(95, 200)
    
    ax2.text(1100, 175, r'$R = R_0 (1 + \delta_R \sqrt{n/N_{ref}})$',
             fontsize=10, ha='center', 
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9E6', alpha=0.9, edgecolor='#F39C12'))
    
    # === Plot 3: Battery life at different ages ===
    ax3 = axes[1, 0]
    
    cycle_counts = [0, 200, 500, 800, 1000]
    # Use distinct colors instead of colormap
    age_colors = ['#27AE60', '#3498DB', '#9B59B6', '#E67E22', '#C0392B']
    
    for n_cyc, color in zip(cycle_counts, age_colors):
        params_aged = ECMParameters(n_cycles=n_cyc)
        model_aged = EquivalentCircuitModel(params_aged)
        
        result = model_aged.simulate(
            I_func=lambda t: 500,  # 500mA moderate usage
            t_span=(0, 12*3600),
            T_amb=298.15
        )
        
        label = 'New' if n_cyc == 0 else f'{n_cyc} cycles'
        ax3.plot(result['t_hours'], result['SOC_percent'], 
                color=color, linewidth=2.5, label=label)
    
    ax3.axhline(10, color='#C0392B', linestyle='--', linewidth=2, alpha=0.7, label='Critical (10%)')
    ax3.axhline(20, color='#F39C12', linestyle=':', linewidth=1.5, alpha=0.6, label='_nolegend_')
    ax3.set_xlabel('Time [hours]', fontsize=11)
    ax3.set_ylabel('State of Charge [%]', fontsize=11)
    ax3.set_title('Discharge Profile at Different Ages', fontweight='bold', fontsize=12)
    ax3.legend(title='Battery Age', loc='upper right', fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 12)
    ax3.set_ylim(0, 100)
    
    # === Plot 4: Time-to-Empty vs Age ===
    ax4 = axes[1, 1]
    
    cycle_range = np.array([0, 200, 400, 600, 800, 1000])
    tte_age = []
    
    for n_cyc in cycle_range:
        params_aged = ECMParameters(n_cycles=n_cyc)
        model_aged = EquivalentCircuitModel(params_aged)
        
        result = model_aged.simulate(
            I_func=lambda t: 500,
            t_span=(0, 16*3600),
            T_amb=298.15
        )
        
        tte = result['time_to_empty'] if result['time_to_empty'] else 16
        tte_age.append(tte)
    
    # Create color gradient from green to red
    n_bars = len(cycle_range)
    bar_colors = ['#27AE60', '#3498DB', '#9B59B6', '#E67E22', '#E74C3C', '#C0392B'][:n_bars]
    
    bars = ax4.bar(cycle_range, tte_age, width=150, color=bar_colors,
            edgecolor='#333333', linewidth=1.2)
    
    ax4.set_xlabel('Cycle Count', fontsize=11)
    ax4.set_ylabel('Battery Life [hours]', fontsize=11)
    ax4.set_title('Battery Life Degradation Over Time', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xlim(-100, 1100)
    
    # Add percentage labels with better formatting
    tte_new = tte_age[0]
    for i, (cyc, tte, bar) in enumerate(zip(cycle_range, tte_age, bars)):
        pct = (tte / tte_new) * 100
        # Position label above bar with background
        ax4.text(cyc, tte + 0.3, f'{pct:.0f}%', ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Add insight annotation arrow
    ax4.annotate('', xy=(900, tte_age[-1]), xytext=(100, tte_age[0]),
                arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=2, ls='--'))
    
    plt.tight_layout()
    return fig


def draw_realistic_day_simulation():
    """
    Simulate a realistic day of smartphone usage.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    params = ECMParameters()
    model = EquivalentCircuitModel(params)
    
    # Simulate 24 hours
    result = model.simulate(
        I_func=realistic_day_profile(),
        t_span=(0, 24*3600),
        SOC_0=1.0,
        T_amb=298.15
    )
    
    t_h = result['t_hours']
    
    # === Plot 1: SOC and current ===
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    l1 = ax1.plot(t_h, result['SOC_percent'], 'b-', linewidth=2, label='SOC')
    ax1.set_ylabel('State of Charge [%]', color='b')
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='y', labelcolor='b')
    
    l2 = ax1_twin.plot(t_h, result['current'], 'r-', linewidth=1, alpha=0.7, label='Current')
    ax1_twin.set_ylabel('Current [mA]', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Shade different periods
    ax1.axvspan(0, 7, alpha=0.1, color='navy', label='Sleep')
    ax1.axvspan(8, 9, alpha=0.1, color='orange', label='Commute')
    ax1.axvspan(9, 17, alpha=0.1, color='gray', label='Work')
    ax1.axvspan(18, 21, alpha=0.1, color='green', label='Evening')
    
    ax1.set_xlabel('Time of Day [hours]')
    ax1.set_title('Daily Usage Profile - SOC & Current', fontweight='bold')
    ax1.set_xlim(0, 24)
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: Voltage profile ===
    ax2 = axes[0, 1]
    ax2.plot(t_h, result['voltage'], 'g-', linewidth=2)
    ax2.axhline(3.0, color='red', linestyle='--', label='Cutoff (3.0V)')
    ax2.axhline(3.7, color='gray', linestyle=':', alpha=0.5, label='Nominal (3.7V)')
    ax2.fill_between(t_h, 3.0, result['voltage'], where=result['voltage']>3.0,
                     alpha=0.3, color='green')
    
    ax2.set_xlabel('Time of Day [hours]')
    ax2.set_ylabel('Terminal Voltage [V]')
    ax2.set_title('Terminal Voltage Throughout Day', fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 24)
    
    # === Plot 3: Temperature ===
    ax3 = axes[1, 0]
    ax3.plot(t_h, result['T_celsius'], 'orange', linewidth=2)
    ax3.axhline(25, color='gray', linestyle='--', label='Ambient')
    ax3.fill_between(t_h, 25, result['T_celsius'], alpha=0.3, color='orange')
    
    ax3.set_xlabel('Time of Day [hours]')
    ax3.set_ylabel('Battery Temperature [°C]')
    ax3.set_title('Thermal Response', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 24)
    
    # === Plot 4: Power consumption breakdown ===
    ax4 = axes[1, 1]
    
    # Create cumulative energy plot
    dt = np.diff(result['t'])
    dt = np.append(dt, dt[-1])
    energy_cumulative = np.cumsum(result['power_mW'] * dt / 3600 / 1000)  # Wh
    
    ax4.fill_between(t_h, 0, energy_cumulative, alpha=0.5, color='purple')
    ax4.plot(t_h, energy_cumulative, 'purple', linewidth=2)
    
    ax4.set_xlabel('Time of Day [hours]')
    ax4.set_ylabel('Cumulative Energy [Wh]')
    ax4.set_title('Energy Consumption Over Day', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 24)
    
    # Add total energy text
    total_energy = energy_cumulative[-1]
    ax4.text(12, total_energy*0.5, f'Total: {total_energy:.1f} Wh\n({total_energy*1000:.0f} mWh)',
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def draw_thermal_runaway_phase_plane():
    """
    O-Prize Enhancement: Thermal Runaway Phase Plane Plot.

    This is the mathematically elegant visualization that shows:
    - X-axis: Temperature (T)
    - Y-axis: Heat rates (Q_gen = I²R and Q_cool = hA(T - T_amb))
    - Where lines cross = stable operating point
    - If Q_gen > Q_cool at high T, lines diverge → thermal runaway

    This figure visualizes the "invisible" electro-thermal feedback loop.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Temperature range
    T = np.linspace(10, 60, 100)  # °C
    T_amb = 25  # Ambient temperature

    # Thermal parameters (from paper)
    hA = 0.12  # W/K (heat dissipation coefficient)
    E_a = 0.3  # eV (Arrhenius activation energy)
    k_B = 8.617e-5  # eV/K (Boltzmann constant)
    T_ref = 25 + 273.15  # K (reference temperature)

    # Current scenarios
    I_gaming = 1.8   # High load (gaming) - Amps
    I_normal = 0.5   # Normal load - Amps
    I_idle = 0.15    # Idle - Amps

    # Base resistance at 25°C
    R_ref = 0.030  # Ohms

    # Calculate temperature-dependent resistance (Arrhenius)
    T_kelvin = T + 273.15
    R_T = R_ref * np.exp((E_a / k_B) * (1/T_ref - 1/T_kelvin))

    # Heat generation for different scenarios
    Q_gen_gaming = I_gaming**2 * R_T
    Q_gen_normal = I_normal**2 * R_T
    Q_gen_idle = I_idle**2 * R_T

    # Heat dissipation (linear with temperature)
    Q_cool = hA * (T - T_amb)

    # Plot heat generation curves
    ax.plot(T, Q_gen_gaming, 'r-', linewidth=2.5, label=f'$Q_{{gen}}$ Gaming (I={I_gaming}A)')
    ax.plot(T, Q_gen_normal, 'orange', linewidth=2.5, label=f'$Q_{{gen}}$ Normal (I={I_normal}A)')
    ax.plot(T, Q_gen_idle, 'green', linewidth=2.5, label=f'$Q_{{gen}}$ Idle (I={I_idle}A)')

    # Plot heat dissipation line
    ax.plot(T, Q_cool, 'b--', linewidth=3, label=f'$Q_{{cool}} = hA(T-T_{{amb}})$')

    # Fill dangerous region
    danger_mask = Q_gen_gaming > Q_cool
    ax.fill_between(T, Q_gen_gaming, Q_cool, where=danger_mask,
                    alpha=0.3, color='red', label='Runaway Risk Zone')

    # Find equilibrium points (where Q_gen = Q_cool)
    # For gaming scenario
    idx_gaming = np.argmin(np.abs(Q_gen_gaming - Q_cool))
    T_eq_gaming = T[idx_gaming]
    Q_eq_gaming = Q_cool[idx_gaming]

    # For normal scenario
    idx_normal = np.argmin(np.abs(Q_gen_normal - Q_cool))
    T_eq_normal = T[idx_normal]
    Q_eq_normal = Q_cool[idx_normal]

    # Mark equilibrium points
    ax.scatter([T_eq_gaming], [Q_eq_gaming], s=150, c='red', marker='o', zorder=5,
               edgecolors='black', linewidths=2)
    ax.scatter([T_eq_normal], [Q_eq_normal], s=150, c='orange', marker='o', zorder=5,
               edgecolors='black', linewidths=2)

    # Annotate equilibrium points
    # Annotate equilibrium points with VISIBLE text (black with background box)
    ax.annotate(f'Gaming Equilibrium\n(T={T_eq_gaming:.0f}°C)',
                xy=(T_eq_gaming, Q_eq_gaming),
                xytext=(T_eq_gaming + 7, Q_eq_gaming + 0.06),
                fontsize=10, fontweight='bold', color='black',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFCDD2', edgecolor='darkred', alpha=0.95))

    ax.annotate(f'Normal Equilibrium\n(T={T_eq_normal:.0f}°C)',
                xy=(T_eq_normal, Q_eq_normal),
                xytext=(T_eq_normal - 12, Q_eq_normal + 0.05),
                fontsize=10, fontweight='bold', color='black',
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE0B2', edgecolor='darkorange', alpha=0.95))

    # Add operating region shading with labels in legend
    ax.axvspan(10, 25, alpha=0.15, color='blue', label='Cold (Reduced Efficiency)')
    ax.axvspan(25, 40, alpha=0.15, color='green', label='Optimal (25-40°C)')
    ax.axvspan(40, 45, alpha=0.2, color='yellow', label='Caution Zone')
    ax.axvspan(45, 60, alpha=0.2, color='red', label='Danger Zone')

    # Throttle line with better visibility
    ax.axvline(42, color='purple', linestyle=':', linewidth=2.5, label='Throttle Threshold')
    ax.text(44, 0.22, 'Thermal\nThrottling\nActivates', fontsize=10, color='black',
            fontweight='bold', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E1BEE7', edgecolor='purple', alpha=0.95))

    # Physics explanation box - move to avoid overlap
    physics_text = (
        "Stability Criterion:\n"
        "• If $Q_{gen} < Q_{cool}$: System cools → Stable\n"
        "• If $Q_{gen} > Q_{cool}$: System heats → Runaway risk\n"
        "• Equilibrium: $I^2 R(T) = hA(T - T_{amb})$"
    )
    ax.text(0.02, 0.98, physics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=1.5))

    # Labels and title
    ax.set_xlabel('Battery Temperature (°C)', fontsize=12)
    ax.set_ylabel('Heat Rate (W)', fontsize=12)
    ax.set_title('Thermal Runaway Phase Plane: Heat Generation vs. Dissipation\n'
                 '(The Electro-Thermal Feedback Loop Visualized)',
                 fontsize=13, fontweight='bold')

    # Legend outside plot to avoid overlap
    ax.legend(loc='upper center', fontsize=9, ncol=4, bbox_to_anchor=(0.5, -0.10))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(10, 60)
    ax.set_ylim(0, max(Q_gen_gaming) * 1.15)

    plt.tight_layout()
    return fig


def draw_generalization_foldable():
    """
    O-Prize Enhancement: Model Generalization to Foldable Devices.

    Tests the ECM framework on Samsung Galaxy Z Fold5 parameters
    to demonstrate model universality (Golden Rule #4).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Standard phone parameters
    params_standard = {
        'name': 'iPhone 15 Pro',
        'Q_nom': 3.274,  # Ah
        'M_th': 38,      # J/K
        'P_display_max': 1800,  # mW
        'hA': 0.12       # W/K
    }

    # Foldable parameters (Z Fold5: 7.6" screen, larger thermal mass)
    params_fold = {
        'name': 'Samsung Z Fold5',
        'Q_nom': 4.4,    # Ah
        'M_th': 72,      # J/K (larger device)
        'P_display_max': 2800,  # mW (7.6" OLED)
        'hA': 0.18       # W/K (larger surface area)
    }

    # Simulation time (24 hours)
    t = np.linspace(0, 24, 1440)  # 1 minute resolution

    def simulate_day(params):
        """Simulate battery drain over a day."""
        Q = params['Q_nom']
        soc = np.ones_like(t)
        temp = np.ones_like(t) * 25  # Start at ambient

        # Power profile (varies by time of day)
        power = np.zeros_like(t)
        for i, hour in enumerate(t):
            if 0 <= hour < 7:
                power[i] = 200  # Idle overnight
            elif 7 <= hour < 9:
                power[i] = 1200  # Morning use
            elif 9 <= hour < 12:
                power[i] = 400   # Work (occasional check)
            elif 12 <= hour < 13:
                power[i] = 2000  # Lunch streaming
            elif 13 <= hour < 17:
                power[i] = 400   # Work
            elif 17 <= hour < 20:
                power[i] = 2500  # Evening heavy use
            elif 20 <= hour < 22:
                power[i] = 1500  # Gaming/streaming
            else:
                power[i] = 300   # Wind down

        # Scale by display for foldable
        if params['P_display_max'] > 2000:
            power = power * 1.3  # 30% more for larger screen

        # Drain calculation
        for i in range(1, len(t)):
            dt = (t[i] - t[i-1]) * 3600  # seconds
            energy_mwh = power[i] * dt / 3600
            soc[i] = soc[i-1] - energy_mwh / (Q * 3700)  # Q in Ah, 3.7V nominal
            soc[i] = max(0, soc[i])

            # Thermal dynamics
            Q_gen = (power[i] / 1000) * 0.15  # 15% of power as heat
            Q_cool = params['hA'] * (temp[i-1] - 25)
            dT = (Q_gen - Q_cool) / params['M_th'] * dt
            temp[i] = temp[i-1] + dT
            temp[i] = np.clip(temp[i], 20, 50)

        return soc * 100, temp, power

    # Run simulations
    soc_std, temp_std, power_std = simulate_day(params_standard)
    soc_fold, temp_fold, power_fold = simulate_day(params_fold)

    # Plot 1: SOC comparison
    ax1 = axes[0]
    ax1.plot(t, soc_std, 'b-', linewidth=2, label=params_standard['name'])
    ax1.plot(t, soc_fold, 'r-', linewidth=2, label=params_fold['name'])

    ax1.axhline(20, color='gray', linestyle='--', alpha=0.5)
    ax1.text(12, 22, '20% Low Battery', ha='center', fontsize=9, color='gray')

    ax1.set_xlabel('Time of Day (hours)', fontsize=11)
    ax1.set_ylabel('State of Charge (%)', fontsize=11)
    ax1.set_title('Battery Life Comparison: Standard vs. Foldable', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, 100)

    # Find battery life (time to 20%)
    life_std = t[np.argmax(soc_std < 20)] if np.any(soc_std < 20) else 24
    life_fold = t[np.argmax(soc_fold < 20)] if np.any(soc_fold < 20) else 24

    # Stagger annotations to avoid overlap
    ax1.annotate(f'Battery life: {life_std:.1f}h', xy=(life_std, 20),
                 xytext=(life_std-4, 42), fontsize=10, color='blue', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#BBDEFB', edgecolor='blue', alpha=0.9))
    ax1.annotate(f'Battery life: {life_fold:.1f}h', xy=(life_fold, 20),
                 xytext=(life_fold+2, 32), fontsize=10, color='red', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFCDD2', edgecolor='red', alpha=0.9))

    # Plot 2: Temperature comparison
    ax2 = axes[1]
    ax2.plot(t, temp_std, 'b-', linewidth=2, label=params_standard['name'])
    ax2.plot(t, temp_fold, 'r-', linewidth=2, label=params_fold['name'])

    ax2.axhline(42, color='purple', linestyle=':', linewidth=2, label='Throttle Threshold')
    ax2.axhspan(40, 50, alpha=0.15, color='orange', label='Caution Zone')

    ax2.set_xlabel('Time of Day (hours)', fontsize=11)
    ax2.set_ylabel('Battery Temperature (°C)', fontsize=11)
    ax2.set_title('Thermal Behavior: Larger Thermal Mass = Slower Heating', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 24)
    ax2.set_ylim(20, 50)

    # Add parameter table - move to bottom right
    table_text = (
        f"Parameter Adaptation:\n"
        f"{'Parameter':<15} {'Standard':<10} {'Foldable':<10}\n"
        f"{'-'*35}\n"
        f"{'Capacity (Ah)':<15} {params_standard['Q_nom']:<10.2f} {params_fold['Q_nom']:<10.2f}\n"
        f"{'Thermal Mass':<15} {params_standard['M_th']:<10} {params_fold['M_th']:<10}\n"
        f"{'Display (mW)':<15} {params_standard['P_display_max']:<10} {params_fold['P_display_max']:<10}\n"
        f"{'hA (W/K)':<15} {params_standard['hA']:<10.2f} {params_fold['hA']:<10.2f}"
    )
    ax2.text(0.02, 0.35, table_text, transform=ax2.transAxes,
             fontsize=8, family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))

    plt.suptitle('Model Generalization: Adaptation to High-Power Foldable Devices',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    """Generate all ECM visualizations."""
    print("="*70)
    print("GENERATING ECM PUBLICATION FIGURES")
    print("="*70)
    
    figures = [
        ('ecm_circuit_diagram', draw_ecm_circuit_diagram),
        ('ecm_rc_dynamics', draw_rc_dynamics),
        ('ecm_scenario_comparison', draw_model_comparison),
        ('ecm_thermal_feedback', draw_thermal_feedback_ecm),
        ('ecm_aging_effects', draw_aging_effects),
        ('ecm_daily_simulation', draw_realistic_day_simulation),
        ('thermal_runaway', draw_thermal_runaway_phase_plane),
        ('generalization_fold', draw_generalization_foldable),
    ]
    
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    for name, func in figures:
        print(f"\n[*] Generating {name}...", end=' ')
        try:
            fig = func()
            fig.savefig(os.path.join(output_dir, f'{name}.png'), dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "="*70)
    print("All ECM figures generated successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
