"""
=============================================================================
VISUALIZATION MODULE - O-Prize Style Figures
=============================================================================
Generates publication-quality figures matching the O-Prize winning paper style.

Figure 1: Two-Tank KiBaM Schematic (conceptual)
Figure 2: Thermal Feedback Loop Diagram
Figure 3: SOC vs Time Comparison (Linear vs. Our Model)
Figure 4: Temperature Effects on Capacity
Figure 5: Sensitivity Heat Map (Temperature × Usage Intensity)
Figure 6: Usage Scenario Comparison
Figure 7: Recovery Effect Demonstration
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import seaborn as sns  # Optional, not required

from battery_model import (
    BatteryParameters, KineticBatteryModel, ExtendedBatteryModel,
    USAGE_PROFILES, constant_current, periodic_usage, 
    realistic_day_profile, stochastic_usage, compute_time_to_empty
)

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette (professional, colorblind-friendly)
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'quaternary': '#C73E1D',
    'success': '#3A7D44',
    'neutral': '#4A4A4A',
}


def figure1_two_tank_schematic():
    """
    Figure 1: The Two-Tank KiBaM Schematic
    
    This is the conceptual diagram showing:
    - Available Charge Tank (q1) - connected to load
    - Bound Charge Tank (q2) - chemical reservoir
    - Flow between them (k parameter)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Tank 1: Available Charge (left)
    tank1 = FancyBboxPatch((1, 1), 2.5, 4, boxstyle="round,pad=0.05",
                           facecolor='#E3F2FD', edgecolor=COLORS['primary'], 
                           linewidth=2)
    ax.add_patch(tank1)
    
    # Water level in Tank 1 (70% full)
    water1 = Rectangle((1.1, 1.1), 2.3, 2.8, facecolor=COLORS['primary'], 
                        alpha=0.6)
    ax.add_patch(water1)
    
    # Tank 2: Bound Charge (right)
    tank2 = FancyBboxPatch((6, 1), 2.5, 4, boxstyle="round,pad=0.05",
                           facecolor='#FFF3E0', edgecolor=COLORS['tertiary'], 
                           linewidth=2)
    ax.add_patch(tank2)
    
    # Water level in Tank 2 (85% full)
    water2 = Rectangle((6.1, 1.1), 2.3, 3.4, facecolor=COLORS['tertiary'], 
                        alpha=0.6)
    ax.add_patch(water2)
    
    # Connecting pipe
    pipe = Rectangle((3.5, 2.2), 2.5, 0.6, facecolor='#BDBDBD', 
                      edgecolor='#616161', linewidth=1.5)
    ax.add_patch(pipe)
    
    # Flow arrow (right to left during rest)
    ax.annotate('', xy=(4.0, 2.5), xytext=(5.5, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], 
                               lw=2, mutation_scale=15))
    
    # Drain from Tank 1 (load)
    drain = Rectangle((2.0, 0.2), 0.5, 0.9, facecolor='#BDBDBD', 
                       edgecolor='#616161', linewidth=1.5)
    ax.add_patch(drain)
    ax.annotate('', xy=(2.25, -0.3), xytext=(2.25, 0.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['quaternary'], 
                               lw=2.5, mutation_scale=15))
    
    # Labels
    ax.text(2.25, 5.3, 'Available Charge\n$q_1$ (Direct Power)', 
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=COLORS['primary'])
    ax.text(7.25, 5.3, 'Bound Charge\n$q_2$ (Chemical Reserve)', 
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=COLORS['tertiary'])
    
    # Flow rate label
    ax.text(4.75, 3.2, 'Recovery Flow\n$k(q_2 - q_1\\frac{1-c}{c})$', 
            ha='center', va='bottom', fontsize=10, style='italic',
            color=COLORS['success'])
    
    # Load label
    ax.text(2.25, -0.7, 'Load Current $I(t)$', ha='center', fontsize=11,
            color=COLORS['quaternary'], fontweight='bold')
    
    # Title
    ax.text(5, 6.5, 'Kinetic Battery Model (KiBaM) - Two-Tank Analogy', 
            ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Equation box
    eq_text = (r'$\frac{dq_1}{dt} = -I(t) + k\left(q_2 - q_1\frac{1-c}{c}\right)$'
               '\n'
               r'$\frac{dq_2}{dt} = -k\left(q_2 - q_1\frac{1-c}{c}\right)$')
    ax.text(5, -1.5, eq_text, ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#9E9E9E'))
    
    plt.tight_layout()
    plt.savefig('figure1_kibam_schematic.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figure1_kibam_schematic.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: figure1_kibam_schematic.png/pdf")
    plt.close()


def figure2_thermal_feedback_loop():
    """
    Figure 2: Thermal Feedback Loop Diagram
    
    Shows the positive feedback cycle:
    High Current → Heat Generation → Temperature Rise → 
    Increased Resistance → Higher Power Loss → More Heat
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-2, 12)
    ax.set_ylim(-1, 9)
    ax.axis('off')
    
    # Define node positions (circular layout)
    nodes = {
        'current': (5, 8),
        'heat': (9, 6),
        'temp': (9, 2),
        'resistance': (5, 0),
        'efficiency': (1, 2),
        'drain': (1, 6),
    }
    
    labels = {
        'current': 'High Current\nDraw $I(t)$',
        'heat': 'Joule Heating\n$P = I^2 R$',
        'temp': 'Temperature\nRise $\\Delta T$',
        'resistance': 'Internal Resistance\n$R(T) \\uparrow$',
        'efficiency': 'Reduced\nEfficiency $\\eta$',
        'drain': 'Accelerated\nBattery Drain',
    }
    
    colors = {
        'current': '#E3F2FD',
        'heat': '#FFEBEE',
        'temp': '#FFF3E0',
        'resistance': '#F3E5F5',
        'efficiency': '#E8F5E9',
        'drain': '#FCE4EC',
    }
    
    edge_colors = {
        'current': COLORS['primary'],
        'heat': COLORS['quaternary'],
        'temp': COLORS['tertiary'],
        'resistance': COLORS['secondary'],
        'efficiency': COLORS['success'],
        'drain': COLORS['quaternary'],
    }
    
    # Draw nodes
    for name, (x, y) in nodes.items():
        box = FancyBboxPatch((x-1.3, y-0.7), 2.6, 1.4, 
                             boxstyle="round,pad=0.1",
                             facecolor=colors[name], 
                             edgecolor=edge_colors[name],
                             linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, labels[name], ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Draw arrows (the feedback loop)
    arrow_pairs = [
        ('current', 'heat'),
        ('heat', 'temp'),
        ('temp', 'resistance'),
        ('resistance', 'efficiency'),
        ('efficiency', 'drain'),
        ('drain', 'current'),
    ]
    
    for start, end in arrow_pairs:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        
        # Calculate arrow direction
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        # Offset to start/end at box edges
        offset = 1.5
        x1_adj = x1 + offset * dx / length
        y1_adj = y1 + offset * dy / length
        x2_adj = x2 - offset * dx / length
        y2_adj = y2 - offset * dy / length
        
        ax.annotate('', xy=(x2_adj, y2_adj), xytext=(x1_adj, y1_adj),
                    arrowprops=dict(arrowstyle='->', color='#424242',
                                   lw=2, mutation_scale=20,
                                   connectionstyle='arc3,rad=0.1'))
    
    # Central feedback indicator
    ax.text(5, 4, '⟳ POSITIVE\nFEEDBACK', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['quaternary'],
            bbox=dict(boxstyle='round', facecolor='#FFCDD2', 
                     edgecolor=COLORS['quaternary'], linewidth=2))
    
    # Title
    ax.text(5, 9.5, 'Thermal Feedback Loop in Battery Discharge',
            ha='center', fontsize=14, fontweight='bold')
    
    # Caption
    caption = ('This positive feedback loop explains "thermal runaway":\n'
               'Heavy use → heat → higher resistance → more heat → faster drain')
    ax.text(5, -1.5, caption, ha='center', fontsize=10, style='italic',
            color='#616161')
    
    plt.tight_layout()
    plt.savefig('figure2_thermal_feedback.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('figure2_thermal_feedback.pdf', bbox_inches='tight',
                facecolor='white')
    print("Saved: figure2_thermal_feedback.png/pdf")
    plt.close()


def figure3_model_comparison():
    """
    Figure 3: Comparison of Linear Model vs. Our Physicochemical Model
    
    Shows how our model captures:
    1. Non-linear discharge (especially at low SOC)
    2. Temperature effects
    3. Recovery during rest periods
    """
    params = BatteryParameters()
    extended_model = ExtendedBatteryModel(params)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== Panel A: Constant load comparison =====
    ax1 = axes[0]
    
    # Our model
    result = extended_model.simulate(
        constant_current(500),
        t_span=(0, 12*3600),
        T_ambient=298.15
    )
    
    # Linear model (simple coulomb counting)
    t_linear = np.linspace(0, 12, 200)
    Q_nom = params.Q_nom
    I = 500  # mA
    SOC_linear = np.maximum(0, 100 * (1 - I * t_linear / Q_nom))
    
    ax1.plot(result['t_hours'], result['SOC_percent'], 
             color=COLORS['primary'], linewidth=2.5, label='Our Model (KiBaM + Thermal)')
    ax1.plot(t_linear, SOC_linear, '--', color=COLORS['tertiary'], 
             linewidth=2.5, label='Linear Model (Coulomb Counting)')
    
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('State of Charge (%)')
    ax1.set_title('(a) Constant 500mA Load at 25°C')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=20, color='red', linestyle=':', alpha=0.5, label='Low Battery Warning')
    
    # ===== Panel B: Variable load with recovery =====
    ax2 = axes[1]
    
    # Our model with periodic usage
    result_periodic = extended_model.simulate(
        periodic_usage(800, 30, period=3600, duty_cycle=0.25),
        t_span=(0, 14*3600),
        T_ambient=298.15
    )
    
    # Equivalent linear model
    I_avg = 0.25 * 800 + 0.75 * 30  # Average current
    t_linear = np.linspace(0, 14, 200)
    SOC_linear_avg = np.maximum(0, 100 * (1 - I_avg * t_linear / Q_nom))
    
    ax2.plot(result_periodic['t_hours'], result_periodic['SOC_percent'],
             color=COLORS['primary'], linewidth=2.5, label='Our Model (with Recovery)')
    ax2.plot(t_linear, SOC_linear_avg, '--', color=COLORS['tertiary'],
             linewidth=2.5, label='Linear Model (Average Current)')
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('State of Charge (%)')
    ax2.set_title('(b) Periodic Usage (800mA active / 30mA idle, 25% duty)')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison: Traditional Linear Model vs. Our Physicochemical Model',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('figure3_model_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('figure3_model_comparison.pdf', bbox_inches='tight',
                facecolor='white')
    print("Saved: figure3_model_comparison.png/pdf")
    plt.close()


def figure4_temperature_effects():
    """
    Figure 4: Temperature Effects on Battery Performance
    
    Shows how:
    1. Cold reduces effective capacity (Arrhenius)
    2. Heat increases self-discharge but allows faster recovery
    """
    params = BatteryParameters()
    extended_model = ExtendedBatteryModel(params)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== Panel A: SOC curves at different temperatures =====
    ax1 = axes[0]
    
    temperatures = [-10, 0, 15, 25, 35, 45]  # Celsius
    colors_temp = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(temperatures)))
    
    for T_c, color in zip(temperatures, colors_temp):
        T_k = T_c + 273.15
        result = extended_model.simulate(
            constant_current(500),
            t_span=(0, 12*3600),
            T_ambient=T_k
        )
        ax1.plot(result['t_hours'], result['SOC_percent'],
                color=color, linewidth=2, label=f'{T_c}°C')
    
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('State of Charge (%)')
    ax1.set_title('(a) Battery Discharge at Various Temperatures')
    ax1.legend(title='Ambient Temp', loc='upper right', ncol=2)
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    
    # ===== Panel B: Effective capacity vs temperature =====
    ax2 = axes[1]
    
    T_range = np.linspace(-20, 50, 100)
    capacities = []
    
    for T_c in T_range:
        T_k = T_c + 273.15
        cap = extended_model.effective_capacity(T_k)
        capacities.append(cap)
    
    capacities = np.array(capacities)
    relative_cap = 100 * capacities / params.Q_nom
    
    ax2.fill_between(T_range, relative_cap, alpha=0.3, color=COLORS['primary'])
    ax2.plot(T_range, relative_cap, color=COLORS['primary'], linewidth=2.5)
    
    # Mark key temperatures
    key_temps = [-10, 0, 25, 40]
    for T_c in key_temps:
        T_k = T_c + 273.15
        cap = 100 * extended_model.effective_capacity(T_k) / params.Q_nom
        ax2.plot(T_c, cap, 'o', color=COLORS['quaternary'], markersize=10)
        ax2.annotate(f'{cap:.0f}%', xy=(T_c, cap), xytext=(T_c+3, cap+5),
                    fontsize=10, fontweight='bold')
    
    ax2.axvline(x=25, color='green', linestyle='--', alpha=0.5, 
                label='Reference (25°C)')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Effective Capacity (% of nominal)')
    ax2.set_title('(b) Temperature-Dependent Effective Capacity (Arrhenius)')
    ax2.legend()
    ax2.set_xlim(-20, 50)
    ax2.set_ylim(50, 110)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Temperature Effects on Li-ion Battery Performance',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('figure4_temperature_effects.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('figure4_temperature_effects.pdf', bbox_inches='tight',
                facecolor='white')
    print("Saved: figure4_temperature_effects.png/pdf")
    plt.close()


def figure5_sensitivity_heatmap():
    """
    Figure 5: Sensitivity Analysis Heat Map
    
    2D map showing Time-to-Empty as function of:
    X-axis: Average Current Draw (Usage Intensity)
    Y-axis: Ambient Temperature
    
    This is the "O-Prize signature" visualization.
    """
    params = BatteryParameters()
    extended_model = ExtendedBatteryModel(params)
    
    # Define grid
    currents = np.linspace(100, 1500, 25)  # mA
    temperatures = np.linspace(-15, 45, 20)  # Celsius
    
    # Compute time-to-empty for each combination
    tte_grid = np.zeros((len(temperatures), len(currents)))
    
    print("Computing sensitivity heat map (this may take a minute)...")
    for i, T_c in enumerate(temperatures):
        for j, I in enumerate(currents):
            T_k = T_c + 273.15
            result = extended_model.simulate(
                constant_current(I),
                t_span=(0, 48*3600),
                T_ambient=T_k,
                dt_eval=300  # 5-minute steps for speed
            )
            tte = result['time_to_empty']
            tte_grid[i, j] = tte if tte else 48.0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Heat map
    im = ax.imshow(tte_grid, aspect='auto', origin='lower',
                   extent=[currents.min(), currents.max(), 
                          temperatures.min(), temperatures.max()],
                   cmap='RdYlGn', vmin=0, vmax=24)
    
    # Contour lines
    CS = ax.contour(currents, temperatures, tte_grid, 
                    levels=[2, 4, 6, 8, 10, 12, 16, 20],
                    colors='black', linewidths=1, alpha=0.7)
    ax.clabel(CS, inline=True, fontsize=9, fmt='%d hrs')
    
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Time to Empty (hours)', fontsize=12)
    
    # Labels and title
    ax.set_xlabel('Average Current Draw (mA)', fontsize=12)
    ax.set_ylabel('Ambient Temperature (°C)', fontsize=12)
    ax.set_title('Sensitivity Analysis: Battery Life under Varying Conditions',
                 fontsize=14, fontweight='bold')
    
    # Mark typical scenarios
    scenarios = {
        'Idle': (50, 25),
        'Light Use': (200, 25),
        'Gaming': (1200, 35),
        'Cold Weather': (400, -10),
        'Navigation': (800, 30),
    }
    
    for name, (I, T) in scenarios.items():
        ax.plot(I, T, 'ko', markersize=8)
        ax.annotate(name, xy=(I, T), xytext=(I+50, T+3),
                   fontsize=9, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    
    plt.tight_layout()
    plt.savefig('figure5_sensitivity_heatmap.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('figure5_sensitivity_heatmap.pdf', bbox_inches='tight',
                facecolor='white')
    print("Saved: figure5_sensitivity_heatmap.png/pdf")
    plt.close()


def figure6_usage_scenarios():
    """
    Figure 6: Comparison of Different Usage Scenarios
    
    Simulates a "Gamer", "Casual User", "Business User", etc.
    """
    params = BatteryParameters()
    extended_model = ExtendedBatteryModel(params)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scenarios = [
        ("Gaming Session", constant_current(1400), 298.15, axes[0, 0]),
        ("Casual Use", periodic_usage(400, 30, period=1800, duty_cycle=0.2), 298.15, axes[0, 1]),
        ("Navigation + Hot Car", constant_current(900), 313.15, axes[1, 0]),  # 40°C
        ("Winter Commute", constant_current(600), 263.15, axes[1, 1]),  # -10°C
    ]
    
    for name, I_func, T_amb, ax in scenarios:
        result = extended_model.simulate(
            I_func,
            t_span=(0, 12*3600),
            T_ambient=T_amb
        )
        
        # Plot SOC
        color = COLORS['primary']
        ax.plot(result['t_hours'], result['SOC_percent'], 
                color=color, linewidth=2.5)
        ax.fill_between(result['t_hours'], 0, result['SOC_percent'],
                       alpha=0.2, color=color)
        
        # Mark time-to-empty
        tte = result['time_to_empty']
        if tte:
            ax.axvline(x=tte, color=COLORS['quaternary'], linestyle='--', 
                      linewidth=2, label=f'Empty: {tte:.1f}h')
        
        T_c = T_amb - 273.15
        ax.set_title(f'{name} (Ambient: {T_c:.0f}°C)', fontweight='bold')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('SOC (%)')
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Add battery indicator icons at certain thresholds
        ax.axhline(y=20, color='orange', linestyle=':', alpha=0.5)
        ax.axhline(y=5, color='red', linestyle=':', alpha=0.5)
    
    plt.suptitle('Battery Drain Under Different Real-World Scenarios',
                 fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    plt.savefig('figure6_usage_scenarios.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('figure6_usage_scenarios.pdf', bbox_inches='tight',
                facecolor='white')
    print("Saved: figure6_usage_scenarios.png/pdf")
    plt.close()


def figure7_recovery_effect():
    """
    Figure 7: Demonstrating the Recovery Effect
    
    The key differentiator of KiBaM - shows the "bounce back" during rest.
    """
    params = BatteryParameters()
    
    # Use basic KiBaM to clearly show recovery
    basic_model = KineticBatteryModel(params)
    
    # Create a "pulse" usage pattern
    def pulse_usage(t):
        """30 min heavy use, 30 min rest, repeat"""
        cycle_pos = t % 3600  # 1-hour cycles
        if cycle_pos < 1800:  # First 30 min: heavy
            return 1000  # mA
        else:  # Second 30 min: rest
            return 20  # mA
    
    result_pulse = basic_model.simulate(
        pulse_usage,
        t_span=(0, 8*3600),
        SOC_0=1.0,
        dt_eval=30
    )
    
    # Compare with continuous equivalent
    result_continuous = basic_model.simulate(
        constant_current(510),  # Approximately same average
        t_span=(0, 8*3600),
        SOC_0=1.0,
        dt_eval=30
    )
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # ===== Panel A: SOC comparison =====
    ax1 = axes[0]
    
    ax1.plot(result_pulse['t_hours'], result_pulse['SOC_percent'],
             color=COLORS['primary'], linewidth=2.5, label='Pulsed Usage (1000mA/20mA)')
    ax1.plot(result_continuous['t_hours'], result_continuous['SOC_percent'],
             '--', color=COLORS['tertiary'], linewidth=2.5, label='Continuous ~510mA')
    
    ax1.set_ylabel('State of Charge (%)', fontsize=12)
    ax1.set_title('(a) SOC: Recovery Effect Extends Battery Life', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    
    # ===== Panel B: Two-tank dynamics =====
    ax2 = axes[1]
    
    ax2.plot(result_pulse['t_hours'], result_pulse['q1'], 
             color=COLORS['primary'], linewidth=2, label='Available Charge $q_1$')
    ax2.plot(result_pulse['t_hours'], result_pulse['q2'],
             color=COLORS['tertiary'], linewidth=2, label='Bound Charge $q_2$')
    
    # Shade rest periods
    for i in range(8):
        ax2.axvspan(i + 0.5, i + 1.0, alpha=0.1, color='green', 
                   label='Rest Period' if i == 0 else None)
    
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Charge (mAh)', fontsize=12)
    ax2.set_title('(b) KiBaM Two-Tank Dynamics: Recovery Flow During Rest', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 8)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('The Recovery Effect: Why Intermittent Use Beats Continuous Use',
                 fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    plt.savefig('figure7_recovery_effect.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('figure7_recovery_effect.pdf', bbox_inches='tight',
                facecolor='white')
    print("Saved: figure7_recovery_effect.png/pdf")
    plt.close()


def figure8_aging_effect():
    """
    Figure 8: Effect of Battery Aging on Performance
    
    Shows capacity fade over charge cycles.
    """
    params_new = BatteryParameters(n_cycles=0)
    params_1yr = BatteryParameters(n_cycles=365)
    params_2yr = BatteryParameters(n_cycles=730)
    params_3yr = BatteryParameters(n_cycles=1095)
    
    models = [
        ("New Battery", ExtendedBatteryModel(params_new)),
        ("1 Year (~365 cycles)", ExtendedBatteryModel(params_1yr)),
        ("2 Years (~730 cycles)", ExtendedBatteryModel(params_2yr)),
        ("3 Years (~1095 cycles)", ExtendedBatteryModel(params_3yr)),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(models)))
    
    for (name, model), color in zip(models, colors):
        result = model.simulate(
            constant_current(500),
            t_span=(0, 12*3600),
            T_ambient=298.15
        )
        
        tte = result['time_to_empty']
        tte_str = f" ({tte:.1f}h)" if tte else ""
        ax.plot(result['t_hours'], result['SOC_percent'],
                color=color, linewidth=2.5, label=f'{name}{tte_str}')
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('State of Charge (%)', fontsize=12)
    ax.set_title('Effect of Battery Aging on Discharge Performance (500mA load)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Battery Age', loc='upper right')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure8_aging_effect.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('figure8_aging_effect.pdf', bbox_inches='tight',
                facecolor='white')
    print("Saved: figure8_aging_effect.png/pdf")
    plt.close()


def generate_all_figures():
    """Generate all publication figures."""
    print("="*60)
    print("GENERATING O-PRIZE STYLE FIGURES")
    print("="*60)
    
    figure1_two_tank_schematic()
    figure2_thermal_feedback_loop()
    figure3_model_comparison()
    figure4_temperature_effects()
    figure5_sensitivity_heatmap()
    figure6_usage_scenarios()
    figure7_recovery_effect()
    figure8_aging_effect()
    
    print("\n" + "="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    generate_all_figures()
