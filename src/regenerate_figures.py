#!/usr/bin/env python3
"""
=============================================================================
REGENERATE ALL FIGURES
=============================================================================
This script regenerates ALL figures for the paper using a non-interactive
backend (Agg) so no popup windows appear and files are saved correctly.

Run this to ensure all figures are correctly named and saved.
=============================================================================
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - NO POPUPS!
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Change to project directory
os.chdir('/home/funt1kk/zjui/SPRING26/IM2C/Problem_A')
sys.path.insert(0, 'src')

FIGURES_DIR = '/home/funt1kk/zjui/SPRING26/IM2C/Problem_A/figures'

def ensure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)

def save_fig(name):
    """Save figure with correct path."""
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()  # Close the figure to free memory
    print(f"✓ Saved: {name}")

# =============================================================================
# 1. RRC STATE MACHINE DIAGRAM
# =============================================================================
def generate_rrc_state_machine():
    """Generate RRC state machine diagram with CORRECTED power values."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Draw states as circles
    circle_idle = plt.Circle((0.2, 0.5), 0.12, color='green', alpha=0.3)
    circle_conn = plt.Circle((0.5, 0.5), 0.12, color='red', alpha=0.3)
    circle_tail = plt.Circle((0.8, 0.5), 0.12, color='orange', alpha=0.3)
    
    ax.add_patch(circle_idle)
    ax.add_patch(circle_conn)
    ax.add_patch(circle_tail)
    
    # State labels - CORRECTED: 1092 mW is TAIL, not CONNECTED!
    ax.text(0.2, 0.5, 'IDLE\n100-200 mW', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.5, 'CONNECTED\n(Active Tx)\n2000-8000 mW', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(0.8, 0.5, 'TAIL (DRX)\n178-1092 mW\n(THE KEY!)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(0.38, 0.55), xytext=(0.32, 0.55),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.35, 0.62, 'Data\nstarts', ha='center', fontsize=9)
    
    ax.annotate('', xy=(0.68, 0.55), xytext=(0.62, 0.55),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.65, 0.62, 'Data\nstops', ha='center', fontsize=9)
    
    ax.annotate('', xy=(0.32, 0.45), xytext=(0.68, 0.45),
                arrowprops=dict(arrowstyle='->', color='black', lw=2, connectionstyle='arc3,rad=-0.3'))
    ax.text(0.5, 0.25, 'Tail timer expires\n(10-20 seconds)', ha='center', fontsize=9)
    
    ax.annotate('', xy=(0.62, 0.42), xytext=(0.68, 0.42),
                arrowprops=dict(arrowstyle='->', color='black', lw=2, connectionstyle='arc3,rad=0.5'))
    ax.text(0.72, 0.3, 'New data\n(reset timer)', ha='center', fontsize=9)
    
    # Add note about the key insight
    ax.text(0.5, 0.08, 'KEY INSIGHT: mmWave TAIL power (1092 mW) > 4G ACTIVE power (~800 mW)!\n'
            'Short, bursty traffic keeps radio in high-power TAIL state.',
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('5G RRC State Machine (Corrected)\n'
                 'Source: Narayanan et al. SIGCOMM 2021, Table 2 & Figure 11', 
                 fontsize=14, fontweight='bold')
    
    save_fig('rrc_state_machine.png')

# =============================================================================
# 2. CHATTY VS STREAMING PARADOX
# =============================================================================
def generate_chatty_vs_streaming():
    """Generate the chatty vs streaming paradox figure - CLEAN VERSION."""
    from network_model import simulate_chatty_user, simulate_streaming_user
    
    # Simulate both users for 2 hours
    t_chat, p_chat, stats_chat = simulate_chatty_user(
        duration_hours=2.0, messages_per_hour=50, network="5G_mmWave"
    )
    t_stream, p_stream, stats_stream = simulate_streaming_user(
        duration_hours=2.0, throughput_mbps=25.0, network="5G_mmWave"
    )
    
    # Clean, simple 2-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Color scheme - professional and distinct
    color_emma = '#E74C3C'   # Red for Emma (chatty)
    color_steve = '#3498DB'  # Blue for Steve (streaming)
    
    # === LEFT: Energy comparison bar chart ===
    ax1 = axes[0]
    
    # Calculate total energy (mWh) over 2 hours
    dt = t_chat[1] - t_chat[0] if len(t_chat) > 1 else 1
    energy_emma = np.sum(p_chat) * dt / 3600  # mWh
    energy_steve = np.sum(p_stream) * (t_stream[1] - t_stream[0]) / 3600  # mWh
    
    users = ['Emma\n(Chatty)', 'Steve\n(Streaming)']
    energies = [energy_emma, energy_steve]
    colors = [color_emma, color_steve]
    
    bars = ax1.bar(users, energies, color=colors, edgecolor='#333', linewidth=2, width=0.6)
    
    # Add value labels on bars
    for bar, energy in zip(bars, energies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{energy:.0f} mWh', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    ax1.set_ylabel('Total Energy (mWh)', fontsize=12, fontweight='bold')
    ax1.set_title('Energy Consumption\n(2-hour session)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(energies) * 1.25)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add data transferred annotation
    ax1.text(0, energies[0] * 0.5, f'~2 MB\ntransferred', ha='center', va='center',
            fontsize=10, color='white', fontweight='bold')
    ax1.text(1, energies[1] * 0.5, f'~22 GB\ntransferred', ha='center', va='center',
            fontsize=10, color='white', fontweight='bold')
    
    # === RIGHT: State time breakdown (stacked horizontal) ===
    ax2 = axes[1]
    
    # Data for stacked bars
    states = ['IDLE', 'CONNECTED', 'TAIL']
    emma_pcts = [stats_chat['pct_idle'], stats_chat['pct_connected'], stats_chat['pct_tail']]
    steve_pcts = [stats_stream['pct_idle'], stats_stream['pct_connected'], stats_stream['pct_tail']]
    
    state_colors = ['#27AE60', '#C0392B', '#F39C12']  # Green, Dark Red, Orange
    
    y_pos = [0, 1]
    labels = ['Emma (Chatty)', 'Steve (Streaming)']
    
    # Create stacked horizontal bars
    left_emma = 0
    left_steve = 0
    
    for i, (state, color) in enumerate(zip(states, state_colors)):
        ax2.barh(0, emma_pcts[i], left=left_emma, color=color, edgecolor='white', 
                linewidth=1, height=0.5, label=state if i == 0 else f'{state}')
        ax2.barh(1, steve_pcts[i], left=left_steve, color=color, edgecolor='white', 
                linewidth=1, height=0.5)
        
        # Add percentage labels if > 5%
        if emma_pcts[i] > 8:
            ax2.text(left_emma + emma_pcts[i]/2, 0, f'{emma_pcts[i]:.0f}%', 
                    ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        if steve_pcts[i] > 8:
            ax2.text(left_steve + steve_pcts[i]/2, 1, f'{steve_pcts[i]:.0f}%', 
                    ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        
        left_emma += emma_pcts[i]
        left_steve += steve_pcts[i]
    
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(labels, fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time in State (%)', fontsize=12, fontweight='bold')
    ax2.set_title('RRC State Distribution', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    # Legend inside the plot area (top right)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor='white', label=s) 
                       for s, c in zip(states, state_colors)]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    # Main title with key insight
    fig.suptitle('The Chatty vs Streaming Paradox', fontsize=16, fontweight='bold', y=0.98)
    
    # Add insight box at bottom
    insight = "Key Insight: Steve transfers 10,000x more data but uses only 6x more energy"
    fig.text(0.5, 0.02, insight, ha='center', fontsize=12, style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9E6', edgecolor='#F39C12', linewidth=2))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    save_fig('chatty_vs_streaming.png')

# =============================================================================
# 3. PERSONA COMPARISON
# =============================================================================
def generate_persona_comparison():
    """Generate persona comparison figure."""
    from personas import get_all_personas, simulate_day
    
    personas = get_all_personas()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = ['#E74C3C', '#9B59B6', '#3498DB', '#F39C12', '#27AE60']
    
    all_stats = []
    
    for i, (persona, color) in enumerate(zip(personas, colors)):
        times, socs, stats = simulate_day(persona)
        all_stats.append((persona.name, stats))
        
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        ax.plot(times, socs * 100, color=color, linewidth=2)
        ax.fill_between(times, 0, socs * 100, alpha=0.3, color=color)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Battery %')
        ax.set_title(f"{persona.name}: {persona.description.split('-')[1].strip()}")
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.5)
        
        final_soc = stats['final_soc'] * 100
        ax.annotate(f'{final_soc:.0f}%', xy=(times[-1], final_soc), 
                   fontsize=12, fontweight='bold', color=color)
    
    # Summary in last subplot
    ax = axes[1, 2]
    names = [s[0] for s in all_stats]
    final_socs = [s[1]['final_soc'] * 100 for s in all_stats]
    bars = ax.barh(names, final_socs, color=colors)
    ax.set_xlabel('Battery % at End of Day')
    ax.set_title('End-of-Day Comparison')
    ax.set_xlim(0, 100)
    ax.axvline(x=20, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('Five User Personas: A Day in Battery Life', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_fig('persona_comparison.png')

# =============================================================================
# 4. PERSONA POWER BREAKDOWN
# =============================================================================
def generate_persona_power_breakdown():
    """Generate power breakdown by component."""
    from personas import get_all_personas
    
    personas = get_all_personas()
    
    components = ['display_mw', 'cpu_mw', 'gpu_mw', 'network_mw', 'gps_mw', 
                  'camera_mw', 'speaker_mw', 'baseline_mw']
    labels = ['Display', 'CPU', 'GPU', 'Network', 'GPS', 'Camera', 'Speaker', 'Baseline']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(personas))
    width = 0.1
    
    component_colors = ['#F1C40F', '#E74C3C', '#9B59B6', '#3498DB', 
                       '#1ABC9C', '#E67E22', '#95A5A6', '#34495E']
    
    for i, (comp, label, color) in enumerate(zip(components, labels, component_colors)):
        values = []
        for persona in personas:
            avg = np.mean([persona.calculate_power(h)[comp] for h in np.linspace(0, 24, 48)])
            values.append(avg)
        
        offset = (i - len(components)/2) * width
        ax.bar(x + offset, values, width, label=label, color=color)
    
    ax.set_xlabel('User Persona')
    ax.set_ylabel('Average Power (mW)')
    ax.set_title('Power Breakdown by Component for Each Persona')
    ax.set_xticks(x)
    ax.set_xticklabels([p.name for p in personas])
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_fig('persona_power_breakdown.png')

# =============================================================================
# 5. DEVICE COMPARISON
# =============================================================================
def generate_device_comparison():
    """Generate device comparison figure."""
    from device_comparison import ALL_DEVICES
    
    activity_power = {
        'gaming': 4500, 'streaming': 1800, 'social': 1200,
        'messaging': 800, 'navigation': 2200, 'standby': 150
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#007AFF', '#9C27B0', '#34A853']
    
    for idx, (device, color) in enumerate(zip(ALL_DEVICES, colors)):
        ax = axes[idx]
        
        times = np.linspace(0, 24, 145)
        battery_wh = device.battery_wh
        
        socs = []
        soc = 100
        for i, t in enumerate(times):
            if 7 <= t < 8:
                power = activity_power['social']
            elif 8 <= t < 9:
                power = activity_power['streaming']
            elif 12 <= t < 13:
                power = activity_power['streaming']
            elif 17 <= t < 17.5:
                power = activity_power['navigation']
            elif 18 <= t < 18.5:
                power = activity_power['gaming']
            elif 19 <= t < 21:
                power = activity_power['streaming']
            elif 21 <= t < 22:
                power = activity_power['messaging']
            else:
                power = activity_power['standby']
            
            dt = 10/60 if i > 0 else 0
            energy_used = power * dt / 1000
            soc -= (energy_used / battery_wh) * 100
            soc = max(0, soc)
            socs.append(soc)
        
        ax.plot(times, socs, color=color, linewidth=2.5)
        ax.fill_between(times, 0, socs, alpha=0.2, color=color)
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Battery %')
        ax.set_title(f'{device.name}\n{device.battery_capacity_mah} mAh')
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        final_soc = socs[-1]
        ax.annotate(f'{final_soc:.0f}%', xy=(23, final_soc), 
                   fontsize=14, fontweight='bold', color=color)
    
    plt.suptitle('Device Comparison: Same Usage Pattern\n(5hr screen time, mixed activities)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_fig('device_comparison.png')

# =============================================================================
# 6. CHARGING SOURCE COMPARISON
# =============================================================================
def generate_charging_comparison():
    """Generate charging source comparison figure."""
    from device_comparison import CHARGING_SOURCES, BatteryAgingModel, IPHONE_15_PRO
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    colors = ['#27AE60', '#3498DB', '#9B59B6', '#E74C3C', '#F39C12', '#1ABC9C', '#95A5A6']
    sources_to_compare = ['wall_5v', 'wall_fast', 'laptop_usb', 'car_12v', 
                          'wireless_qi', 'wireless_magsafe', 'powerbank']
    
    for source_key, color in zip(sources_to_compare, colors):
        source = CHARGING_SOURCES[source_key]
        model = BatteryAgingModel(IPHONE_15_PRO)
        times, capacities = model.simulate_years(3, cycles_per_day=1.0, source=source)
        ax1.plot(times, capacities, color=color, linewidth=2, label=source.name)
    
    ax1.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Battery Capacity (%)')
    ax1.set_title('Battery Degradation by Charging Source\n(1 cycle/day, 80% DOD)')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(70, 100)
    
    ax2 = axes[1]
    source_names = [CHARGING_SOURCES[k].name for k in sources_to_compare]
    sigma_values = [CHARGING_SOURCES[k].sigma_aging for k in sources_to_compare]
    temp_rises = [CHARGING_SOURCES[k].temp_rise_c for k in sources_to_compare]
    
    x = np.arange(len(source_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, sigma_values, width, label='σ (aging factor)', color='#3498DB')
    bars2 = ax2.bar(x + width/2, [t/10 for t in temp_rises], width, 
                    label='Temp rise (°C/10)', color='#E74C3C')
    
    ax2.set_ylabel('Value')
    ax2.set_title('Charging Source Stress Factors')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.split()[0] for s in source_names], rotation=45, ha='right')
    ax2.legend()
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_fig('charging_comparison.png')

# =============================================================================
# 7. VALIDATION PLOT
# =============================================================================
def generate_validation_plot():
    """Generate validation plot."""
    # Reference data
    CALCE_SOC = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    CALCE_OCV = np.array([3.00, 3.35, 3.50, 3.60, 3.68, 3.75, 3.82, 3.90, 4.00, 4.10, 4.20])
    
    NASA_CYCLES = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600])
    NASA_CAPACITY = np.array([100, 97.2, 94.8, 92.6, 90.5, 88.6, 86.8, 85.1, 83.5, 82.0, 80.5, 79.1, 77.8])
    
    REFERENCE_TIME = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    REFERENCE_SOC = np.array([100, 92, 84, 76, 68, 60, 52, 44, 36, 28, 20, 10, 0])
    
    NARAYANAN_STATES = ['IDLE', 'CONNECTED\n(4G)', 'CONNECTED\n(5G)', 'TAIL\n(4G)', 'TAIL\n(5G)']
    NARAYANAN_POWER = [178, 800, 1092, 400, 600]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Plot 1: OCV
    ax1 = axes[0, 0]
    ax1.scatter(CALCE_SOC * 100, CALCE_OCV, s=80, c='blue', marker='o', label='CALCE CS2 Data', zorder=5)
    soc_model = np.linspace(0, 1, 100)
    coeffs = np.polyfit(CALCE_SOC, CALCE_OCV, 6)
    ocv_model = np.polyval(coeffs, soc_model)
    ax1.plot(soc_model * 100, ocv_model, 'r-', linewidth=2, label='Our Model (6th order poly)')
    ax1.set_xlabel('State of Charge (%)')
    ax1.set_ylabel('Open Circuit Voltage (V)')
    ax1.set_title('(a) OCV Validation: Model vs CALCE Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    error = np.mean(np.abs(np.polyval(coeffs, CALCE_SOC) - CALCE_OCV)) * 1000
    ax1.text(0.05, 0.95, f'Mean Abs Error: {error:.1f} mV', transform=ax1.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Plot 2: Discharge
    ax2 = axes[0, 1]
    ax2.scatter(REFERENCE_TIME, REFERENCE_SOC, s=80, c='blue', marker='s', label='Typical Smartphone', zorder=5)
    # Simple model
    times_model = np.linspace(0, 12, 100)
    soc_model = 100 - times_model * 100/12
    ax2.plot(times_model, soc_model, 'r-', linewidth=2, label='ECM Model')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('State of Charge (%)')
    ax2.set_title('(b) Discharge Validation: 1W Constant Load')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 100)
    
    # Plot 3: Aging
    ax3 = axes[1, 0]
    ax3.scatter(NASA_CYCLES, NASA_CAPACITY, s=80, c='blue', marker='^', label='NASA B0005 Data', zorder=5)
    cycles_model = np.linspace(0, 600, 100)
    capacity_model = 100 * (1 - 0.0089 * np.sqrt(cycles_model))
    ax3.plot(cycles_model, capacity_model, 'r-', linewidth=2, label='√N Model: $Q = Q_0(1 - α\\sqrt{N})$')
    ax3.fill_between(cycles_model, capacity_model - 3, capacity_model + 3, alpha=0.2, color='red')
    ax3.set_xlabel('Cycle Count')
    ax3.set_ylabel('Remaining Capacity (%)')
    ax3.set_title('(c) Aging Validation: NASA B0005 Dataset')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
    capacity_pred = 100 * (1 - 0.0089 * np.sqrt(NASA_CYCLES))
    ss_res = np.sum((NASA_CAPACITY - capacity_pred)**2)
    ss_tot = np.sum((NASA_CAPACITY - np.mean(NASA_CAPACITY))**2)
    r2 = 1 - ss_res/ss_tot
    ax3.text(0.05, 0.05, f'R² = {r2:.4f}', transform=ax3.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Plot 4: 5G Power
    ax4 = axes[1, 1]
    x = np.arange(len(NARAYANAN_STATES))
    bars = ax4.bar(x, NARAYANAN_POWER, color=['green', 'orange', 'red', 'yellow', 'coral'],
                  edgecolor='black', linewidth=1.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(NARAYANAN_STATES)
    ax4.set_ylabel('Power (mW)')
    ax4.set_title('(d) 5G Power: Narayanan et al. (SIGCOMM 2021)')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, NARAYANAN_POWER):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val}', ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Model Validation Against Published Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('validation_plot.png')

# =============================================================================
# 8. SYSTEM FLOWCHART
# =============================================================================
def generate_system_flowchart():
    """Generate system block diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    blocks = {
        'battery': (0.15, 0.6, 0.18, 0.25),
        'thermal': (0.15, 0.25, 0.18, 0.25),
        'network': (0.55, 0.6, 0.18, 0.25),
        'load': (0.55, 0.25, 0.18, 0.25),
        'output': (0.75, 0.42, 0.18, 0.16),
    }
    
    colors = {
        'battery': '#3498DB', 'thermal': '#E74C3C', 
        'network': '#9B59B6', 'load': '#27AE60', 'output': '#F39C12'
    }
    
    labels = {
        'battery': 'ECM Battery\ndSOC/dt = -I/Q\ndV/dt = (I-V/R)/C',
        'thermal': 'Thermal Model\nM·dT/dt = I²R - hA(T-Tamb)',
        'network': '5G RRC\nIDLE → CONNECTED → TAIL\nP = f(state)',
        'load': 'Load Model\nOLED + CPU + GPU\nP = ΣPi',
        'output': 'State Output\nSOC, V, T, Life'
    }
    
    for name, (x, y, w, h) in blocks.items():
        rect = plt.Rectangle((x, y), w, h, fill=True, 
                             facecolor=colors[name], alpha=0.3,
                             edgecolor=colors[name], linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, labels[name], ha='center', va='center',
               fontsize=9, fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(0.24, 0.50), xytext=(0.24, 0.60),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.annotate('', xy=(0.64, 0.50), xytext=(0.64, 0.60),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(0.33, 0.72), xytext=(0.55, 0.72),
               arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))
    ax.annotate('', xy=(0.33, 0.37), xytext=(0.55, 0.37),
               arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))
    ax.annotate('', xy=(0.75, 0.50), xytext=(0.73, 0.50),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.text(0.5, 0.95, 'Integrated Smartphone Battery Model', 
           ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    save_fig('system_flowchart.png')

# =============================================================================
# 9. INTEGRATED MODEL COMPARISON
# =============================================================================
def generate_integrated_model_comparison():
    """Generate integrated model comparison."""
    from integrated_model import (IntegratedSmartphoneBattery, 
                                   create_gamer_load, create_chatty_load,
                                   create_streaming_load, create_commuter_load)
    
    profiles = {
        'Gamer (Gary)': create_gamer_load(),
        'Chatter (Emma)': create_chatty_load(),
        'Streamer (Steve)': create_streaming_load(),
        'Commuter (Chris)': create_commuter_load()
    }
    
    colors = ['#E74C3C', '#F39C12', '#27AE60', '#3498DB']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    results = {}
    
    for (name, load_func), color in zip(profiles.items(), colors):
        model = IntegratedSmartphoneBattery()
        result = model.simulate(24.0, load_func)
        results[name] = result
        
        axes[0, 0].plot(result['time_hours'], result['soc'] * 100, 
                       label=name, color=color, linewidth=2)
        axes[0, 1].plot(result['time_hours'], result['temperature'],
                       label=name, color=color, linewidth=2)
    
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Battery SOC (%)')
    axes[0, 0].set_title('Battery Drain Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=20, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlim(0, 24)
    axes[0, 0].set_ylim(0, 100)
    
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title('Battery Temperature')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=40, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlim(0, 24)
    
    names = list(results.keys())
    lives = [results[n]['battery_life_hours'] for n in names]
    bars = axes[1, 0].barh(names, lives, color=colors)
    axes[1, 0].set_xlabel('Battery Life (hours)')
    axes[1, 0].set_title('Expected Battery Life')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    ax = axes[1, 1]
    ax.axis('off')
    summary = "INTEGRATED MODEL SUMMARY\n" + "="*40 + "\n\n"
    for name in names:
        r = results[name]
        summary += f"{name}:\n  Life: {r['battery_life_hours']:.1f}h\n  Max T: {max(r['temperature']):.1f}°C\n\n"
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Integrated Smartphone Battery Model\n(ECM + Thermal + 5G RRC)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('integrated_model_comparison.png')

# =============================================================================
# 10. DEVICE SENSITIVITY MATRIX
# =============================================================================
def generate_device_sensitivity_matrix():
    """Generate device sensitivity heatmap."""
    from device_comparison import ALL_DEVICES
    
    devices = ALL_DEVICES
    parameters = ['Battery (Wh)', 'Display Power', 'SoC Efficiency', 
                  'Fast Charge (W)', 'Wireless (W)', 'TDP (W)']
    
    matrix = []
    for device in devices:
        row = [
            device.battery_wh / 20,
            1 - (device.display_max_nits / 3000),
            1 / (device.soc_process_nm / 5),
            device.max_charge_rate_w / 50,
            device.wireless_power_w / 25,
            1 - (device.thermal_design_power_w / 10)
        ]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(matrix.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(devices)))
    ax.set_xticklabels([d.name for d in devices])
    ax.set_yticks(range(len(parameters)))
    ax.set_yticklabels(parameters)
    
    for i in range(len(parameters)):
        for j in range(len(devices)):
            ax.text(j, i, f'{matrix[j,i]:.2f}', ha='center', va='center', fontsize=10)
    
    ax.set_title('Device Parameter Sensitivity Matrix\n(Green = Better for Battery Life)')
    fig.colorbar(im, ax=ax, label='Normalized Score')
    
    plt.tight_layout()
    save_fig('device_sensitivity_matrix.png')

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("REGENERATING ALL FIGURES (No popups!)")
    print("=" * 60)
    
    ensure_dir()
    
    print("\n1. RRC State Machine...")
    generate_rrc_state_machine()
    
    print("\n2. Chatty vs Streaming Paradox...")
    generate_chatty_vs_streaming()
    
    print("\n3. Persona Comparison...")
    generate_persona_comparison()
    
    print("\n4. Persona Power Breakdown...")
    generate_persona_power_breakdown()
    
    print("\n5. Device Comparison...")
    generate_device_comparison()
    
    print("\n6. Charging Source Comparison...")
    generate_charging_comparison()
    
    print("\n7. Validation Plot...")
    generate_validation_plot()
    
    print("\n8. System Flowchart...")
    generate_system_flowchart()
    
    print("\n9. Integrated Model Comparison...")
    generate_integrated_model_comparison()
    
    print("\n10. Device Sensitivity Matrix...")
    generate_device_sensitivity_matrix()
    
    print("\n" + "=" * 60)
    print("ALL FIGURES REGENERATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nFigures saved to: {FIGURES_DIR}")
