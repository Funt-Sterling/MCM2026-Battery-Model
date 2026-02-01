"""
=============================================================================
INTEGRATED SMARTPHONE BATTERY MODEL
=============================================================================
Combines all subsystems into a unified continuous-time differential equation model:

1. ECM Battery Model (electrical)
2. Thermal Model (temperature coupling)  
3. RRC Network Model (5G state machine)
4. Component Load Models (OLED, CPU/GPU, sensors)
5. Aging Model (long-term degradation)

This is the "Model II" from the paper - the full physics-based approach.

Authors: MCM 2026 Team
Based on CALCE, NASA, and Narayanan et al. (2021)
=============================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Tuple, Dict, List, Callable
import matplotlib.pyplot as plt


@dataclass
class IntegratedModelParameters:
    """All parameters for the integrated model."""
    
    # Battery parameters (CALCE CS2 LCO cell)
    Q_nom: float = 2.0      # Nominal capacity (Ah)
    V_nom: float = 3.7      # Nominal voltage (V)
    R0: float = 0.025       # Series resistance (Ω)
    R1: float = 0.015       # First RC pair resistance (Ω)
    C1: float = 1000.0      # First RC pair capacitance (F)
    R2: float = 0.010       # Second RC pair resistance (Ω)
    C2: float = 5000.0      # Second RC pair capacitance (F)
    
    # OCV polynomial coefficients (LCO, normalized SOC 0-1)
    # V_oc(SOC) = a0 + a1*SOC + a2*SOC^2 + ... + a6*SOC^6
    ocv_coeffs: Tuple[float, ...] = (3.0, 0.8, -0.6, 0.4, -0.2, 0.1, -0.02)
    
    # Thermal parameters
    M_th: float = 42.0      # Thermal mass (J/K)
    h_A: float = 0.12       # Heat transfer coefficient (W/K)
    T_amb: float = 25.0     # Ambient temperature (°C)
    T_throttle: float = 40.0  # Throttling temperature (°C)
    
    # Arrhenius temperature coefficients
    E_a: float = 20000.0    # Activation energy (J/mol)
    R_gas: float = 8.314    # Gas constant (J/mol·K)
    T_ref: float = 298.15   # Reference temperature (K)
    
    # Network (5G mmWave from Narayanan et al. 2021 - CORRECTED)
    # CRITICAL: 1092 mW is TAIL power, NOT connected power!
    P_idle: float = 0.200         # Idle power (W) - radio sleeping
    P_tail: float = 1.092         # TAIL power (W) - the 1092 mW from Table 2!
    P_active_baseline: float = 3.0  # Active baseline power (W) at zero throughput
    P_active_slope: float = 0.003   # W per Mbps (3 mW/Mbps)
    
    # Energy efficiency crossover (Narayanan et al. 2021 Figure 11)
    crossover_throughput_mbps: float = 187.0  # Below this, 4G more efficient
    P_active_max: float = 8.0      # Maximum active power (W) at peak throughput
    tau_tail: float = 15.0        # Tail duration (s)
    
    # Aging parameters (√N law - fitted to NASA B0005 REAL data)
    alpha_aging: float = 0.022    # Aging coefficient (28.6% fade over 168 cycles)


class IntegratedSmartphoneBattery:
    """
    Full integrated smartphone battery model.
    
    State vector: [SOC, V1, V2, T, Q_loss]
    - SOC: State of charge (0-1)
    - V1, V2: RC pair voltages (V)
    - T: Temperature (°C)
    - Q_loss: Cumulative capacity loss (Ah)
    """
    
    def __init__(self, params: IntegratedModelParameters = None):
        self.p = params or IntegratedModelParameters()
        self.rrc_state = 'IDLE'
        self.tail_timer = 0.0
        self.cycle_count = 0.0
        
        # State history for analysis
        self.history = {
            'time': [], 'soc': [], 'voltage': [], 'current': [],
            'temperature': [], 'power': [], 'rrc_state': []
        }
    
    def ocv(self, soc: float) -> float:
        """Open circuit voltage as function of SOC."""
        soc = np.clip(soc, 0, 1)
        v = sum(c * soc**i for i, c in enumerate(self.p.ocv_coeffs))
        return v
    
    def docv_dsoc(self, soc: float) -> float:
        """Derivative of OCV with respect to SOC."""
        soc = np.clip(soc, 0.01, 0.99)
        dv = sum(i * c * soc**(i-1) for i, c in enumerate(self.p.ocv_coeffs) if i > 0)
        return dv
    
    def temp_factor(self, T_celsius: float) -> float:
        """Temperature correction factor (Arrhenius)."""
        T_kelvin = T_celsius + 273.15
        exponent = (self.p.E_a / self.p.R_gas) * (1/self.p.T_ref - 1/T_kelvin)
        return np.exp(exponent)
    
    def get_network_power(self, data_active: bool, throughput_mbps: float = 0) -> float:
        """
        Get network power based on RRC state and throughput.
        
        CORRECTED: Uses throughput-dependent model for CONNECTED state.
        P(T) = P_baseline + slope * T, clamped to P_max
        """
        if self.rrc_state == 'IDLE':
            return self.p.P_idle
        elif self.rrc_state == 'CONNECTED':
            # Throughput-dependent active power
            power = self.p.P_active_baseline + self.p.P_active_slope * throughput_mbps
            return min(power, self.p.P_active_max)
        else:  # TAIL - the 1092 mW value!
            return self.p.P_tail
    
    def update_rrc_state(self, dt: float, data_active: bool):
        """Update RRC state machine."""
        if data_active:
            self.rrc_state = 'CONNECTED'
            self.tail_timer = self.p.tau_tail
        else:
            if self.rrc_state == 'CONNECTED':
                self.rrc_state = 'TAIL'
            elif self.rrc_state == 'TAIL':
                self.tail_timer -= dt
                if self.tail_timer <= 0:
                    self.rrc_state = 'IDLE'
                    self.tail_timer = 0
    
    def dynamics(self, t: float, y: np.ndarray, 
                 load_func: Callable[[float], Tuple[float, bool, float]]) -> np.ndarray:
        """
        System dynamics as differential equations.
        
        Args:
            t: Time (s)
            y: State vector [SOC, V1, V2, T, Q_loss]
            load_func: Function that returns (power_W, data_active, throughput_mbps) at time t
        
        Returns:
            dy/dt: State derivatives
        """
        SOC, V1, V2, T, Q_loss = y
        
        # Get load
        P_load, data_active, throughput = load_func(t)
        
        # Update RRC state (discrete update, but we track it)
        self.update_rrc_state(0.1, data_active)
        P_network = self.get_network_power(data_active, throughput)
        
        # Total power
        P_total = P_load + P_network
        
        # Temperature correction
        temp_factor = self.temp_factor(T)
        R0_T = self.p.R0 * temp_factor
        R1_T = self.p.R1 * temp_factor
        R2_T = self.p.R2 * temp_factor
        
        # Effective capacity
        Q_eff = self.p.Q_nom * (1 - Q_loss / self.p.Q_nom)
        Q_eff = max(Q_eff, 0.1)  # Minimum capacity
        
        # Current from power and voltage
        V_oc = self.ocv(SOC)
        V_terminal = V_oc - V1 - V2  # Approximate
        I = P_total / max(V_terminal, 3.0) if V_terminal > 0 else 0
        
        # State equations
        # dSOC/dt = -I / Q_eff (in Ah, I in A, so convert)
        dSOC = -I / (Q_eff * 3600)  # 3600 to convert Ah to As
        
        # RC dynamics: dV/dt = (I - V/R) / C
        dV1 = (I - V1 / R1_T) / self.p.C1
        dV2 = (I - V2 / R2_T) / self.p.C2
        
        # Thermal: M_th * dT/dt = I²R - hA(T - T_amb)
        P_heat = I**2 * (R0_T + R1_T + R2_T)  # Joule heating
        dT = (P_heat - self.p.h_A * (T - self.p.T_amb)) / self.p.M_th
        
        # Aging: dQ_loss/dt based on cycle counting
        # Simplified: loss proportional to absolute current
        dQ_loss = self.p.alpha_aging * abs(I) / 3600  # Very slow aging
        
        return np.array([dSOC, dV1, dV2, dT, dQ_loss])
    
    def simulate(self, duration_hours: float, 
                 load_func: Callable[[float], Tuple[float, bool, float]],
                 initial_soc: float = 1.0,
                 initial_temp: float = 25.0) -> Dict:
        """
        Simulate battery behavior over time.
        
        Args:
            duration_hours: Simulation duration
            load_func: Function(t) -> (power_W, data_active, throughput_mbps)
            initial_soc: Initial state of charge (0-1)
            initial_temp: Initial temperature (°C)
        
        Returns:
            Dictionary with simulation results
        """
        duration_s = duration_hours * 3600
        
        # Initial state
        y0 = np.array([initial_soc, 0.0, 0.0, initial_temp, 0.0])
        
        # Time points
        t_eval = np.linspace(0, duration_s, int(duration_hours * 60))  # 1-min resolution
        
        # Solve ODE
        def wrapped_dynamics(t, y):
            return self.dynamics(t, y, load_func)
        
        sol = solve_ivp(
            wrapped_dynamics,
            (0, duration_s),
            y0,
            method='RK45',
            t_eval=t_eval,
            max_step=60.0  # Max 1 minute steps
        )
        
        # Extract results
        times_hours = sol.t / 3600
        soc = sol.y[0]
        V1 = sol.y[1]
        V2 = sol.y[2]
        temperature = sol.y[3]
        capacity_loss = sol.y[4]
        
        # Calculate voltage and power
        voltage = np.array([self.ocv(s) - v1 - v2 
                          for s, v1, v2 in zip(soc, V1, V2)])
        power = np.array([load_func(t)[0] for t in sol.t])
        
        # Find battery life (when SOC hits 0)
        battery_life = None
        for i, s in enumerate(soc):
            if s <= 0:
                battery_life = times_hours[i]
                break
        
        return {
            'time_hours': times_hours,
            'soc': soc,
            'voltage': voltage,
            'temperature': temperature,
            'power': power,
            'capacity_loss': capacity_loss,
            'battery_life_hours': battery_life or duration_hours
        }


# =============================================================================
# LOAD PROFILES
# =============================================================================

def create_gamer_load() -> Callable:
    """Gaming load profile - high GPU, thermal throttling."""
    def load(t):
        hour = (t / 3600) % 24
        if 15.5 <= hour < 18 or 19 <= hour < 22:  # Gaming sessions
            power = 6.5  # W (high)
            data_active = True
            throughput = 2.0  # Mbps
        elif 7 <= hour < 15:  # School
            power = 0.15
            data_active = False
            throughput = 0
        else:
            power = 0.8
            data_active = True
            throughput = 3.0
        return (power, data_active, throughput)
    return load


def create_chatty_load() -> Callable:
    """Chatty user - frequent small data bursts."""
    def load(t):
        hour = (t / 3600) % 24
        # Messages every 30-90 seconds on average
        if 7 <= hour < 23:  # Awake hours
            # Simulate bursts
            burst_period = 60  # Average 1 minute between messages
            in_burst = (t % burst_period) < 2  # 2 second burst
            if in_burst:
                power = 1.2
                data_active = True
                throughput = 0.5
            else:
                power = 0.5
                data_active = False
                throughput = 0
        else:
            power = 0.1
            data_active = False
            throughput = 0
        return (power, data_active, throughput)
    return load


def create_streaming_load() -> Callable:
    """Streaming user - continuous high throughput."""
    def load(t):
        hour = (t / 3600) % 24
        if 7.5 <= hour < 8.5 or 12 <= hour < 13 or 17.5 <= hour < 21:  # Streaming
            power = 2.0  # W
            data_active = True
            throughput = 25.0  # Mbps
        elif 22 <= hour < 24:  # Music
            power = 0.5
            data_active = True
            throughput = 0.3
        else:
            power = 0.15
            data_active = False
            throughput = 0
        return (power, data_active, throughput)
    return load


def create_commuter_load() -> Callable:
    """Commuter - GPS active, frequent handoffs."""
    def load(t):
        hour = (t / 3600) % 24
        if 6.5 <= hour < 8 or 17 <= hour < 18.5:  # Commute
            power = 2.5  # GPS + display
            data_active = True
            throughput = 2.0
        elif 8 <= hour < 17:  # Work
            power = 0.2
            data_active = False
            throughput = 0
        else:
            power = 1.0
            data_active = True
            throughput = 5.0
        return (power, data_active, throughput)
    return load


def compare_all_profiles(save_path: str = None):
    """Run simulation for all profiles and compare."""
    
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
        
        # SOC plot
        axes[0, 0].plot(result['time_hours'], result['soc'] * 100, 
                       label=name, color=color, linewidth=2)
        
        # Temperature plot
        axes[0, 1].plot(result['time_hours'], result['temperature'],
                       label=name, color=color, linewidth=2)
    
    # Format SOC plot
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Battery SOC (%)')
    axes[0, 0].set_title('Battery Drain Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=20, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlim(0, 24)
    axes[0, 0].set_ylim(0, 100)
    
    # Format temperature plot
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title('Battery Temperature')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=40, color='red', linestyle='--', alpha=0.5, label='Throttle')
    axes[0, 1].set_xlim(0, 24)
    
    # Battery life comparison bar chart
    names = list(results.keys())
    lives = [results[n]['battery_life_hours'] for n in names]
    bars = axes[1, 0].barh(names, lives, color=colors)
    axes[1, 0].set_xlabel('Battery Life (hours)')
    axes[1, 0].set_title('Expected Battery Life')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    for bar, life in zip(bars, lives):
        axes[1, 0].text(life + 0.2, bar.get_y() + bar.get_height()/2,
                       f'{life:.1f}h', va='center', fontweight='bold')
    
    # Final summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary = "INTEGRATED MODEL SUMMARY\n" + "="*40 + "\n\n"
    for name in names:
        r = results[name]
        final_soc = r['soc'][-1] * 100
        max_temp = max(r['temperature'])
        summary += f"{name}:\n"
        summary += f"  Battery life: {r['battery_life_hours']:.1f} hours\n"
        summary += f"  Max temperature: {max_temp:.1f}°C\n\n"
    
    summary += "Key Insight:\n"
    summary += "The integrated ECM + Thermal + 5G model\n"
    summary += "captures complex interactions between\n"
    summary += "usage patterns and battery behavior."
    
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Integrated Smartphone Battery Model\n(ECM + Thermal + 5G RRC)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return results


def plot_system_diagram(save_path: str = None):
    """Generate system block diagram for the paper."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw blocks
    blocks = {
        'battery': (0.15, 0.6, 0.18, 0.25),    # ECM
        'thermal': (0.15, 0.25, 0.18, 0.25),   # Thermal
        'network': (0.55, 0.6, 0.18, 0.25),    # 5G RRC
        'load': (0.55, 0.25, 0.18, 0.25),      # Load Model
        'output': (0.75, 0.42, 0.18, 0.16),    # Output
    }
    
    colors = {
        'battery': '#3498DB',
        'thermal': '#E74C3C', 
        'network': '#9B59B6',
        'load': '#27AE60',
        'output': '#F39C12'
    }
    
    labels = {
        'battery': 'ECM Battery\n$\\frac{dSOC}{dt} = -\\frac{I}{Q}$\n$\\frac{dV_i}{dt} = \\frac{I - V_i/R_i}{C_i}$',
        'thermal': 'Thermal Model\n$M_{th}\\frac{dT}{dt} = I^2R - hA(T-T_{amb})$',
        'network': '5G RRC\nIDLE → CONNECTED → TAIL\n$P = f(state)$',
        'load': 'Load Model\nOLED + CPU + GPU\n$P = \\sum P_i$',
        'output': 'State Output\nSOC, V, T, Life'
    }
    
    for name, (x, y, w, h) in blocks.items():
        rect = plt.Rectangle((x, y), w, h, fill=True, 
                             facecolor=colors[name], alpha=0.3,
                             edgecolor=colors[name], linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, labels[name], ha='center', va='center',
               fontsize=9, fontweight='bold')
    
    # Draw arrows
    # Battery → Thermal (bidirectional)
    ax.annotate('', xy=(0.24, 0.50), xytext=(0.24, 0.60),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(0.27, 0.55, 'T', fontsize=8)
    
    # Network → Load
    ax.annotate('', xy=(0.64, 0.50), xytext=(0.64, 0.60),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(0.66, 0.55, '$P_{net}$', fontsize=8)
    
    # Load → Battery
    ax.annotate('', xy=(0.33, 0.72), xytext=(0.55, 0.72),
               arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))
    ax.text(0.44, 0.75, 'I', fontsize=10)
    
    # Load → Thermal
    ax.annotate('', xy=(0.33, 0.37), xytext=(0.55, 0.37),
               arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))
    ax.text(0.44, 0.40, '$Q_{heat}$', fontsize=8)
    
    # All → Output
    ax.annotate('', xy=(0.75, 0.50), xytext=(0.73, 0.50),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Add title
    ax.text(0.5, 0.95, 'Integrated Smartphone Battery Model', 
           ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.90, '(Continuous-Time Differential Equation Framework)',
           ha='center', fontsize=10, transform=ax.transAxes)
    
    # Add equation box
    eq_text = """Master System:
$\\frac{d\\mathbf{x}}{dt} = f(\\mathbf{x}, \\mathbf{u}, t)$

where $\\mathbf{x} = [SOC, V_1, V_2, T, Q_{loss}]^T$
and $\\mathbf{u}$ = user behavior profile"""
    
    ax.text(0.02, 0.15, eq_text, fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
           transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Generating system diagram...")
    plot_system_diagram("../figures/system_flowchart.png")
    
    print("\nRunning integrated model comparison...")
    results = compare_all_profiles("../figures/integrated_model_comparison.png")
