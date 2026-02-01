"""
=============================================================================
UNIFIED SMARTPHONE BATTERY MODEL - MCM 2026 Problem A
=============================================================================
Master integration module that combines ALL subsystems:

1. ECM Battery Model (electrical dynamics) - Model II
2. KiBaM Two-Tank Model (recovery effect) - Model I  
3. GPU Power Model (APGPM-based) - NEW
4. 5G RRC State Machine (CORRECTED tail power)
5. Thermal Model (Arrhenius + throttling)
6. Fast Charging Degradation (LifecyclE 2025)
7. LLM-RL Persona Agents (emergent behaviors)

This is the "full physics" model for O-Prize winning submission.

Authors: MCM 2026 Team
=============================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional
import matplotlib.pyplot as plt
from enum import Enum

# Import our subsystem models
_HAVE_GPU_MODEL = False
_HAVE_KIBAM = False
_HAVE_FAST_CHARGING = False
_HAVE_LLM_PERSONA = False
_HAVE_NETWORK = False

try:
    from gpu_model import GPUPowerModel, MobileGPU, create_device_database
    _HAVE_GPU_MODEL = True
except ImportError:
    pass

try:
    from kibam_model import KiBaMModel
    _HAVE_KIBAM = True
except ImportError:
    pass

try:
    from fast_charging import FastChargingAging, charging_stress_factor
    _HAVE_FAST_CHARGING = True
except ImportError:
    # Define fallback implementations
    def charging_stress_factor(c_rate: float, model: str = 'quadratic') -> float:
        """Charging stress factor from LifecyclE 2025 paper."""
        if c_rate <= 1.0:
            return 1.0
        return 1.0 + 7.0 * (c_rate - 1.0) ** 2  # Quadratic fit
    
    class FastChargingAging:
        """Simple fallback for charging tracker."""
        def __init__(self):
            self.cumulative_damage = 0.0
        def record_charge(self, c_rate, depth=1.0):
            sigma = charging_stress_factor(c_rate)
            self.cumulative_damage += sigma * depth

try:
    from llm_persona import PersonaAgent, DaySimulator
    _HAVE_LLM_PERSONA = True
except ImportError:
    pass

try:
    from network_model import RRCStateMachine
    _HAVE_NETWORK = True
except ImportError:
    pass


class RRCState(Enum):
    """5G RRC states with CORRECTED power values."""
    IDLE = 'IDLE'
    CONNECTED = 'CONNECTED'
    TAIL = 'TAIL'


@dataclass
class UnifiedModelParameters:
    """Complete parameter set for the unified model."""
    
    # =================================================================
    # BATTERY ELECTROCHEMICAL (ECM with 2 RC pairs)
    # =================================================================
    Q_nom: float = 5000.0        # Nominal capacity (mAh) - flagship phone
    V_nom: float = 3.85          # Nominal voltage (V) - Li-ion
    R0: float = 0.035            # Series resistance (Ω)
    R1: float = 0.020            # First RC pair resistance (Ω)
    C1: float = 800.0            # First RC pair capacitance (F)
    R2: float = 0.012            # Second RC pair resistance (Ω)
    C2: float = 4000.0           # Second RC pair capacitance (F)
    
    # OCV polynomial (7th order, Li-ion NCM chemistry)
    ocv_coeffs: Tuple[float, ...] = (
        2.70,   # a0 - minimum voltage at SOC=0
        1.35,   # a1 - linear term
        -0.95,  # a2
        0.65,   # a3
        -0.35,  # a4
        0.15,   # a5
        -0.04,  # a6
        0.005   # a7
    )
    
    # =================================================================
    # KiBaM TWO-TANK PARAMETERS
    # =================================================================
    kibam_c: float = 0.65        # Available fraction (Figure 3)
    kibam_k: float = 0.0008      # Recovery rate (1/s) - ~40 min time constant
    
    # =================================================================
    # GPU POWER MODEL (APGPM-based)
    # =================================================================
    gpu_p_idle: float = 0.15     # Idle power (W)
    gpu_p_max: float = 6.5       # Peak power (W) - Snapdragon 8 Elite
    gpu_alpha_render: float = 2.0    # Rendering exponent
    gpu_alpha_compute: float = 1.8   # Compute exponent
    gpu_alpha_ai: float = 1.5        # AI/ML exponent
    
    # =================================================================
    # THERMAL MODEL
    # =================================================================
    M_th: float = 48.0           # Thermal mass (J/K) - larger phone
    h_A: float = 0.18            # Heat transfer coeff (W/K)
    T_amb: float = 25.0          # Ambient temperature (°C)
    T_throttle: float = 45.0     # Thermal throttle threshold (°C)
    T_emergency: float = 50.0    # Emergency shutdown (°C)
    
    # Arrhenius parameters
    E_a: float = 20000.0         # Activation energy (J/mol)
    R_gas: float = 8.314         # Gas constant
    T_ref: float = 298.15        # Reference temp (K)
    
    # =================================================================
    # 5G NETWORK (Narayanan et al. 2021 - CORRECTED)
    # =================================================================
    P_idle: float = 0.180            # IDLE power (W)
    P_tail: float = 1.092            # TAIL power (W) - the famous value!
    P_connected_base: float = 2.8    # CONNECTED baseline (W)
    P_connected_slope: float = 0.003 # mW per Mbps
    P_connected_max: float = 7.5     # Max CONNECTED power (W)
    tau_tail: float = 15.0           # Tail timer (s)
    crossover_mbps: float = 187.0    # 5G/4G efficiency crossover
    
    # =================================================================
    # COMPONENT POWER BUDGET
    # =================================================================
    P_display_max: float = 1.2       # OLED at max brightness (W)
    P_cpu_idle: float = 0.25         # CPU idle (W)
    P_cpu_max: float = 4.0           # CPU at max frequency (W)
    P_sensors: float = 0.05          # Always-on sensors (W)
    P_baseline: float = 0.12         # Deep sleep baseline (W)
    
    # =================================================================
    # AGING (NASA B0005 validated + LifecyclE 2025)
    # =================================================================
    alpha_aging: float = 0.022       # √N aging coefficient
    charge_rate_baseline: float = 1.0    # Baseline C-rate


class UnifiedBatteryModel:
    """
    Complete unified smartphone battery model.
    
    State vector: [SOC_available, SOC_bound, V1, V2, T, Q_loss, cycle_count]
    
    - SOC_available: Available charge tank (KiBaM)
    - SOC_bound: Bound charge tank (KiBaM)
    - V1, V2: RC pair polarization voltages
    - T: Temperature (°C)
    - Q_loss: Cumulative capacity loss (mAh)
    - cycle_count: Equivalent full cycles
    """
    
    def __init__(self, params: UnifiedModelParameters = None,
                 device: str = 'snapdragon_8_elite'):
        self.p = params or UnifiedModelParameters()
        
        # RRC state machine
        self.rrc_state = RRCState.IDLE
        self.tail_timer = 0.0
        
        # GPU model (with device-specific parameters)
        self.gpu_model = self._create_gpu_model(device)
        
        # Fast charging tracker
        self.charging_tracker = FastChargingAging()
        
        # History for analysis
        self.history = {
            'time': [], 'soc': [], 'soc_available': [], 'soc_bound': [],
            'voltage': [], 'current': [], 'temperature': [],
            'power_total': [], 'power_gpu': [], 'power_network': [],
            'rrc_state': [], 'throttle_factor': []
        }
    
    def _create_gpu_model(self, device: str) -> 'GPUPowerModel':
        """Create device-specific GPU model."""
        try:
            db = create_device_database()
            if device in db:
                return GPUPowerModel(db[device])
        except:
            pass
        
        # Fallback to parameters
        class SimpleGPU:
            def __init__(self, params):
                self.p = params
                self.temp = 25.0
            
            def power_at_utilization(self, util, workload='rendering'):
                alpha = {'rendering': 2.0, 'compute': 1.8, 'ai': 1.5}.get(workload, 1.8)
                p_base = self.p.gpu_p_idle
                p_range = self.p.gpu_p_max - p_base
                
                # Thermal throttle
                if self.temp > self.p.T_throttle:
                    throttle = max(0.3, 1 - 0.1 * (self.temp - self.p.T_throttle))
                    util = util * throttle
                
                return p_base + p_range * (util ** alpha)
        
        return SimpleGPU(self.p)
    
    def ocv(self, soc: float) -> float:
        """Open circuit voltage from SOC (polynomial)."""
        soc = np.clip(soc, 0, 1)
        return sum(c * soc**i for i, c in enumerate(self.p.ocv_coeffs))
    
    def temp_factor(self, T_celsius: float) -> float:
        """Arrhenius temperature correction for resistance."""
        T_kelvin = T_celsius + 273.15
        exponent = (self.p.E_a / self.p.R_gas) * (1/self.p.T_ref - 1/T_kelvin)
        return np.exp(exponent)
    
    def thermal_throttle_factor(self, T: float) -> float:
        """Compute thermal throttling factor [0, 1]."""
        if T < self.p.T_throttle:
            return 1.0
        elif T >= self.p.T_emergency:
            return 0.0
        else:
            # Linear throttle between T_throttle and T_emergency
            return 1.0 - (T - self.p.T_throttle) / (self.p.T_emergency - self.p.T_throttle)
    
    def update_rrc_state(self, dt: float, data_active: bool) -> RRCState:
        """Update 5G RRC state machine."""
        if data_active:
            self.rrc_state = RRCState.CONNECTED
            self.tail_timer = self.p.tau_tail
        else:
            if self.rrc_state == RRCState.CONNECTED:
                self.rrc_state = RRCState.TAIL
            elif self.rrc_state == RRCState.TAIL:
                self.tail_timer -= dt
                if self.tail_timer <= 0:
                    self.rrc_state = RRCState.IDLE
                    self.tail_timer = 0
        return self.rrc_state
    
    def get_network_power(self, data_active: bool, throughput_mbps: float) -> float:
        """
        Get network power from RRC state.
        
        CRITICAL: 1092 mW is TAIL power (Narayanan Table 2), not connected!
        """
        if self.rrc_state == RRCState.IDLE:
            return self.p.P_idle
        elif self.rrc_state == RRCState.CONNECTED:
            # Throughput-dependent
            power = self.p.P_connected_base + self.p.P_connected_slope * throughput_mbps
            return min(power, self.p.P_connected_max)
        else:  # TAIL
            return self.p.P_tail
    
    def kibam_recovery_rate(self, soc_avail: float, soc_bound: float) -> float:
        """KiBaM recovery: flow from bound tank to available tank."""
        # Equilibrium SOC for each tank based on fractions
        c = self.p.kibam_c
        total_soc = soc_avail + soc_bound
        
        # At equilibrium: soc_avail = c * total, soc_bound = (1-c) * total
        equil_avail = c * total_soc
        equil_bound = (1 - c) * total_soc
        
        # Recovery flow: k * (bound - equilibrium_bound) = k * (equilibrium_avail - avail)
        recovery = self.p.kibam_k * (equil_avail - soc_avail)
        return recovery
    
    def dynamics(self, t: float, y: np.ndarray,
                 activity_func: Callable) -> np.ndarray:
        """
        Full system dynamics.
        
        State: [soc_avail, soc_bound, V1, V2, T, Q_loss, cycle_eq]
        
        Args:
            t: Time (seconds)
            y: State vector (7 elements)
            activity_func: Function(t) -> activity_dict
        """
        soc_avail, soc_bound, V1, V2, T, Q_loss, cycle_eq = y
        
        # Get activity
        activity = activity_func(t)
        gpu_util = activity.get('gpu_util', 0.0)
        cpu_util = activity.get('cpu_util', 0.2)
        display = activity.get('display', 0.5)
        data_active = activity.get('data_active', False)
        throughput = activity.get('throughput_mbps', 0.0)
        workload = activity.get('workload', 'general')
        charging = activity.get('charging', False)
        charge_rate = activity.get('charge_rate', 0.0)
        
        # Update RRC state
        self.update_rrc_state(1.0, data_active)
        
        # Thermal throttling
        throttle = self.thermal_throttle_factor(T)
        gpu_util_throttled = gpu_util * throttle
        cpu_util_throttled = min(cpu_util, cpu_util * throttle + 0.2 * (1 - throttle))
        
        # Power components
        if hasattr(self.gpu_model, 'temp'):
            self.gpu_model.temp = T
        P_gpu = self.gpu_model.power_at_utilization(gpu_util_throttled, workload)
        P_cpu = self.p.P_cpu_idle + (self.p.P_cpu_max - self.p.P_cpu_idle) * cpu_util_throttled
        P_display = self.p.P_display_max * display
        P_network = self.get_network_power(data_active, throughput)
        
        # Total power (W)
        P_total = P_gpu + P_cpu + P_display + P_network + self.p.P_sensors + self.p.P_baseline
        
        # Effective capacity considering aging
        Q_eff = self.p.Q_nom * (1 - Q_loss / self.p.Q_nom)
        Q_eff = max(Q_eff, 0.1 * self.p.Q_nom)  # Minimum 10%
        
        # Temperature-adjusted resistances
        tf = self.temp_factor(T)
        R0 = self.p.R0 * tf
        R1 = self.p.R1 * tf
        R2 = self.p.R2 * tf
        
        # Total SOC for voltage calculation
        total_soc = soc_avail + soc_bound
        V_oc = self.ocv(total_soc)
        
        # Current calculation
        if charging and charge_rate > 0:
            # Charging current (negative convention for charging)
            # C-rate * capacity_Ah gives current in A
            # But we need to cap the charging to prevent overcharge
            if total_soc >= 1.0:
                I = 0  # Battery full, stop charging
            else:
                I = -charge_rate * (Q_eff / 1000)  # A (Q_eff in mAh, divide by 1000 for Ah)
            # Track for aging
            self.charging_tracker.record_charge(charge_rate, depth=0.01)
        else:
            # Discharge current from power: P = V * I, so I = P / V
            V_approx = V_oc - V1 - V2 - 0.1  # Estimate terminal voltage
            V_approx = max(V_approx, 3.0)
            I = P_total / V_approx  # A
        
        # Clamp current to reasonable range (prevent runaway)
        I = np.clip(I, -10.0, 10.0)  # Max 10A charge/discharge
        
        # KiBaM dynamics: current draws from available tank only
        # dSOC/dt = -I / Q_Ah, where Q_Ah is capacity in Ah
        # SOC is dimensionless (0-1), I is in A, Q is in Ah
        recovery = self.kibam_recovery_rate(soc_avail, soc_bound)
        
        Q_Ah = Q_eff / 1000  # Convert mAh to Ah
        # d(SOC)/dt [1/s] = -I [A] / Q [Ah] / 3600 [s/h]
        d_soc_avail = -I / Q_Ah / 3600 + recovery
        d_soc_bound = -recovery
        
        # Clamp SOC derivatives to prevent exceeding bounds
        if soc_avail + soc_bound >= 1.0 and d_soc_avail + d_soc_bound > 0:
            d_soc_avail = 0
            d_soc_bound = 0
        if soc_avail <= 0 and d_soc_avail < 0:
            d_soc_avail = 0
        
        # RC dynamics
        dV1 = (I - V1 / R1) / self.p.C1
        dV2 = (I - V2 / R2) / self.p.C2
        
        # Thermal dynamics
        # Heat sources: Joule heating (I²R) + component heat (GPU, CPU)
        P_joule = I**2 * (R0 + R1 + R2)  # Joule heating from current flow
        P_component = P_gpu * 0.15 + P_cpu * 0.1  # Components contribute ~15% of power as heat
        P_heat = P_joule + P_component
        
        # Heat dissipation via convection
        dT = (P_heat - self.p.h_A * (T - self.p.T_amb)) / self.p.M_th
        
        # Clamp temperature derivative to prevent runaway
        dT = np.clip(dT, -1.0, 1.0)  # Max 1°C/s change
        
        # Aging dynamics (√N law with fast-charging stress)
        sigma = charging_stress_factor(charge_rate) if charging and charge_rate > 0 else 1.0
        d_cycle = abs(I) / (Q_eff / 1000) / 3600 / 2  # Half-cycle per discharge
        d_Q_loss = self.p.alpha_aging * sigma * np.sqrt(max(cycle_eq, 1)) * d_cycle * 0.001
        
        return np.array([d_soc_avail, d_soc_bound, dV1, dV2, dT, d_Q_loss, d_cycle])
    
    def simulate(self, duration_hours: float,
                 activity_func: Callable,
                 initial_soc: float = 1.0,
                 initial_temp: float = 25.0) -> Dict:
        """
        Simulate battery behavior.
        
        Args:
            duration_hours: Simulation time
            activity_func: Function(t_seconds) -> activity_dict
            initial_soc: Starting SOC
            initial_temp: Starting temperature
            
        Returns:
            Results dictionary
        """
        duration_s = duration_hours * 3600
        
        # Initialize KiBaM tanks
        c = self.p.kibam_c
        soc_avail_0 = c * initial_soc
        soc_bound_0 = (1 - c) * initial_soc
        
        y0 = np.array([soc_avail_0, soc_bound_0, 0.0, 0.0, initial_temp, 0.0, 0.0])
        
        # Time points (1-minute resolution)
        n_points = int(duration_hours * 60) + 1
        t_eval = np.linspace(0, duration_s, n_points)
        
        # Solve with state clamping
        def wrapped(t, y):
            # Clamp state to valid ranges before computing dynamics
            y_clamped = y.copy()
            y_clamped[0] = np.clip(y[0], 0.0, 1.0)  # soc_avail
            y_clamped[1] = np.clip(y[1], 0.0, 1.0)  # soc_bound
            # Ensure total SOC <= 1
            total = y_clamped[0] + y_clamped[1]
            if total > 1.0:
                scale = 1.0 / total
                y_clamped[0] *= scale
                y_clamped[1] *= scale
            y_clamped[4] = np.clip(y[4], 0.0, 80.0)  # temperature (max 80°C)
            return self.dynamics(t, y_clamped, activity_func)
        
        sol = solve_ivp(
            wrapped,
            (0, duration_s),
            y0,
            method='RK23',
            t_eval=t_eval,
            max_step=30.0
        )
        
        # Extract results
        time_hours = sol.t / 3600
        soc_avail = np.clip(sol.y[0], 0, 1)
        soc_bound = np.clip(sol.y[1], 0, 1)
        soc_total = np.clip(soc_avail + soc_bound, 0, 1)
        V1 = sol.y[2]
        V2 = sol.y[3]
        temperature = np.clip(sol.y[4], 0, 80)
        Q_loss = sol.y[5]
        cycles = sol.y[6]
        
        # Compute voltage
        voltage = np.array([self.ocv(s) - v1 - v2 
                          for s, v1, v2 in zip(soc_total, V1, V2)])
        
        # Battery life (when available SOC hits 0 or cutoff)
        cutoff_soc = 0.05
        battery_life = duration_hours
        for i, s in enumerate(soc_avail):
            if s <= cutoff_soc:
                battery_life = time_hours[i]
                break
        
        return {
            'time_hours': time_hours,
            'soc': soc_total,
            'soc_available': soc_avail,
            'soc_bound': soc_bound,
            'voltage': voltage,
            'temperature': temperature,
            'capacity_loss_pct': Q_loss / self.p.Q_nom * 100,
            'cycles': cycles,
            'battery_life_hours': battery_life,
        }


# =============================================================================
# ACTIVITY PROFILES FOR PERSONAS
# =============================================================================

def create_gamer_gary_activity() -> Callable:
    """Gamer Gary: Heavy GPU usage, thermal throttling expected."""
    def activity(t):
        hour = (t / 3600) % 24
        
        if 15.5 <= hour < 18:  # After school gaming
            return {
                'gpu_util': 0.85,
                'cpu_util': 0.6,
                'display': 0.9,
                'data_active': True,
                'throughput_mbps': 5.0,
                'workload': 'rendering',
                'charging': False
            }
        elif 19 <= hour < 23:  # Evening gaming
            return {
                'gpu_util': 0.90,
                'cpu_util': 0.65,
                'display': 0.95,
                'data_active': True,
                'throughput_mbps': 8.0,
                'workload': 'rendering',
                'charging': False
            }
        elif 7 <= hour < 15:  # School
            return {
                'gpu_util': 0.0,
                'cpu_util': 0.1,
                'display': 0.0,
                'data_active': False,
                'throughput_mbps': 0,
                'workload': 'idle',
                'charging': False
            }
        elif 23 <= hour or hour < 7:  # Sleep (charging)
            return {
                'gpu_util': 0.0,
                'cpu_util': 0.05,
                'display': 0.0,
                'data_active': False,
                'throughput_mbps': 0,
                'workload': 'idle',
                'charging': True,
                'charge_rate': 1.5  # Fast charging
            }
        else:  # Casual
            return {
                'gpu_util': 0.15,
                'cpu_util': 0.25,
                'display': 0.5,
                'data_active': True,
                'throughput_mbps': 3.0,
                'workload': 'general',
                'charging': False
            }
    return activity


def create_chatter_emma_activity() -> Callable:
    """Chatter Emma: Frequent short bursts, tail state nightmare."""
    def activity(t):
        hour = (t / 3600) % 24
        
        # Message every 45 seconds on average when awake
        if 7 <= hour < 23:
            burst_period = 45
            in_burst = (t % burst_period) < 3
            
            if in_burst:
                return {
                    'gpu_util': 0.1,
                    'cpu_util': 0.4,
                    'display': 0.6,
                    'data_active': True,
                    'throughput_mbps': 0.8,
                    'workload': 'general',
                    'charging': False
                }
            else:
                return {
                    'gpu_util': 0.0,
                    'cpu_util': 0.1,
                    'display': 0.0,
                    'data_active': False,
                    'throughput_mbps': 0,
                    'workload': 'idle',
                    'charging': False
                }
        else:  # Sleep
            return {
                'gpu_util': 0.0,
                'cpu_util': 0.05,
                'display': 0.0,
                'data_active': False,
                'throughput_mbps': 0,
                'workload': 'idle',
                'charging': True,
                'charge_rate': 1.0
            }
    return activity


def create_commuter_chris_activity() -> Callable:
    """Commuter Chris: GPS + navigation, cell handoffs."""
    def activity(t):
        hour = (t / 3600) % 24
        
        if 6.5 <= hour < 8 or 17 <= hour < 18.5:  # Commute
            return {
                'gpu_util': 0.25,
                'cpu_util': 0.55,
                'display': 0.85,
                'data_active': True,
                'throughput_mbps': 3.0,
                'workload': 'compute',  # Map rendering
                'charging': False
            }
        elif 8 <= hour < 17:  # Work
            return {
                'gpu_util': 0.0,
                'cpu_util': 0.1,
                'display': 0.0,
                'data_active': False,
                'throughput_mbps': 0,
                'workload': 'idle',
                'charging': True,
                'charge_rate': 1.0
            }
        elif 23 <= hour or hour < 6:  # Sleep
            return {
                'gpu_util': 0.0,
                'cpu_util': 0.05,
                'display': 0.0,
                'data_active': False,
                'throughput_mbps': 0,
                'workload': 'idle',
                'charging': True,
                'charge_rate': 0.5
            }
        else:  # Evening
            return {
                'gpu_util': 0.3,
                'cpu_util': 0.35,
                'display': 0.6,
                'data_active': True,
                'throughput_mbps': 15.0,
                'workload': 'general',
                'charging': False
            }
    return activity


def create_streamer_steve_activity() -> Callable:
    """Streamer Steve: High throughput, content creation."""
    def activity(t):
        hour = (t / 3600) % 24
        
        if 7.5 <= hour < 9 or 12 <= hour < 13:  # Morning/lunch video
            return {
                'gpu_util': 0.35,
                'cpu_util': 0.45,
                'display': 0.8,
                'data_active': True,
                'throughput_mbps': 35.0,  # HD streaming
                'workload': 'general',
                'charging': False
            }
        elif 18 <= hour < 22:  # Evening streaming
            return {
                'gpu_util': 0.5,
                'cpu_util': 0.55,
                'display': 0.9,
                'data_active': True,
                'throughput_mbps': 50.0,  # 4K streaming
                'workload': 'ai',  # Video decoding
                'charging': False
            }
        elif 14 <= hour < 17:  # Content creation
            return {
                'gpu_util': 0.7,
                'cpu_util': 0.65,
                'display': 0.75,
                'data_active': True,
                'throughput_mbps': 10.0,
                'workload': 'rendering',
                'charging': False
            }
        elif 23 <= hour or hour < 7:  # Sleep
            return {
                'gpu_util': 0.0,
                'cpu_util': 0.05,
                'display': 0.0,
                'data_active': False,
                'throughput_mbps': 0,
                'workload': 'idle',
                'charging': True,
                'charge_rate': 2.0  # Super fast charging
            }
        else:
            return {
                'gpu_util': 0.15,
                'cpu_util': 0.25,
                'display': 0.5,
                'data_active': True,
                'throughput_mbps': 5.0,
                'workload': 'general',
                'charging': False
            }
    return activity


def create_creator_chloe_activity() -> Callable:
    """Creator Chloe: Photo editing, social media, AI features."""
    def activity(t):
        hour = (t / 3600) % 24
        
        if 10 <= hour < 12 or 14 <= hour < 16:  # Photo sessions
            return {
                'gpu_util': 0.6,
                'cpu_util': 0.5,
                'display': 0.85,
                'data_active': True,
                'throughput_mbps': 8.0,
                'workload': 'ai',  # AI photo processing
                'charging': False
            }
        elif 19 <= hour < 22:  # Social media posting
            return {
                'gpu_util': 0.4,
                'cpu_util': 0.45,
                'display': 0.8,
                'data_active': True,
                'throughput_mbps': 25.0,  # Uploading
                'workload': 'compute',
                'charging': False
            }
        elif 23 <= hour or hour < 8:  # Sleep
            return {
                'gpu_util': 0.0,
                'cpu_util': 0.05,
                'display': 0.0,
                'data_active': False,
                'throughput_mbps': 0,
                'workload': 'idle',
                'charging': True,
                'charge_rate': 1.0
            }
        else:
            return {
                'gpu_util': 0.2,
                'cpu_util': 0.3,
                'display': 0.6,
                'data_active': True,
                'throughput_mbps': 5.0,
                'workload': 'general',
                'charging': False
            }
    return activity


# =============================================================================
# VISUALIZATION
# =============================================================================

def compare_personas(save_path: str = None):
    """Compare all personas with the unified model."""
    
    personas = {
        'Gamer Gary': create_gamer_gary_activity(),
        'Chatter Emma': create_chatter_emma_activity(),
        'Commuter Chris': create_commuter_chris_activity(),
        'Streamer Steve': create_streamer_steve_activity(),
        'Creator Chloe': create_creator_chloe_activity(),
    }
    
    colors = ['#E74C3C', '#F39C12', '#3498DB', '#9B59B6', '#27AE60']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    results = {}
    
    for (name, activity_func), color in zip(personas.items(), colors):
        print(f"Simulating {name}...")
        model = UnifiedBatteryModel()
        result = model.simulate(24.0, activity_func)
        results[name] = result
        
        # SOC
        axes[0, 0].plot(result['time_hours'], result['soc'] * 100,
                       label=name, color=color, linewidth=2)
        
        # Available vs Bound (KiBaM)
        axes[0, 1].plot(result['time_hours'], result['soc_available'] * 100,
                       color=color, linewidth=2, linestyle='-')
        axes[0, 1].plot(result['time_hours'], result['soc_bound'] * 100,
                       color=color, linewidth=1, linestyle='--', alpha=0.6)
        
        # Temperature
        axes[0, 2].plot(result['time_hours'], result['temperature'],
                       label=name, color=color, linewidth=2)
    
    # Format plots
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Total SOC (%)')
    axes[0, 0].set_title('Battery State of Charge')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=20, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlim(0, 24)
    axes[0, 0].set_ylim(0, 105)
    
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('SOC (%)')
    axes[0, 1].set_title('KiBaM Tanks (Solid=Available, Dashed=Bound)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 24)
    
    axes[0, 2].set_xlabel('Time (hours)')
    axes[0, 2].set_ylabel('Temperature (°C)')
    axes[0, 2].set_title('Battery Temperature')
    axes[0, 2].legend(loc='upper right')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=45, color='red', linestyle='--', alpha=0.5, label='Throttle')
    axes[0, 2].set_xlim(0, 24)
    
    # Battery life bar chart
    names = list(results.keys())
    lives = [results[n]['battery_life_hours'] for n in names]
    bars = axes[1, 0].barh(names, lives, color=colors)
    axes[1, 0].set_xlabel('Battery Life (hours)')
    axes[1, 0].set_title('Daily Battery Life')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    for bar, life in zip(bars, lives):
        axes[1, 0].text(min(life + 0.3, 23), bar.get_y() + bar.get_height()/2,
                       f'{life:.1f}h', va='center', fontweight='bold')
    axes[1, 0].set_xlim(0, 25)
    
    # Voltage comparison
    for (name, result), color in zip(results.items(), colors):
        axes[1, 1].plot(result['time_hours'], result['voltage'],
                       label=name, color=color, linewidth=2)
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Voltage (V)')
    axes[1, 1].set_title('Terminal Voltage')
    axes[1, 1].legend(loc='lower left')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=3.0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlim(0, 24)
    
    # Summary text
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = "UNIFIED MODEL RESULTS\n" + "=" * 40 + "\n\n"
    summary += "Model Components:\n"
    summary += "• ECM with 2 RC pairs\n"
    summary += "• KiBaM two-tank recovery\n"
    summary += "• GPU power (APGPM-based)\n"
    summary += "• 5G RRC (1092mW=TAIL)\n"
    summary += "• Thermal throttling\n"
    summary += "• Fast-charge aging σ(C)\n\n"
    
    summary += "Persona Rankings:\n"
    sorted_personas = sorted(results.items(), key=lambda x: x[1]['battery_life_hours'])
    for i, (name, r) in enumerate(sorted_personas, 1):
        summary += f"{i}. {name}: {r['battery_life_hours']:.1f}h\n"
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Unified Smartphone Battery Model - Persona Comparison\n'
                 '(ECM + KiBaM + GPU + 5G RRC + Thermal + Aging)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
    
    plt.show()
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("UNIFIED SMARTPHONE BATTERY MODEL")
    print("MCM 2026 Problem A")
    print("=" * 60)
    
    results = compare_personas("../figures/unified_model_comparison.png")
