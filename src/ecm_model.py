"""
=============================================================================
EQUIVALENT CIRCUIT MODEL (ECM) - Industry Standard Battery Model
=============================================================================
Based on MathWorks Simscape Battery documentation.

This implements the PROPER physics-based model that satisfies MCM requirements:
1. Kirchhoff's Voltage Law for terminal voltage
2. Coulomb counting for SOC (integral calculus)
3. RC dynamics for transient response
4. Lumped thermal mass heat equation
5. Cycling aging with √N relationship

Reference: MathWorks Simscape Battery - Battery Equivalent Circuit Block
=============================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Tuple, Dict, List, Optional
import warnings

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

R_GAS = 8.314       # Universal gas constant [J/(mol·K)]
F_CONST = 96485     # Faraday constant [C/mol]
K_BOLTZ = 8.617e-5  # Boltzmann constant [eV/K]


# =============================================================================
# SECTION 1: OCV LOOKUP TABLE (SOC-dependent Open Circuit Voltage)
# =============================================================================

# Standard Li-ion OCV curve (from MathWorks default parameters)
# SOC breakpoints: [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
# OCV values at 25°C: [3.0, 3.5, 3.63, 3.71, 3.93, 4.08, 4.2] V

SOC_BREAKPOINTS = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
OCV_25C = np.array([3.0, 3.5057, 3.566, 3.6337, 3.7127, 3.9259, 4.0777, 4.1928])

# Temperature-dependent OCV (at 5°C, 20°C, 40°C)
TEMP_BREAKPOINTS = np.array([278, 293, 313])  # Kelvin (5°C, 20°C, 40°C)
OCV_THERMAL = np.array([
    [3.49, 3.50, 3.51],  # SOC = 0
    [3.55, 3.57, 3.56],  # SOC = 0.1
    [3.62, 3.63, 3.64],  # SOC = 0.25
    [3.71, 3.71, 3.72],  # SOC = 0.5
    [3.91, 3.93, 3.94],  # SOC = 0.75
    [4.07, 4.08, 4.08],  # SOC = 0.9
    [4.19, 4.19, 4.19],  # SOC = 1.0
])


def get_ocv(soc: float, T: float = 298.15) -> float:
    """
    Get Open Circuit Voltage as function of SOC and Temperature.
    Uses bilinear interpolation from lookup tables.
    
    Args:
        soc: State of charge [0, 1]
        T: Temperature [K]
    
    Returns:
        OCV in Volts
    """
    soc = np.clip(soc, 0, 1)
    T = np.clip(T, 263, 323)  # Limit to -10°C to 50°C
    
    # Simple linear interpolation for SOC at reference temp
    ocv_interp = interp1d(SOC_BREAKPOINTS, OCV_THERMAL[:, 1], 
                          kind='linear', fill_value='extrapolate')
    
    # Temperature correction (simplified Arrhenius-like)
    T_ref = 298.15
    dOCV_dT = -0.0003  # Typical entropic coefficient [V/K]
    
    return float(ocv_interp(soc)) + dOCV_dT * (T - T_ref)


# =============================================================================
# SECTION 2: EQUIVALENT CIRCUIT PARAMETERS
# =============================================================================

@dataclass
class ECMParameters:
    """
    Equivalent Circuit Model Parameters.
    
    Based on MathWorks Simscape Battery defaults with smartphone-typical values.
    
    The equivalent circuit is:
    
        OCV(SOC,T) ─┬─ R0 ─┬─ R1║C1 ─┬─ R2║C2 ─┬─ Terminal
                    │      │        │        │
                   GND    GND      GND      GND
    
    Where:
    - OCV: Open Circuit Voltage (SOC and T dependent)
    - R0: Instantaneous (ohmic) resistance
    - R1, C1: First RC pair (activation polarization, fast ~30s)
    - R2, C2: Second RC pair (concentration polarization, slow ~300s)
    """
    
    # === Battery Capacity ===
    Q_nom: float = 4.0            # Nominal capacity [Ah] (4000 mAh)
    V_nom: float = 3.7            # Nominal voltage [V]
    V_max: float = 4.2            # Max voltage (fully charged)
    V_min: float = 3.0            # Cutoff voltage
    
    # === Instantaneous Resistance R0(SOC, T) ===
    # Values in Ohms, from MathWorks defaults scaled for smartphone
    R0_ref: float = 0.085         # Reference R0 at 25°C, 50% SOC [Ohm]
    R0_soc_factor: np.ndarray = field(default_factory=lambda: 
        np.array([1.1, 1.0, 1.02, 0.97, 0.98, 1.0, 1.0]))  # SOC multipliers
    R0_temp_coeff: float = 0.02   # Temperature coefficient [1/K]
    
    # === First RC Pair (Activation Polarization) ===
    R1_ref: float = 0.029         # Reference R1 [Ohm]
    tau1_ref: float = 36.0        # Time constant τ1 = R1*C1 [seconds]
    # SOC-dependent time constants from MathWorks
    tau1_soc: np.ndarray = field(default_factory=lambda:
        np.array([36, 45, 105, 29, 77, 33, 39]))
    
    # === Second RC Pair (Concentration Polarization) ===
    R2_ref: float = 0.020         # Reference R2 [Ohm]
    tau2_ref: float = 300.0       # Time constant τ2 [seconds]
    tau2_soc: np.ndarray = field(default_factory=lambda:
        np.array([200, 250, 400, 180, 350, 200, 220]))
    
    # === Thermal Parameters ===
    M_th: float = 45.0            # Thermal mass [J/K] (battery + casing)
    R_th: float = 8.0             # Thermal resistance to ambient [K/W]
    
    # === Aging Parameters (√N relationship from MathWorks) ===
    N_ref: float = 500            # Reference number of cycles for aging calc
    delta_C_N: float = -20.0      # % change in capacity after N_ref cycles
    delta_R_N: float = 30.0       # % change in resistance after N_ref cycles
    n_cycles: float = 0           # Current cycle count
    
    # === Self-Discharge ===
    R_sd: float = 7000.0          # Self-discharge resistance [Ohm]


# =============================================================================
# SECTION 3: TEMPERATURE-DEPENDENT PARAMETER FUNCTIONS
# =============================================================================

def get_R0(params: ECMParameters, soc: float, T: float) -> float:
    """
    Get instantaneous resistance R0(SOC, T).
    
    R0 increases at:
    - Low temperatures (ion mobility decreases)
    - Low/high SOC (concentration gradients)
    
    Based on Arrhenius-type relationship.
    """
    T_ref = 298.15
    
    # SOC factor (interpolate)
    soc_idx = np.clip(soc, 0, 1) * 6  # Map to [0, 6]
    soc_floor = int(np.floor(soc_idx))
    soc_ceil = min(soc_floor + 1, 6)
    alpha = soc_idx - soc_floor
    soc_factor = (1 - alpha) * params.R0_soc_factor[soc_floor] + \
                 alpha * params.R0_soc_factor[soc_ceil]
    
    # Temperature factor (resistance increases at low T)
    temp_factor = np.exp(params.R0_temp_coeff * (T_ref - T))
    
    return params.R0_ref * soc_factor * temp_factor


def get_R1_tau1(params: ECMParameters, soc: float, T: float) -> Tuple[float, float]:
    """Get first RC pair parameters R1 and τ1."""
    # SOC interpolation for tau1
    soc_idx = np.clip(soc, 0, 1) * 6
    soc_floor = int(np.floor(soc_idx))
    soc_ceil = min(soc_floor + 1, 6)
    alpha = soc_idx - soc_floor
    tau1 = (1 - alpha) * params.tau1_soc[soc_floor] + \
           alpha * params.tau1_soc[soc_ceil]
    
    # Temperature correction (faster dynamics at higher T)
    T_ref = 298.15
    tau1 = tau1 * np.exp(0.01 * (T_ref - T))
    
    # R1 scales with temperature similar to R0
    R1 = params.R1_ref * np.exp(params.R0_temp_coeff * (T_ref - T))
    
    return R1, tau1


def get_R2_tau2(params: ECMParameters, soc: float, T: float) -> Tuple[float, float]:
    """Get second RC pair parameters R2 and τ2."""
    soc_idx = np.clip(soc, 0, 1) * 6
    soc_floor = int(np.floor(soc_idx))
    soc_ceil = min(soc_floor + 1, 6)
    alpha = soc_idx - soc_floor
    tau2 = (1 - alpha) * params.tau2_soc[soc_floor] + \
           alpha * params.tau2_soc[soc_ceil]
    
    T_ref = 298.15
    tau2 = tau2 * np.exp(0.01 * (T_ref - T))
    R2 = params.R2_ref * np.exp(params.R0_temp_coeff * (T_ref - T))
    
    return R2, tau2


def get_aged_capacity(params: ECMParameters) -> float:
    """
    Calculate aged capacity using √N relationship.
    
    From MathWorks documentation:
    C_faded = C * (1 + δC/100 * √(n/N_ref))
    
    Where δC is negative (capacity decreases).
    """
    if params.n_cycles <= 0:
        return params.Q_nom
    
    fade_factor = 1 + (params.delta_C_N / 100) * np.sqrt(params.n_cycles / params.N_ref)
    return params.Q_nom * max(0.5, fade_factor)  # Minimum 50% capacity


def get_aged_resistance_factor(params: ECMParameters) -> float:
    """
    Calculate resistance increase factor due to aging.
    
    R_faded = R * (1 + δR/100 * √(n/N_ref))
    """
    if params.n_cycles <= 0:
        return 1.0
    
    return 1 + (params.delta_R_N / 100) * np.sqrt(params.n_cycles / params.N_ref)


# =============================================================================
# SECTION 4: THE EQUIVALENT CIRCUIT MODEL CLASS
# =============================================================================

class EquivalentCircuitModel:
    """
    Equivalent Circuit Model (ECM) for Li-ion Battery.
    
    Implements the governing equations from MathWorks Simscape Battery:
    
    === ELECTRICAL MODEL ===
    Terminal Voltage (Kirchhoff's Voltage Law):
        U = OCV(SOC, T) - I·R0 - ΔU_RC1 - ΔU_RC2
    
    RC Dynamics (First-order ODEs):
        τ1 · d(ΔU_RC1)/dt + ΔU_RC1 = I · R1
        τ2 · d(ΔU_RC2)/dt + ΔU_RC2 = I · R2
    
    State of Charge (Coulomb Counting - Integral):
        dSOC/dt = -I / (C_aged · 3600)
    
    === THERMAL MODEL ===
    Lumped Thermal Mass:
        M_th · dT/dt = Q_gen - Q_diss
    
    Heat Generation:
        Q_gen = I²·R0 + ΔU_RC1·I + ΔU_RC2·I + Q_rev
    
    Heat Dissipation (Newton's Cooling):
        Q_diss = (T - T_amb) / R_th
    
    State Vector: [SOC, ΔU_RC1, ΔU_RC2, T]
    """
    
    def __init__(self, params: ECMParameters = None):
        self.params = params if params else ECMParameters()
        self.history = []
        
    def terminal_voltage(self, soc: float, T: float, I: float,
                        dU_RC1: float, dU_RC2: float) -> float:
        """
        Calculate terminal voltage using Kirchhoff's Voltage Law.
        
        U = OCV(SOC,T) - η_inst - η_dyn
        
        Where:
        - η_inst = I · R0  (instantaneous overpotential)
        - η_dyn = ΔU_RC1 + ΔU_RC2  (dynamic overpotential)
        """
        p = self.params
        age_factor = get_aged_resistance_factor(p)
        
        OCV = get_ocv(soc, T)
        R0 = get_R0(p, soc, T) * age_factor
        
        # Sign convention: I > 0 for discharge
        eta_inst = I * R0
        eta_dyn = dU_RC1 + dU_RC2
        
        return OCV - eta_inst - eta_dyn
    
    def heat_generation(self, soc: float, T: float, I: float,
                       dU_RC1: float, dU_RC2: float) -> float:
        """
        Calculate heat generation rate [W].
        
        Q_gen = P_diss + Q_rev
        
        Where:
        - P_diss = I²R0 + ΔU_RC1·I + ΔU_RC2·I  (irreversible Joule heating)
        - Q_rev = I·T·dOCV/dT  (reversible entropic heat)
        """
        p = self.params
        age_factor = get_aged_resistance_factor(p)
        R0 = get_R0(p, soc, T) * age_factor
        
        # Joule heating (irreversible)
        P_ohmic = I**2 * R0
        P_RC = abs(dU_RC1 * I) + abs(dU_RC2 * I)
        P_diss = P_ohmic + P_RC
        
        # Entropic heat (reversible) - typically small
        dOCV_dT = -0.0003  # V/K (typical for Li-ion)
        Q_rev = I * T * dOCV_dT
        
        return P_diss + Q_rev
    
    def ecm_ode(self, t: float, y: np.ndarray, 
                I_func: Callable, T_amb: float) -> np.ndarray:
        """
        The ECM governing differential equations.
        
        State vector y = [SOC, ΔU_RC1, ΔU_RC2, T]
        
        Returns dy/dt = [dSOC/dt, dΔU_RC1/dt, dΔU_RC2/dt, dT/dt]
        """
        soc, dU_RC1, dU_RC2, T = y
        p = self.params
        
        # Clamp states to physical bounds
        soc = np.clip(soc, 0, 1)
        T = np.clip(T, 253, 353)  # -20°C to 80°C
        
        # Get current
        I = I_func(t) / 1000.0  # Convert mA to A
        
        # Get parameters at current state
        age_factor = get_aged_resistance_factor(p)
        C_aged = get_aged_capacity(p)
        R1, tau1 = get_R1_tau1(p, soc, T)
        R2, tau2 = get_R2_tau2(p, soc, T)
        R1 *= age_factor
        R2 *= age_factor
        
        # === STATE OF CHARGE (Coulomb Counting) ===
        # dSOC/dt = -I / (C_aged * 3600)
        # Note: I > 0 for discharge, so SOC decreases
        dSOC_dt = -I / (C_aged * 3600)
        
        # === RC DYNAMICS ===
        # τ · d(ΔU)/dt + ΔU = I · R
        # => d(ΔU)/dt = (I·R - ΔU) / τ
        dU_RC1_dt = (I * R1 - dU_RC1) / tau1
        dU_RC2_dt = (I * R2 - dU_RC2) / tau2
        
        # === THERMAL DYNAMICS ===
        # M_th · dT/dt = Q_gen - (T - T_amb)/R_th
        Q_gen = self.heat_generation(soc, T, I, dU_RC1, dU_RC2)
        Q_diss = (T - T_amb) / p.R_th
        dT_dt = (Q_gen - Q_diss) / p.M_th
        
        return np.array([dSOC_dt, dU_RC1_dt, dU_RC2_dt, dT_dt])
    
    def simulate(self,
                 I_func: Callable[[float], float],
                 t_span: Tuple[float, float],
                 SOC_0: float = 1.0,
                 T_amb: float = 298.15,
                 T_0: float = None,
                 dt_eval: float = 60.0) -> Dict:
        """
        Simulate the ECM battery model.
        
        Args:
            I_func: Current function I(t) [mA], positive for discharge
            t_span: (t_start, t_end) in seconds
            SOC_0: Initial state of charge [0, 1]
            T_amb: Ambient temperature [K]
            T_0: Initial battery temperature [K]
            dt_eval: Output time step [s]
        
        Returns:
            Dictionary with time series data
        """
        if T_0 is None:
            T_0 = T_amb
        
        # Initial state: [SOC, ΔU_RC1, ΔU_RC2, T]
        y0 = np.array([SOC_0, 0.0, 0.0, T_0])
        
        t_eval = np.arange(t_span[0], t_span[1], dt_eval)
        
        # Event: battery empty (SOC < 1%)
        def battery_empty(t, y):
            return y[0] - 0.01
        battery_empty.terminal = True
        battery_empty.direction = -1
        
        # Event: voltage cutoff
        def voltage_cutoff(t, y):
            soc, dU_RC1, dU_RC2, T = y
            I = I_func(t) / 1000.0
            V = self.terminal_voltage(soc, T, I, dU_RC1, dU_RC2)
            return V - self.params.V_min
        voltage_cutoff.terminal = True
        voltage_cutoff.direction = -1
        
        # Solve ODE system
        sol = solve_ivp(
            lambda t, y: self.ecm_ode(t, y, I_func, T_amb),
            t_span,
            y0,
            method='RK45',
            t_eval=t_eval,
            events=[battery_empty, voltage_cutoff],
            max_step=30.0  # Max 30s steps for accuracy
        )
        
        # Extract states
        SOC = sol.y[0]
        dU_RC1 = sol.y[1]
        dU_RC2 = sol.y[2]
        T = sol.y[3]
        
        # Calculate derived quantities
        current = np.array([I_func(t) for t in sol.t])
        current_A = current / 1000.0
        
        voltage = np.array([
            self.terminal_voltage(soc, temp, I, du1, du2)
            for soc, temp, I, du1, du2 in zip(SOC, T, current_A, dU_RC1, dU_RC2)
        ])
        
        power = voltage * current_A * 1000  # mW
        
        # Determine time-to-empty
        tte = None
        if len(sol.t_events[0]) > 0 or len(sol.t_events[1]) > 0:
            events = list(sol.t_events[0]) + list(sol.t_events[1])
            if events:
                tte = min(events) / 3600.0
        
        return {
            't': sol.t,
            't_hours': sol.t / 3600.0,
            'SOC': SOC,
            'SOC_percent': SOC * 100,
            'voltage': voltage,
            'current': current,
            'current_A': current_A,
            'power_mW': power,
            'temperature': T,
            'T_celsius': T - 273.15,
            'dU_RC1': dU_RC1,
            'dU_RC2': dU_RC2,
            'time_to_empty': tte
        }


# =============================================================================
# SECTION 5: SMARTPHONE LOAD MODEL (The "Bridge")
# =============================================================================

@dataclass
class SmartphoneLoadModel:
    """
    Maps smartphone activities to current draw.
    
    This is the "Bridge Model" that connects user activities to the
    physical battery current input I(t).
    
    I_total(t) = I_base + I_screen(b,r) + I_cpu(u) + I_network(m,s) + I_gps + I_sensors
    
    Where:
    - b: brightness [0-1]
    - r: refresh rate [Hz]
    - u: CPU utilization [0-1]
    - m: network mode ('wifi', '4g', '5g')
    - s: signal strength [0-1]
    """
    
    # Baseline currents [mA]
    I_base: float = 15.0          # Always-on (RTC, RAM refresh)
    
    # Screen model: I = I_min + (I_max - I_min) * brightness
    I_screen_min: float = 30.0    # Screen on, min brightness [mA]
    I_screen_max: float = 350.0   # Screen on, max brightness [mA]
    I_screen_off: float = 0.0     # Screen off
    
    # Refresh rate multiplier (120Hz vs 60Hz)
    refresh_60Hz: float = 1.0
    refresh_120Hz: float = 1.3    # 30% more power at 120Hz
    
    # CPU model: I = I_idle + (I_max - I_idle) * utilization
    I_cpu_idle: float = 30.0      # CPU idle [mA]
    I_cpu_max: float = 1500.0     # CPU max load [mA]
    
    # Network model [mA]
    I_wifi: float = 50.0
    I_4g_idle: float = 80.0
    I_4g_active: float = 250.0
    I_5g_idle: float = 120.0
    I_5g_active: float = 450.0
    
    # GPS [mA]
    I_gps: float = 180.0
    
    # Sensors [mA]
    I_sensors_base: float = 5.0
    I_accelerometer: float = 10.0
    I_gyroscope: float = 15.0
    I_proximity: float = 5.0
    
    # Bluetooth [mA]
    I_bluetooth_idle: float = 5.0
    I_bluetooth_active: float = 40.0
    
    def calculate_current(self,
                          screen_on: bool = True,
                          brightness: float = 0.5,
                          refresh_rate: int = 60,
                          cpu_util: float = 0.3,
                          network_mode: str = 'wifi',
                          network_active: bool = True,
                          gps_on: bool = False,
                          bluetooth_on: bool = False,
                          sensors_active: bool = True) -> float:
        """
        Calculate total current draw from component states.
        
        Returns:
            Total current in mA
        """
        I_total = self.I_base
        
        # Screen
        if screen_on:
            brightness = np.clip(brightness, 0, 1)
            I_screen = self.I_screen_min + \
                       (self.I_screen_max - self.I_screen_min) * brightness
            if refresh_rate >= 90:
                I_screen *= self.refresh_120Hz
            I_total += I_screen
        
        # CPU
        cpu_util = np.clip(cpu_util, 0, 1)
        I_total += self.I_cpu_idle + (self.I_cpu_max - self.I_cpu_idle) * cpu_util
        
        # Network
        if network_mode == 'wifi':
            I_total += self.I_wifi if network_active else self.I_wifi * 0.3
        elif network_mode == '4g':
            I_total += self.I_4g_active if network_active else self.I_4g_idle
        elif network_mode == '5g':
            I_total += self.I_5g_active if network_active else self.I_5g_idle
        
        # GPS
        if gps_on:
            I_total += self.I_gps
        
        # Bluetooth
        if bluetooth_on:
            I_total += self.I_bluetooth_idle
        
        # Sensors
        if sensors_active:
            I_total += self.I_sensors_base + self.I_accelerometer
        
        return I_total

    def get_current_breakdown(self,
                              screen_on: bool = True,
                              brightness: float = 0.5,
                              refresh_rate: int = 60,
                              cpu_util: float = 0.3,
                              network_mode: str = 'wifi',
                              network_active: bool = True,
                              gps_on: bool = False,
                              bluetooth_on: bool = False,
                              sensors_active: bool = True) -> Dict[str, float]:
        """
        Get breakdown of current draw by component.
        
        Returns:
            Dictionary {component_name: current_mA}
        """
        breakdown = {'baseline': self.I_base}
        
        # Screen
        if screen_on:
            brightness = np.clip(brightness, 0, 1)
            I_screen = self.I_screen_min + \
                       (self.I_screen_max - self.I_screen_min) * brightness
            if refresh_rate >= 90:
                I_screen *= self.refresh_120Hz
            breakdown['screen'] = I_screen
        else:
            breakdown['screen'] = 0.0
        
        # CPU
        cpu_util = np.clip(cpu_util, 0, 1)
        breakdown['cpu'] = self.I_cpu_idle + (self.I_cpu_max - self.I_cpu_idle) * cpu_util
        
        # Network
        I_net = 0.0
        if network_mode == 'wifi':
            I_net = self.I_wifi if network_active else self.I_wifi * 0.3
        elif network_mode == '4g':
            I_net = self.I_4g_active if network_active else self.I_4g_idle
        elif network_mode == '5g':
            I_net = self.I_5g_active if network_active else self.I_5g_idle
        breakdown['network'] = I_net
        
        # GPS
        breakdown['gps'] = self.I_gps if gps_on else 0.0
        
        # Bluetooth
        breakdown['bluetooth'] = self.I_bluetooth_idle if bluetooth_on else 0.0
        
        # Sensors
        breakdown['sensors'] = (self.I_sensors_base + self.I_accelerometer) if sensors_active else 0.0
        
        return breakdown


# =============================================================================
# SECTION 6: PREDEFINED USAGE SCENARIOS
# =============================================================================

def create_scenario_current_func(scenario: str, load_model: SmartphoneLoadModel = None) -> Callable:
    """
    Create current function for predefined usage scenarios.
    """
    if load_model is None:
        load_model = SmartphoneLoadModel()
    
    scenarios = {
        'idle': lambda t: load_model.calculate_current(
            screen_on=False, cpu_util=0.05, network_mode='wifi', network_active=False
        ),
        'light': lambda t: load_model.calculate_current(
            screen_on=True, brightness=0.4, cpu_util=0.15, network_mode='wifi'
        ),
        'moderate': lambda t: load_model.calculate_current(
            screen_on=True, brightness=0.5, cpu_util=0.35, network_mode='4g'
        ),
        'heavy': lambda t: load_model.calculate_current(
            screen_on=True, brightness=0.7, cpu_util=0.6, network_mode='4g', 
            refresh_rate=120
        ),
        'gaming': lambda t: load_model.calculate_current(
            screen_on=True, brightness=0.9, cpu_util=0.95, network_mode='wifi',
            refresh_rate=120, sensors_active=True
        ),
        'navigation': lambda t: load_model.calculate_current(
            screen_on=True, brightness=0.8, cpu_util=0.5, network_mode='4g',
            gps_on=True
        ),
        'video_call': lambda t: load_model.calculate_current(
            screen_on=True, brightness=0.6, cpu_util=0.5, network_mode='wifi',
            network_active=True, bluetooth_on=True
        ),
    }
    
    return scenarios.get(scenario, scenarios['moderate'])


def realistic_day_profile(load_model: SmartphoneLoadModel = None) -> Callable:
    """
    Realistic daily usage pattern.
    
    Models usage from wake to sleep with realistic variations.
    """
    if load_model is None:
        load_model = SmartphoneLoadModel()
    
    def current_func(t):
        hour = (t / 3600.0) % 24
        
        if 0 <= hour < 7:    # Sleep
            return load_model.calculate_current(screen_on=False, cpu_util=0.02)
        elif 7 <= hour < 8:  # Morning routine
            return load_model.calculate_current(screen_on=True, brightness=0.5, 
                                                cpu_util=0.3, network_mode='wifi')
        elif 8 <= hour < 9:  # Commute (navigation)
            return load_model.calculate_current(screen_on=True, brightness=0.8,
                                                cpu_util=0.5, network_mode='4g', gps_on=True)
        elif 9 <= hour < 12: # Work morning
            return load_model.calculate_current(screen_on=True, brightness=0.4,
                                                cpu_util=0.2, network_mode='wifi')
        elif 12 <= hour < 13: # Lunch (social media)
            return load_model.calculate_current(screen_on=True, brightness=0.6,
                                                cpu_util=0.5, network_mode='wifi', refresh_rate=120)
        elif 13 <= hour < 17: # Work afternoon
            return load_model.calculate_current(screen_on=True, brightness=0.4,
                                                cpu_util=0.25, network_mode='wifi')
        elif 17 <= hour < 18: # Commute
            return load_model.calculate_current(screen_on=True, brightness=0.8,
                                                cpu_util=0.5, network_mode='4g', gps_on=True)
        elif 18 <= hour < 21: # Evening (video/gaming)
            return load_model.calculate_current(screen_on=True, brightness=0.7,
                                                cpu_util=0.6, network_mode='wifi', 
                                                refresh_rate=120)
        elif 21 <= hour < 23: # Wind down
            return load_model.calculate_current(screen_on=True, brightness=0.3,
                                                cpu_util=0.2, network_mode='wifi')
        else:  # Late night
            return load_model.calculate_current(screen_on=False, cpu_util=0.02)
    
    return current_func


# =============================================================================
# SECTION 7: MAIN DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EQUIVALENT CIRCUIT MODEL (ECM) - Industry Standard Battery Model")
    print("Based on MathWorks Simscape Battery Documentation")
    print("="*70)
    
    # Initialize
    params = ECMParameters()
    model = EquivalentCircuitModel(params)
    load_model = SmartphoneLoadModel()
    
    print("\n[1] Battery Parameters:")
    print(f"    Nominal Capacity: {params.Q_nom} Ah ({params.Q_nom*1000} mAh)")
    print(f"    Nominal Voltage: {params.V_nom} V")
    print(f"    R0 (reference): {params.R0_ref*1000:.1f} mΩ")
    print(f"    τ1 (fast RC): {params.tau1_ref} s")
    print(f"    τ2 (slow RC): {params.tau2_ref} s")
    
    print("\n[2] Testing OCV Lookup:")
    for soc in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ocv = get_ocv(soc, 298.15)
        print(f"    SOC={soc:.2f}: OCV={ocv:.3f} V")
    
    print("\n[3] Simulating Different Scenarios:")
    scenarios = ['idle', 'light', 'moderate', 'heavy', 'gaming', 'navigation']
    
    for scenario in scenarios:
        I_func = create_scenario_current_func(scenario, load_model)
        result = model.simulate(I_func, t_span=(0, 12*3600), T_amb=298.15)
        
        tte = result['time_to_empty']
        tte_str = f"{tte:.2f}" if tte else ">12"
        avg_current = np.mean(result['current'])
        
        print(f"    {scenario:12s}: I_avg={avg_current:6.0f} mA, TTE={tte_str} hours")
    
    print("\n[4] Temperature Effects:")
    for T_c in [-10, 0, 25, 40]:
        T_k = T_c + 273.15
        I_func = lambda t: 500  # 500mA constant
        result = model.simulate(I_func, t_span=(0, 12*3600), T_amb=T_k)
        tte = result['time_to_empty']
        tte_str = f"{tte:.2f}" if tte else ">12"
        print(f"    At {T_c:+3d}°C: TTE={tte_str} hours, Final V={result['voltage'][-1]:.3f} V")
    
    print("\n[5] Aging Effects:")
    for cycles in [0, 200, 500, 1000]:
        params_aged = ECMParameters(n_cycles=cycles)
        model_aged = EquivalentCircuitModel(params_aged)
        
        C_aged = get_aged_capacity(params_aged)
        R_factor = get_aged_resistance_factor(params_aged)
        
        result = model_aged.simulate(
            lambda t: 500, 
            t_span=(0, 12*3600), 
            T_amb=298.15
        )
        tte = result['time_to_empty']
        tte_str = f"{tte:.2f}" if tte else ">12"
        
        print(f"    {cycles:4d} cycles: C={C_aged:.2f}Ah ({100*C_aged/params.Q_nom:.0f}%), "
              f"R×{R_factor:.2f}, TTE={tte_str}h")
    
    print("\n" + "="*70)
    print("ECM Model demonstration complete.")
    print("="*70)
