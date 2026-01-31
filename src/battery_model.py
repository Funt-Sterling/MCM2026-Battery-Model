"""
=============================================================================
THE PULSE OF POWER: A Kinetic Battery Model for Smartphone Drain
=============================================================================
2026 MCM Problem A - Smartphone Battery Drain Modeling

This implements:
1. Basic Model: Kinetic Battery Model (KiBaM) - Two-Tank System
2. Extended Model: KiBaM + Thermal Dynamics + Peukert Effect + Aging

Authors: [Your Team Number]
Date: January 2026
=============================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict
import warnings

# =============================================================================
# SECTION 1: PHYSICAL CONSTANTS & BATTERY PARAMETERS
# =============================================================================

@dataclass
class BatteryParameters:
    """
    Physical parameters for a typical smartphone Li-ion battery.
    
    References:
    - Manwell & McGowan (1993) - Original KiBaM formulation
    - Peukert (1897) - Capacity-rate relationship
    - Arrhenius equation for temperature dependence
    """
    # Nominal specifications (typical smartphone)
    Q_nom: float = 4000.0       # Nominal capacity [mAh]
    V_nom: float = 3.7          # Nominal voltage [V]
    V_max: float = 4.2          # Max voltage (fully charged) [V]
    V_min: float = 3.0          # Cutoff voltage [V]
    
    # KiBaM parameters (Two-Tank Model)
    c: float = 0.625            # Capacity ratio (available/total), typically 0.4-0.8
    k: float = 0.002            # Rate constant [1/s], recovery speed
    
    # Peukert parameters
    n_peukert: float = 1.05     # Peukert exponent (1.0-1.3 for Li-ion)
    I_ref: float = 400.0        # Reference current [mA] (0.1C rate)
    
    # Thermal parameters
    E_a: float = 0.35           # Activation energy [eV] (0.3-0.5 for Li-ion)
    T_ref: float = 298.15       # Reference temperature [K] (25°C)
    C_th: float = 45.0          # Thermal capacitance [J/K]
    R_th: float = 8.0           # Thermal resistance to ambient [K/W]
    
    # Aging parameters
    alpha_cycle: float = 0.0002 # Capacity fade per cycle (0.02% per cycle)
    n_cycles: int = 0           # Number of charge cycles experienced
    
    # Internal resistance model (temperature dependent)
    R_int_ref: float = 0.1      # Internal resistance at T_ref [Ohm]
    beta_R: float = 0.02        # Temperature coefficient for resistance


@dataclass
class UsageProfile:
    """
    Defines a smartphone usage scenario with component-level power draws.
    All currents in [mA].
    """
    name: str
    I_base: float = 15.0        # Baseline (idle) current [mA]
    I_screen: float = 0.0       # Screen power (brightness dependent)
    I_cpu: float = 0.0          # Processor load
    I_network: float = 0.0      # Network (WiFi/4G/5G)
    I_gps: float = 0.0          # GPS module
    I_sensors: float = 0.0      # Other sensors (accelerometer, etc.)
    
    @property
    def I_total(self) -> float:
        """Total instantaneous current draw [mA]"""
        return (self.I_base + self.I_screen + self.I_cpu + 
                self.I_network + self.I_gps + self.I_sensors)


# =============================================================================
# SECTION 2: PREDEFINED USAGE SCENARIOS
# =============================================================================

USAGE_PROFILES = {
    "idle": UsageProfile(
        name="Idle (Screen Off)",
        I_base=15, I_screen=0, I_cpu=10, I_network=5, I_gps=0, I_sensors=2
    ),
    "light": UsageProfile(
        name="Light Use (Reading, Messaging)",
        I_base=15, I_screen=150, I_cpu=100, I_network=50, I_gps=0, I_sensors=5
    ),
    "moderate": UsageProfile(
        name="Moderate Use (Social Media, Browsing)",
        I_base=15, I_screen=200, I_cpu=300, I_network=100, I_gps=0, I_sensors=10
    ),
    "heavy": UsageProfile(
        name="Heavy Use (Video Streaming)",
        I_base=15, I_screen=250, I_cpu=500, I_network=200, I_gps=0, I_sensors=10
    ),
    "gaming": UsageProfile(
        name="Gaming (Max Load)",
        I_base=15, I_screen=300, I_cpu=1200, I_network=150, I_gps=0, I_sensors=50
    ),
    "navigation": UsageProfile(
        name="Navigation (GPS Active)",
        I_base=15, I_screen=250, I_cpu=400, I_network=100, I_gps=200, I_sensors=20
    ),
}


# =============================================================================
# SECTION 3: THE BASIC MODEL - Kinetic Battery Model (KiBaM)
# =============================================================================

class KineticBatteryModel:
    """
    The Kinetic Battery Model (KiBaM) - Two-Tank System
    
    Concept: The battery is modeled as two charge reservoirs:
    - q1: Available charge (directly powers the device)
    - q2: Bound charge (chemical reservoir, slowly replenishes q1)
    
    Governing Equations:
        dq1/dt = -I(t) + k*(q2 - q1*(1-c)/c)
        dq2/dt = -k*(q2 - q1*(1-c)/c)
    
    Where:
        c = fraction of total capacity that is "available"
        k = rate constant for charge transfer between tanks
        I(t) = instantaneous current draw
    
    Reference: Manwell & McGowan, "Lead acid battery storage model for hybrid 
               energy systems", Solar Energy, 1993.
    """
    
    def __init__(self, params: BatteryParameters):
        self.params = params
        self.history = []
        
    def initial_state(self, SOC_0: float = 1.0) -> np.ndarray:
        """
        Initialize the two-tank state from SOC.
        
        Args:
            SOC_0: Initial state of charge (0 to 1)
        
        Returns:
            [q1_0, q2_0]: Initial available and bound charge [mAh]
        """
        Q_total = self.params.Q_nom * SOC_0
        q1_0 = self.params.c * Q_total
        q2_0 = (1 - self.params.c) * Q_total
        return np.array([q1_0, q2_0])
    
    def compute_SOC(self, q1: float, q2: float) -> float:
        """Compute SOC from tank charges."""
        return (q1 + q2) / self.params.Q_nom
    
    def kibam_ode(self, t: float, y: np.ndarray, I_func: Callable) -> np.ndarray:
        """
        The KiBaM differential equations.
        
        Args:
            t: Time [seconds]
            y: State vector [q1, q2] in [mAh]
            I_func: Function returning current draw at time t [mA]
        
        Returns:
            dy/dt: Time derivatives [dq1/dt, dq2/dt]
        """
        q1, q2 = y
        c = self.params.c
        k = self.params.k
        
        # Current draw (convert time to hours for mAh consistency)
        I = I_func(t) / 3600.0  # mA to mAh/s
        
        # The "height difference" term (driving force for recovery)
        h_diff = q2 - q1 * (1 - c) / c
        
        # KiBaM equations
        dq1_dt = -I + k * h_diff
        dq2_dt = -k * h_diff
        
        return np.array([dq1_dt, dq2_dt])
    
    def simulate(self, 
                 I_func: Callable[[float], float],
                 t_span: Tuple[float, float],
                 SOC_0: float = 1.0,
                 dt_eval: float = 60.0) -> Dict:
        """
        Simulate battery discharge.
        
        Args:
            I_func: Current draw function I(t) [mA]
            t_span: (t_start, t_end) in seconds
            SOC_0: Initial SOC (0 to 1)
            dt_eval: Time step for output [seconds]
        
        Returns:
            Dictionary with time series data
        """
        y0 = self.initial_state(SOC_0)
        t_eval = np.arange(t_span[0], t_span[1], dt_eval)
        
        # Event function: stop when battery is empty
        def battery_empty(t, y):
            return y[0] + y[1] - 0.01 * self.params.Q_nom  # Stop at 1% SOC
        battery_empty.terminal = True
        battery_empty.direction = -1
        
        # Solve ODE
        sol = solve_ivp(
            lambda t, y: self.kibam_ode(t, y, I_func),
            t_span,
            y0,
            method='RK45',
            t_eval=t_eval,
            events=battery_empty,
            max_step=60.0
        )
        
        # Compute derived quantities
        q1 = sol.y[0]
        q2 = sol.y[1]
        SOC = (q1 + q2) / self.params.Q_nom
        current = np.array([I_func(t) for t in sol.t])
        
        return {
            't': sol.t,
            't_hours': sol.t / 3600.0,
            'q1': q1,
            'q2': q2,
            'SOC': SOC,
            'SOC_percent': SOC * 100,
            'current': current,
            'time_to_empty': sol.t[-1] / 3600.0 if sol.t_events[0].size > 0 else None
        }


# =============================================================================
# SECTION 4: THE EXTENDED MODEL - KiBaM + Thermal + Peukert + Aging
# =============================================================================

class ExtendedBatteryModel:
    """
    Extended Battery Model incorporating:
    1. KiBaM two-tank dynamics (recovery effect)
    2. Temperature-dependent capacity (Arrhenius)
    3. Peukert effect (capacity loss at high currents)
    4. Thermal dynamics (self-heating)
    5. Battery aging (capacity fade)
    
    State Vector: [q1, q2, T]
        q1: Available charge [mAh]
        q2: Bound charge [mAh]
        T: Battery temperature [K]
    
    Governing Equations:
        dq1/dt = -I_eff(t,T) + k(T)*(q2 - q1*(1-c)/c)
        dq2/dt = -k(T)*(q2 - q1*(1-c)/c)
        dT/dt = (P_dissipated - P_cooling) / C_th
    """
    
    def __init__(self, params: BatteryParameters):
        self.params = params
        
    def effective_capacity(self, T: float) -> float:
        """
        Temperature-corrected effective capacity using Arrhenius equation.
        
        Q_eff(T) = Q_nom * exp(-E_a/k_B * (1/T - 1/T_ref))
        
        At low temperatures, capacity is significantly reduced.
        """
        k_B = 8.617e-5  # Boltzmann constant [eV/K]
        p = self.params
        
        # Arrhenius factor
        arrhenius = np.exp(-p.E_a / k_B * (1/T - 1/p.T_ref))
        
        # Aging factor
        aging_factor = 1.0 - p.alpha_cycle * p.n_cycles
        aging_factor = max(0.5, aging_factor)  # Minimum 50% capacity
        
        return p.Q_nom * arrhenius * aging_factor
    
    def temperature_dependent_k(self, T: float) -> float:
        """
        Temperature-dependent rate constant.
        Recovery is faster at higher temperatures (ion mobility increases).
        """
        k_B = 8.617e-5
        p = self.params
        return p.k * np.exp(-p.E_a / (2 * k_B) * (1/T - 1/p.T_ref))
    
    def internal_resistance(self, T: float) -> float:
        """
        Temperature-dependent internal resistance.
        R(T) = R_ref * exp(beta * (T_ref - T))
        
        Resistance increases at low temperatures, decreases at high.
        """
        p = self.params
        return p.R_int_ref * np.exp(p.beta_R * (p.T_ref - T))
    
    def peukert_effective_current(self, I: float) -> float:
        """
        Peukert's Law: Effective current accounting for rate-dependent losses.
        
        I_eff = I * (I / I_ref)^(n-1)
        
        At high currents, the "effective" drain is higher than actual current.
        """
        p = self.params
        if I <= 0:
            return 0
        return I * (I / p.I_ref) ** (p.n_peukert - 1)
    
    def extended_ode(self, t: float, y: np.ndarray, 
                     I_func: Callable, T_ambient: float) -> np.ndarray:
        """
        The complete extended model differential equations.
        
        Args:
            t: Time [seconds]
            y: State vector [q1, q2, T]
            I_func: Current function [mA]
            T_ambient: Ambient temperature [K]
        """
        q1, q2, T = y
        p = self.params
        
        # Ensure physical bounds
        q1 = max(0, q1)
        q2 = max(0, q2)
        T = max(253, min(353, T))  # -20°C to 80°C
        
        # Get current and temperature-dependent parameters
        I = I_func(t)
        k_T = self.temperature_dependent_k(T)
        R_int = self.internal_resistance(T)
        
        # Effective current (Peukert + thermal efficiency loss)
        I_eff = self.peukert_effective_current(I) / 3600.0  # Convert to mAh/s
        
        # Height difference for recovery
        c = p.c
        h_diff = q2 - q1 * (1 - c) / c if q1 > 0 else q2
        
        # KiBaM equations with temperature effects
        dq1_dt = -I_eff + k_T * h_diff
        dq2_dt = -k_T * h_diff
        
        # Thermal dynamics
        # Power dissipated = I²R (Joule heating)
        P_joule = (I / 1000.0) ** 2 * R_int  # Convert mA to A, result in Watts
        
        # Additional heat from battery inefficiency
        V = p.V_nom  # Simplified; could use SOC-dependent voltage
        P_total = (I / 1000.0) * V  # Total power draw
        efficiency = 0.95 - 0.05 * (I / p.I_ref)  # Decreases at high current
        P_heat = P_total * (1 - efficiency) + P_joule
        
        # Cooling to ambient (Newton's law of cooling)
        P_cooling = (T - T_ambient) / p.R_th
        
        dT_dt = (P_heat - P_cooling) / p.C_th
        
        return np.array([dq1_dt, dq2_dt, dT_dt])
    
    def simulate(self,
                 I_func: Callable[[float], float],
                 t_span: Tuple[float, float],
                 SOC_0: float = 1.0,
                 T_ambient: float = 298.15,
                 T_0: float = None,
                 dt_eval: float = 60.0) -> Dict:
        """
        Simulate the extended battery model.
        
        Args:
            I_func: Current draw function [mA]
            t_span: Time span (t_start, t_end) [seconds]
            SOC_0: Initial state of charge
            T_ambient: Ambient temperature [K]
            T_0: Initial battery temperature [K] (defaults to T_ambient)
            dt_eval: Output time step [seconds]
        """
        p = self.params
        
        # Initial state
        if T_0 is None:
            T_0 = T_ambient
        
        Q_eff = self.effective_capacity(T_0)
        q1_0 = p.c * Q_eff * SOC_0
        q2_0 = (1 - p.c) * Q_eff * SOC_0
        y0 = np.array([q1_0, q2_0, T_0])
        
        t_eval = np.arange(t_span[0], t_span[1], dt_eval)
        
        # Event: battery empty
        def battery_empty(t, y):
            return y[0] + y[1] - 0.01 * p.Q_nom
        battery_empty.terminal = True
        battery_empty.direction = -1
        
        # Solve
        sol = solve_ivp(
            lambda t, y: self.extended_ode(t, y, I_func, T_ambient),
            t_span,
            y0,
            method='RK45',
            t_eval=t_eval,
            events=battery_empty,
            max_step=60.0
        )
        
        # Derive outputs
        q1, q2, T = sol.y
        current = np.array([I_func(t) for t in sol.t])
        
        # SOC relative to temperature-corrected capacity
        Q_eff_array = np.array([self.effective_capacity(temp) for temp in T])
        SOC = (q1 + q2) / Q_eff_array
        SOC = np.clip(SOC, 0, 1)
        
        return {
            't': sol.t,
            't_hours': sol.t / 3600.0,
            'q1': q1,
            'q2': q2,
            'T': T,
            'T_celsius': T - 273.15,
            'SOC': SOC,
            'SOC_percent': SOC * 100,
            'current': current,
            'Q_effective': Q_eff_array,
            'time_to_empty': sol.t[-1] / 3600.0 if sol.t_events[0].size > 0 else None
        }


# =============================================================================
# SECTION 5: CURRENT PROFILE GENERATORS
# =============================================================================

def constant_current(I: float) -> Callable[[float], float]:
    """Constant current draw."""
    return lambda t: I


def periodic_usage(I_active: float, I_idle: float, 
                   period: float = 3600, duty_cycle: float = 0.3) -> Callable:
    """
    Periodic usage pattern.
    
    Args:
        I_active: Current during active use [mA]
        I_idle: Current during idle [mA]
        period: Period of usage cycle [seconds]
        duty_cycle: Fraction of time in active use
    """
    def current_func(t):
        phase = (t % period) / period
        return I_active if phase < duty_cycle else I_idle
    return current_func


def realistic_day_profile() -> Callable:
    """
    A realistic daily usage pattern based on typical smartphone use.
    
    Models a day from 7 AM to 11 PM with varying usage:
    - Morning: Light use (checking notifications)
    - Commute: Navigation
    - Work: Periodic moderate use
    - Lunch: Heavy social media
    - Afternoon: Moderate use
    - Evening: Video streaming
    - Night: Light/Idle
    """
    def current_func(t):
        hour = (t / 3600.0) % 24
        
        if 0 <= hour < 7:      # Sleep
            return 20
        elif 7 <= hour < 8:    # Morning routine
            return 250
        elif 8 <= hour < 9:    # Commute (navigation)
            return 900
        elif 9 <= hour < 12:   # Work morning
            return 150
        elif 12 <= hour < 13:  # Lunch (social media)
            return 600
        elif 13 <= hour < 17:  # Work afternoon
            return 180
        elif 17 <= hour < 18:  # Commute
            return 850
        elif 18 <= hour < 21:  # Evening (video, gaming)
            return 700
        elif 21 <= hour < 23:  # Wind down
            return 300
        else:                  # Late night
            return 50
    
    return current_func


def stochastic_usage(I_mean: float, I_std: float, 
                     seed: int = 42) -> Callable:
    """
    Stochastic (random) usage pattern with Gaussian variations.
    
    Note: For reproducibility in simulations.
    """
    rng = np.random.RandomState(seed)
    cache = {}
    
    def current_func(t):
        # Discretize time to 1-minute bins for consistency
        t_bin = int(t // 60)
        if t_bin not in cache:
            cache[t_bin] = max(10, rng.normal(I_mean, I_std))
        return cache[t_bin]
    
    return current_func


# =============================================================================
# SECTION 6: TIME-TO-EMPTY COMPUTATION
# =============================================================================

def compute_time_to_empty(model, I_func: Callable, SOC_0: float = 1.0,
                          T_ambient: float = 298.15, max_hours: float = 48) -> float:
    """
    Compute the time until battery reaches empty (< 1% SOC).
    
    Args:
        model: Battery model instance
        I_func: Current draw function
        SOC_0: Initial SOC
        T_ambient: Ambient temperature [K]
        max_hours: Maximum simulation time [hours]
    
    Returns:
        Time to empty in hours, or None if not reached
    """
    t_span = (0, max_hours * 3600)
    
    if isinstance(model, ExtendedBatteryModel):
        result = model.simulate(I_func, t_span, SOC_0, T_ambient)
    else:
        result = model.simulate(I_func, t_span, SOC_0)
    
    return result['time_to_empty']


def time_to_empty_sensitivity(model, I_values: np.ndarray, 
                               T_values: np.ndarray) -> np.ndarray:
    """
    Compute time-to-empty over a grid of current and temperature values.
    For sensitivity analysis heat maps.
    
    Returns:
        2D array of time-to-empty values [hours]
    """
    results = np.zeros((len(T_values), len(I_values)))
    
    for i, T in enumerate(T_values):
        for j, I in enumerate(I_values):
            tte = compute_time_to_empty(
                model, 
                constant_current(I), 
                T_ambient=T + 273.15
            )
            results[i, j] = tte if tte else 48.0
    
    return results


# =============================================================================
# SECTION 7: MAIN EXECUTION - DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("SMARTPHONE BATTERY MODEL - 2026 MCM Problem A")
    print("="*70)
    
    # Initialize parameters
    params = BatteryParameters()
    
    # Test Basic Model
    print("\n[1] Testing Basic KiBaM Model...")
    basic_model = KineticBatteryModel(params)
    
    # Constant 500mA draw
    result_basic = basic_model.simulate(
        constant_current(500),
        t_span=(0, 12*3600),  # 12 hours
        SOC_0=1.0
    )
    print(f"    Basic Model - Time to empty at 500mA: {result_basic['time_to_empty']:.2f} hours")
    
    # Test Extended Model at different temperatures
    print("\n[2] Testing Extended Model (Temperature Effects)...")
    extended_model = ExtendedBatteryModel(params)
    
    for T_celsius in [-10, 25, 40]:
        T_kelvin = T_celsius + 273.15
        result = extended_model.simulate(
            constant_current(500),
            t_span=(0, 12*3600),
            T_ambient=T_kelvin
        )
        tte = result['time_to_empty']
        tte_str = f"{tte:.2f} hours" if tte else ">12 hours"
        print(f"    At {T_celsius:+3d}°C: Time to empty = {tte_str}")
    
    # Test different usage scenarios
    print("\n[3] Comparing Usage Scenarios (at 25°C)...")
    for name, profile in USAGE_PROFILES.items():
        result = extended_model.simulate(
            constant_current(profile.I_total),
            t_span=(0, 24*3600),
            T_ambient=298.15
        )
        tte = result['time_to_empty']
        tte_str = f"{tte:.2f} hours" if tte else ">24 hours"
        print(f"    {profile.name:40s}: {tte_str}")
    
    print("\n[4] Demonstrating Recovery Effect...")
    # Compare continuous vs. periodic usage
    result_continuous = extended_model.simulate(
        constant_current(600),
        t_span=(0, 10*3600),
        T_ambient=298.15
    )
    
    result_periodic = extended_model.simulate(
        periodic_usage(1000, 50, period=1800, duty_cycle=0.3),
        t_span=(0, 10*3600),
        T_ambient=298.15
    )
    
    tte_cont = result_continuous['time_to_empty']
    tte_per = result_periodic['time_to_empty']
    tte_cont_str = f"{tte_cont:.2f}" if tte_cont else ">10"
    tte_per_str = f"{tte_per:.2f}" if tte_per else ">10"
    print(f"    Continuous 600mA: {tte_cont_str} hours")
    print(f"    Periodic (1000mA/50mA, 30% duty): {tte_per_str} hours")
    
    print("\n" + "="*70)
    print("Model demonstration complete. Run visualizations.py for plots.")
    print("="*70)
