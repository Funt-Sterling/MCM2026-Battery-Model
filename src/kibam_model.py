"""
=============================================================================
KINETIC BATTERY MODEL (KiBaM)
=============================================================================
Model I from the paper - The Conceptual Foundation

The KiBaM represents the battery as two connected "tanks":
- Available tank (q1): Energy immediately available for discharge
- Bound tank (q2): Energy that slowly becomes available (recovery)

This explains the "recovery effect" - why batteries seem to regain capacity
after resting. The bound charge slowly flows into the available tank.

Reference:
Manwell, J.F. & McGowan, J.G. (1993). "Lead acid battery storage model 
for hybrid energy systems." Solar Energy, 50(5), 399-405.

Authors: MCM 2026 Team
=============================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Tuple, Dict, List, Callable, Optional
import matplotlib.pyplot as plt


@dataclass
class KiBaMParameters:
    """
    Parameters for the Kinetic Battery Model.
    
    The model is defined by two key parameters:
    - c: Capacity ratio (fraction in available tank at full charge)
    - k: Rate constant (how fast bound charge becomes available)
    
    Typical values for lithium-ion:
    - c ≈ 0.4-0.8 (higher c = more immediately available)
    - k ≈ 0.001-0.01 s^-1 (higher k = faster recovery)
    """
    Q_max: float = 2.0    # Maximum capacity (Ah)
    c: float = 0.6        # Capacity ratio (available/total)
    k: float = 0.005      # Rate constant (1/s)
    
    # Peukert effect (optional extension)
    peukert_n: float = 1.05  # Peukert exponent (1.0-1.3 for Li-ion)
    I_ref: float = 0.5       # Reference current for Peukert (A)


class KineticBatteryModel:
    """
    The Kinetic Battery Model (KiBaM) - A Two-Tank Analogy.
    
    Physical Interpretation:
    ========================
    Imagine two water tanks connected by a pipe with a valve:
    
        ┌─────────┐     valve     ┌─────────┐
        │  q1     │───────────────│  q2     │
        │(avail)  │      k        │ (bound) │
        │         │               │         │
        └────┬────┘               └─────────┘
             │
             ▼ I(t)
         (discharge)
    
    - Tank 1 (q1): Available charge - can be discharged immediately
    - Tank 2 (q2): Bound charge - slowly flows into Tank 1
    - The "valve" k controls how fast recovery happens
    - Only Tank 1 can supply current I(t)
    
    Governing Equations:
    ====================
    dq1/dt = -I(t) + k * (q2/c2 - q1/c1)
    dq2/dt = -k * (q2/c2 - q1/c1)
    
    where c1 = c, c2 = 1 - c
    
    This explains:
    - Why batteries "recover" after rest (q2 flows into q1)
    - Why high currents drain batteries faster than expected (q2 can't keep up)
    - The "rate capacity effect" - capacity depends on discharge rate
    """
    
    def __init__(self, params: KiBaMParameters = None):
        """
        Initialize KiBaM model.
        
        Args:
            params: Model parameters (uses defaults if None)
        """
        self.p = params or KiBaMParameters()
        
        # Derived parameters
        self.c1 = self.p.c           # Fraction in available tank
        self.c2 = 1 - self.p.c       # Fraction in bound tank
        
        # Validate parameters
        assert 0 < self.c1 < 1, "Capacity ratio c must be between 0 and 1"
        assert self.p.k > 0, "Rate constant k must be positive"
        
        # History tracking
        self.history: Dict[str, List] = {
            'time': [], 'q1': [], 'q2': [], 'q_total': [],
            'soc': [], 'current': [], 'voltage_est': []
        }
    
    def initial_state(self, soc: float = 1.0) -> np.ndarray:
        """
        Get initial state for given SOC.
        
        At equilibrium, the charge is distributed according to c:
        q1 = c * Q_total
        q2 = (1-c) * Q_total
        
        Args:
            soc: Initial state of charge (0-1)
            
        Returns:
            Initial state [q1, q2] in Ah
        """
        Q_total = soc * self.p.Q_max
        q1 = self.c1 * Q_total
        q2 = self.c2 * Q_total
        return np.array([q1, q2])
    
    def dynamics(self, t: float, y: np.ndarray, I_func: Callable[[float], float]) -> np.ndarray:
        """
        KiBaM differential equations.
        
        dq1/dt = -I(t) + k * (q2/c2 - q1/c1)
        dq2/dt = -k * (q2/c2 - q1/c1)
        
        The term (q2/c2 - q1/c1) represents the "pressure difference"
        between the tanks, driving flow from bound to available.
        
        Args:
            t: Time (s)
            y: State [q1, q2] in Ah
            I_func: Function that returns current I(t) in Amps
            
        Returns:
            State derivatives [dq1/dt, dq2/dt] in Ah/s
        """
        q1, q2 = y
        I = I_func(t)
        
        # Ensure non-negative
        q1 = max(q1, 0)
        q2 = max(q2, 0)
        
        # Flow rate between tanks (the "valve")
        # Positive flow_rate means q2 → q1 (recovery)
        if self.c1 > 0 and self.c2 > 0:
            height_diff = q2 / self.c2 - q1 / self.c1
            flow_rate = self.p.k * height_diff
        else:
            flow_rate = 0
        
        # Dynamics
        # q1 loses charge to discharge (I) but gains from q2
        dq1_dt = -I / 3600 + flow_rate  # I in A, convert to Ah/s
        
        # q2 loses charge to q1
        dq2_dt = -flow_rate
        
        return np.array([dq1_dt, dq2_dt])
    
    def simulate(
        self,
        duration_hours: float,
        I_func: Callable[[float], float],
        initial_soc: float = 1.0,
        dt_minutes: float = 1.0
    ) -> Dict:
        """
        Simulate battery discharge/charge.
        
        Args:
            duration_hours: Simulation duration (hours)
            I_func: Current function I(t) in Amps (positive = discharge)
            initial_soc: Initial state of charge (0-1)
            dt_minutes: Output resolution (minutes)
            
        Returns:
            Dictionary with simulation results
        """
        duration_s = duration_hours * 3600
        
        # Initial state
        y0 = self.initial_state(initial_soc)
        
        # Time points
        n_points = int(duration_hours * 60 / dt_minutes)
        t_eval = np.linspace(0, duration_s, n_points)
        
        # Solve ODE
        def wrapped_dynamics(t, y):
            return self.dynamics(t, y, I_func)
        
        sol = solve_ivp(
            wrapped_dynamics,
            (0, duration_s),
            y0,
            method='RK45',
            t_eval=t_eval,
            max_step=60.0
        )
        
        # Extract results
        times_hours = sol.t / 3600
        q1 = sol.y[0]
        q2 = sol.y[1]
        q_total = q1 + q2
        soc = q_total / self.p.Q_max
        
        # Current at each time point
        current = np.array([I_func(t) for t in sol.t])
        
        # Estimate voltage (simple linear model)
        # This is approximate - KiBaM doesn't model voltage directly
        V_oc = 3.0 + 1.2 * soc  # Simple linear OCV
        voltage_est = V_oc - 0.1 * current  # Subtract IR drop
        
        # Find when available charge depleted
        battery_life_hours = None
        for i, q in enumerate(q1):
            if q <= 0.01:  # Available charge depleted
                battery_life_hours = times_hours[i]
                break
        
        if battery_life_hours is None:
            battery_life_hours = times_hours[-1]
        
        return {
            'time_hours': times_hours,
            'q_available': q1,
            'q_bound': q2,
            'q_total': q_total,
            'soc': soc,
            'current': current,
            'voltage_estimate': voltage_est,
            'battery_life_hours': battery_life_hours,
            'final_soc': soc[-1],
            'recovery_potential': q2[-1],  # Bound charge still available
        }
    
    def demonstrate_recovery_effect(self, I_discharge: float = 1.0, 
                                    discharge_time: float = 0.5,
                                    rest_time: float = 0.5) -> Dict:
        """
        Demonstrate the recovery effect - battery capacity increases after rest.
        
        This is the KEY insight from KiBaM: If you discharge at high current
        and then rest, the available charge increases as bound charge flows in.
        
        Args:
            I_discharge: Discharge current (A)
            discharge_time: Time to discharge (hours)
            rest_time: Time to rest (hours)
            
        Returns:
            Results showing recovery
        """
        def current_profile(t):
            # Discharge for discharge_time, then rest
            t_hours = t / 3600
            if t_hours < discharge_time:
                return I_discharge
            else:
                return 0.0
        
        total_time = discharge_time + rest_time
        result = self.simulate(total_time, current_profile, initial_soc=1.0)
        
        # Find state at end of discharge and end of rest
        discharge_idx = int(len(result['time_hours']) * discharge_time / total_time)
        
        q1_after_discharge = result['q_available'][discharge_idx]
        q1_after_rest = result['q_available'][-1]
        
        recovery_amount = q1_after_rest - q1_after_discharge
        recovery_percent = 100 * recovery_amount / q1_after_discharge if q1_after_discharge > 0 else 0
        
        result['q1_after_discharge'] = q1_after_discharge
        result['q1_after_rest'] = q1_after_rest
        result['recovery_amount_Ah'] = recovery_amount
        result['recovery_percent'] = recovery_percent
        
        return result
    
    def rate_capacity_effect(self, currents: List[float] = None) -> Dict:
        """
        Demonstrate the rate capacity effect.
        
        Higher discharge currents result in lower effective capacity
        because the bound charge can't flow fast enough.
        
        Args:
            currents: List of discharge currents to test (A)
            
        Returns:
            Dictionary with results for each current
        """
        if currents is None:
            currents = [0.2, 0.5, 1.0, 2.0, 3.0]
        
        results = {}
        
        for I in currents:
            # Constant current discharge until q1 depletes
            def I_func(t):
                return I
            
            result = self.simulate(
                duration_hours=self.p.Q_max / I * 2,  # 2x theoretical time
                I_func=I_func,
                initial_soc=1.0
            )
            
            # Effective capacity = I * time_to_depletion
            effective_capacity = I * result['battery_life_hours']
            capacity_ratio = effective_capacity / self.p.Q_max
            
            results[f"{I}A"] = {
                'current': I,
                'battery_life_hours': result['battery_life_hours'],
                'effective_capacity_Ah': effective_capacity,
                'capacity_ratio': capacity_ratio,
                'final_bound_charge': result['q_bound'][-1],
            }
        
        return results


def plot_kibam_concepts():
    """Generate visualization of KiBaM concepts for the paper."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Model instance
    kibam = KineticBatteryModel()
    
    # 1. Basic discharge showing two tanks
    ax1 = axes[0, 0]
    result = kibam.simulate(
        duration_hours=3,
        I_func=lambda t: 0.5,  # 0.5A constant
        initial_soc=1.0
    )
    
    ax1.plot(result['time_hours'], result['q_available'], 'b-', linewidth=2, label='Available (q₁)')
    ax1.plot(result['time_hours'], result['q_bound'], 'r--', linewidth=2, label='Bound (q₂)')
    ax1.plot(result['time_hours'], result['q_total'], 'k-', linewidth=2, label='Total')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Charge (Ah)')
    ax1.set_title('Two-Tank Discharge Behavior')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Recovery effect demonstration
    ax2 = axes[0, 1]
    recovery = kibam.demonstrate_recovery_effect(
        I_discharge=1.0,
        discharge_time=0.3,
        rest_time=0.7
    )
    
    ax2.plot(recovery['time_hours'], recovery['q_available'], 'b-', linewidth=2, label='Available (q₁)')
    ax2.axvline(x=0.3, color='gray', linestyle=':', label='Rest begins')
    ax2.axhline(y=recovery['q1_after_discharge'], color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=recovery['q1_after_rest'], color='g', linestyle='--', alpha=0.5)
    
    ax2.annotate(f"Recovery: {recovery['recovery_percent']:.1f}%",
                xy=(0.7, recovery['q1_after_rest']),
                xytext=(0.75, recovery['q1_after_rest'] + 0.1),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='green'))
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Available Charge (Ah)')
    ax2.set_title('Recovery Effect: Charge Returns After Rest')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Rate capacity effect
    ax3 = axes[1, 0]
    rate_effect = kibam.rate_capacity_effect([0.2, 0.5, 1.0, 1.5, 2.0])
    
    currents = [v['current'] for v in rate_effect.values()]
    capacities = [v['capacity_ratio'] * 100 for v in rate_effect.values()]
    
    ax3.bar(range(len(currents)), capacities, color='steelblue')
    ax3.set_xticks(range(len(currents)))
    ax3.set_xticklabels([f'{c}A' for c in currents])
    ax3.set_xlabel('Discharge Current')
    ax3.set_ylabel('Effective Capacity (%)')
    ax3.set_title('Rate Capacity Effect: Higher Current = Lower Capacity')
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, cap in enumerate(capacities):
        ax3.text(i, cap + 2, f'{cap:.0f}%', ha='center', fontsize=9)
    
    # 4. SOC comparison at different rates
    ax4 = axes[1, 1]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 4))
    for i, I in enumerate([0.3, 0.6, 1.0, 2.0]):
        result = kibam.simulate(
            duration_hours=4,
            I_func=lambda t, I=I: I,
            initial_soc=1.0
        )
        ax4.plot(result['time_hours'], result['soc'] * 100, 
                color=colors[i], linewidth=2, label=f'{I} A')
    
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('State of Charge (%)')
    ax4.set_title('SOC Depletion at Different Discharge Rates')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('../figures/kibam_concepts.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# ============================================================================
# VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("KINETIC BATTERY MODEL (KiBaM) - Model I")
    print("=" * 60)
    
    # Create model
    kibam = KineticBatteryModel()
    
    # Test 1: Basic discharge
    print("\n[Test 1] Basic Discharge at 0.5A:")
    result = kibam.simulate(
        duration_hours=3,
        I_func=lambda t: 0.5,
        initial_soc=1.0
    )
    print(f"  Battery life: {result['battery_life_hours']:.2f} hours")
    print(f"  Final SOC: {result['final_soc']*100:.1f}%")
    print(f"  Recovery potential (bound): {result['recovery_potential']:.3f} Ah")
    
    # Test 2: Recovery effect
    print("\n[Test 2] Recovery Effect (1A discharge for 30min, then rest):")
    recovery = kibam.demonstrate_recovery_effect(
        I_discharge=1.0,
        discharge_time=0.5,
        rest_time=0.5
    )
    print(f"  q1 after discharge: {recovery['q1_after_discharge']:.3f} Ah")
    print(f"  q1 after rest: {recovery['q1_after_rest']:.3f} Ah")
    print(f"  Recovery: {recovery['recovery_amount_Ah']*1000:.1f} mAh ({recovery['recovery_percent']:.1f}%)")
    
    # Test 3: Rate capacity effect
    print("\n[Test 3] Rate Capacity Effect:")
    rate_effect = kibam.rate_capacity_effect([0.2, 0.5, 1.0, 2.0])
    for name, data in rate_effect.items():
        print(f"  {name}: {data['capacity_ratio']*100:.1f}% effective capacity, "
              f"{data['battery_life_hours']:.2f}h runtime")
    
    print("\n" + "=" * 60)
    print("MODEL VALIDATION COMPLETE")
    print("=" * 60)
    
    # Generate figure
    try:
        plot_kibam_concepts()
        print("\nFigure saved to ../figures/kibam_concepts.png")
    except Exception as e:
        print(f"\nCould not generate figure: {e}")
