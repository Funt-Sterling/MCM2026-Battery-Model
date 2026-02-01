"""
=============================================================================
FAST CHARGING DEGRADATION MODEL
=============================================================================
Based on: Bruj & Calborean, "Lifecycle Evaluation of Lithium-Ion Batteries 
Under Fast Charging and Discharging Conditions" (Batteries, 2025)

Key findings from the paper:
- 1C charge rate: Baseline (~200 cycles to 80% SoH)
- 1.5C charge rate: ~50% lifetime reduction (~95 cycles)
- 2C charge rate: ~70% lifetime reduction (~25 cycles, reaches 80% SoH only)

This module implements charging stress factors (σ) that modify the 
aging rate based on charge C-rate.

σ(C) = 1.0 for C ≤ 1
σ(C) = 1 + 2.3*(C-1) for C > 1 (linear approximation)

Reference:
Bruj, O. & Calborean, A. (2025). Lifecycle Evaluation of Lithium-Ion Batteries 
Under Fast Charging and Discharging Conditions. Batteries, 11(2), 65.
DOI: 10.3390/batteries11020065
=============================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt


@dataclass
class FastChargingParams:
    """
    Parameters from LifecyclE paper (Bruj & Calborean, 2025).
    
    Based on Panasonic NCR18650B cells (3200 mAh, 3.6V nominal).
    """
    # Baseline cycles to 80% SoH at 1C charge rate
    baseline_cycles_1C: float = 200.0
    
    # Measured cycles to threshold at different rates
    cycles_1C: float = 200.0      # ~200 cycles at 1C
    cycles_1_5C: float = 95.0     # ~95 cycles at 1.5C (~50% reduction)
    cycles_2C: float = 25.0       # ~25 cycles at 2C (~70% reduction, reaches 80% only)
    
    # Derived stress factors (σ = baseline_cycles / actual_cycles)
    # σ(1C) = 1.0, σ(1.5C) ≈ 2.1, σ(2C) ≈ 8.0
    # Linear fit: σ(C) = 1 + 2.3*(C-1) for C > 1
    
    # SoH threshold where battery is considered "end of life"
    soh_threshold: float = 0.80


def charging_stress_factor(c_rate: float, model: str = 'linear') -> float:
    """
    Compute charging stress factor σ based on C-rate.
    
    From Bruj & Calborean (2025):
    - 1C → ~200 cycles to 80% SoH (baseline)
    - 1.5C → ~95 cycles (2.1× degradation)
    - 2C → ~25 cycles (8× degradation, limited to 80% SoH)
    
    Args:
        c_rate: Charging C-rate (1.0 = 1C, 2.0 = 2C, etc.)
        model: 'linear', 'quadratic', or 'exponential'
        
    Returns:
        Stress factor σ (≥ 1.0)
    """
    if c_rate <= 1.0:
        return 1.0
    
    if model == 'linear':
        # Linear fit to data points: σ = 1 + 2.3*(C-1)
        # σ(1) = 1.0, σ(1.5) = 2.15, σ(2) = 3.3
        # Note: This underestimates at 2C (actual is ~8×)
        return 1.0 + 2.3 * (c_rate - 1.0)
    
    elif model == 'quadratic':
        # Quadratic fit: σ = 1 + (C-1)² * 7
        # σ(1) = 1.0, σ(1.5) = 2.75, σ(2) = 8.0
        return 1.0 + 7.0 * (c_rate - 1.0) ** 2
    
    elif model == 'exponential':
        # Exponential fit for worst case: σ = exp(2*(C-1))
        # σ(1) = 1.0, σ(1.5) = 2.7, σ(2) = 7.4
        return np.exp(2.0 * (c_rate - 1.0))
    
    else:
        raise ValueError(f"Unknown model: {model}")


def cycle_life_at_rate(c_rate: float, baseline_cycles: float = 200.0, 
                       model: str = 'quadratic') -> float:
    """
    Estimate cycle life at a given charging rate.
    
    Args:
        c_rate: Charging C-rate
        baseline_cycles: Cycles to 80% SoH at 1C (default 200)
        model: Stress factor model
        
    Returns:
        Estimated cycles to 80% SoH
    """
    sigma = charging_stress_factor(c_rate, model)
    return baseline_cycles / sigma


def soh_after_cycles(n_cycles: float, c_rate: float = 1.0,
                     model: str = 'quadratic') -> float:
    """
    Estimate SoH after n cycles at given charge rate.
    
    Uses √N degradation law modified by stress factor:
        SoH = 1 - α * σ(C) * √N
        
    Where α is calibrated so SoH = 0.8 at baseline_cycles for 1C.
    
    Args:
        n_cycles: Number of charge/discharge cycles
        c_rate: Charging C-rate
        model: Stress factor model
        
    Returns:
        State of Health (0-1)
    """
    # Calibrate α so that SoH = 0.8 after 200 cycles at 1C
    # 0.8 = 1 - α * 1.0 * √200
    # α = 0.2 / √200 ≈ 0.0141
    alpha = 0.0141
    
    sigma = charging_stress_factor(c_rate, model)
    soh = 1.0 - alpha * sigma * np.sqrt(n_cycles)
    
    return max(soh, 0.0)


def cycles_to_soh(target_soh: float, c_rate: float = 1.0,
                  model: str = 'quadratic') -> float:
    """
    Calculate cycles needed to reach target SoH.
    
    Args:
        target_soh: Target state of health (e.g., 0.8 for 80%)
        c_rate: Charging C-rate
        model: Stress factor model
        
    Returns:
        Number of cycles
    """
    alpha = 0.0141
    sigma = charging_stress_factor(c_rate, model)
    
    # SoH = 1 - α * σ * √N
    # 1 - SoH = α * σ * √N
    # N = ((1 - SoH) / (α * σ))²
    
    degradation = 1.0 - target_soh
    n_cycles = (degradation / (alpha * sigma)) ** 2
    
    return n_cycles


class FastChargingAging:
    """
    Fast charging-aware aging model for battery simulations.
    
    Tracks cumulative damage from charging at various rates and
    predicts remaining cycle life.
    """
    
    def __init__(self, params: FastChargingParams = None):
        self.p = params or FastChargingParams()
        
        # Cumulative damage (in "equivalent 1C cycles")
        self.cumulative_damage = 0.0
        
        # Charging history
        self.charge_history: list = []
    
    def record_charge(self, c_rate: float, depth: float = 1.0):
        """
        Record a charging event.
        
        Args:
            c_rate: Charging C-rate used
            depth: Depth of charge (0-1), 1.0 = full charge
        """
        # Calculate equivalent damage at 1C
        sigma = charging_stress_factor(c_rate, 'quadratic')
        damage = sigma * depth
        
        self.cumulative_damage += damage
        self.charge_history.append({
            'c_rate': c_rate,
            'depth': depth,
            'damage': damage,
            'cumulative': self.cumulative_damage,
        })
    
    def get_soh(self) -> float:
        """Get current SoH based on accumulated damage."""
        # Equivalent cycles at 1C
        equiv_cycles = self.cumulative_damage
        return soh_after_cycles(equiv_cycles, c_rate=1.0)
    
    def get_remaining_cycles(self, c_rate: float = 1.0, 
                             target_soh: float = 0.8) -> float:
        """
        Estimate remaining cycles at given rate to reach target SoH.
        
        Args:
            c_rate: Future charging rate
            target_soh: Target SoH threshold
            
        Returns:
            Remaining cycles
        """
        current_soh = self.get_soh()
        if current_soh <= target_soh:
            return 0.0
        
        # Current equivalent 1C cycles
        current_equiv = self.cumulative_damage
        
        # Total 1C-equivalent cycles needed to reach target
        total_equiv_needed = cycles_to_soh(target_soh, c_rate=1.0)
        
        # Remaining 1C-equivalent cycles
        remaining_equiv = total_equiv_needed - current_equiv
        
        # Convert to cycles at specified rate
        sigma = charging_stress_factor(c_rate, 'quadratic')
        return remaining_equiv / sigma
    
    def get_summary(self) -> Dict:
        """Get summary of charging history and current state."""
        if not self.charge_history:
            return {
                'total_charges': 0,
                'avg_c_rate': 0,
                'cumulative_damage': 0,
                'current_soh': 1.0,
            }
        
        avg_rate = np.mean([h['c_rate'] for h in self.charge_history])
        
        return {
            'total_charges': len(self.charge_history),
            'avg_c_rate': avg_rate,
            'cumulative_damage': self.cumulative_damage,
            'current_soh': self.get_soh(),
            'remaining_cycles_1C': self.get_remaining_cycles(1.0),
            'remaining_cycles_2C': self.get_remaining_cycles(2.0),
        }


def validate_against_paper():
    """
    Validate model against Bruj & Calborean (2025) results.
    """
    print("=" * 60)
    print("FAST CHARGING MODEL VALIDATION")
    print("Based on Bruj & Calborean (2025)")
    print("=" * 60)
    
    # Paper results
    paper_data = {
        1.0: {'cycles': 200, 'reduction': '0%'},
        1.5: {'cycles': 95, 'reduction': '~50%'},
        2.0: {'cycles': 25, 'reduction': '~70%'},
    }
    
    print("\nComparison with paper results:")
    print("-" * 50)
    print(f"{'C-rate':<10} {'Paper':>12} {'Model (quad)':>15} {'Error':>10}")
    print("-" * 50)
    
    for c_rate, data in paper_data.items():
        paper_cycles = data['cycles']
        model_cycles = cycle_life_at_rate(c_rate, model='quadratic')
        error = 100 * abs(model_cycles - paper_cycles) / paper_cycles
        print(f"{c_rate}C{'':<7} {paper_cycles:>10} cycles  {model_cycles:>10.0f} cycles  {error:>8.1f}%")
    
    print("\nStress factors σ(C):")
    print("-" * 50)
    for c_rate in [1.0, 1.5, 2.0, 2.5, 3.0]:
        sigma_lin = charging_stress_factor(c_rate, 'linear')
        sigma_quad = charging_stress_factor(c_rate, 'quadratic')
        sigma_exp = charging_stress_factor(c_rate, 'exponential')
        print(f"{c_rate}C: σ_linear={sigma_lin:.2f}, σ_quad={sigma_quad:.2f}, σ_exp={sigma_exp:.2f}")
    
    # Test aging model
    print("\n" + "=" * 60)
    print("AGING MODEL SIMULATION")
    print("=" * 60)
    
    aging = FastChargingAging()
    
    # Simulate different charging patterns
    scenarios = {
        'Conservative (1C)': [(1.0, 1.0)] * 100,
        'Mixed (1C/2C)': [(1.0, 1.0)] * 50 + [(2.0, 1.0)] * 50,
        'Aggressive (2C)': [(2.0, 1.0)] * 100,
    }
    
    for name, charges in scenarios.items():
        aging = FastChargingAging()
        for c_rate, depth in charges:
            aging.record_charge(c_rate, depth)
        
        summary = aging.get_summary()
        print(f"\n{name}:")
        print(f"  Total charges: {summary['total_charges']}")
        print(f"  SoH after 100 charges: {summary['current_soh']*100:.1f}%")
        print(f"  Remaining cycles to 80% at 1C: {summary['remaining_cycles_1C']:.0f}")


def plot_degradation_comparison():
    """Generate comparison figure for paper."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Cycle life vs C-rate
    ax1 = axes[0]
    c_rates = np.linspace(0.5, 3.0, 50)
    
    for model, label, color in [('linear', 'Linear', 'blue'),
                                 ('quadratic', 'Quadratic', 'red'),
                                 ('exponential', 'Exponential', 'green')]:
        cycles = [cycle_life_at_rate(c, model=model) for c in c_rates]
        ax1.plot(c_rates, cycles, color=color, label=label, linewidth=2)
    
    # Paper data points
    ax1.scatter([1.0, 1.5, 2.0], [200, 95, 25], color='black', s=100, 
                zorder=5, label='Bruj & Calborean (2025)')
    
    ax1.set_xlabel('Charging C-rate')
    ax1.set_ylabel('Cycles to 80% SoH')
    ax1.set_title('Cycle Life vs. Charging Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, 3.0)
    ax1.set_ylim(0, 300)
    
    # Right: SoH evolution over cycles
    ax2 = axes[1]
    cycles = np.linspace(0, 300, 100)
    
    for c_rate, color in [(1.0, 'green'), (1.5, 'orange'), (2.0, 'red')]:
        soh = [soh_after_cycles(n, c_rate=c_rate) * 100 for n in cycles]
        ax2.plot(cycles, soh, color=color, linewidth=2, label=f'{c_rate}C')
    
    ax2.axhline(y=80, color='gray', linestyle='--', label='80% SoH threshold')
    
    ax2.set_xlabel('Number of Cycles')
    ax2.set_ylabel('State of Health (%)')
    ax2.set_title('SoH Degradation at Different Charge Rates')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 300)
    ax2.set_ylim(50, 105)
    
    plt.tight_layout()
    plt.savefig('../figures/fast_charging_degradation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


if __name__ == "__main__":
    validate_against_paper()
    
    try:
        plot_degradation_comparison()
        print("\nFigure saved to ../figures/fast_charging_degradation.png")
    except Exception as e:
        print(f"\nCould not generate figure: {e}")
