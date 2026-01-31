"""
=============================================================================
RECOMMENDATIONS MODULE
=============================================================================
Translates model findings into practical recommendations.

This addresses MCM Requirement #4:
"Translate your findings into practical recommendations for a cellphone user."

Includes:
1. Power-saving strategy analysis
2. User behavior optimization
3. OS-level recommendations
4. Component-level impact analysis
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass

from battery_model import (
    BatteryParameters, ExtendedBatteryModel, USAGE_PROFILES,
    constant_current, periodic_usage, compute_time_to_empty
)


# =============================================================================
# SECTION 1: COMPONENT IMPACT ANALYSIS
# =============================================================================

@dataclass
class ComponentImpact:
    """Impact of a single component on battery life."""
    name: str
    current_range: Tuple[float, float]  # (min, max) in mA
    controllable: bool
    reduction_potential: float  # Hours gained if minimized


def analyze_component_impacts() -> Dict[str, ComponentImpact]:
    """
    Analyze the impact of each component on battery life.
    
    Returns dictionary mapping component name to its impact metrics.
    """
    params = BatteryParameters()
    model = ExtendedBatteryModel(params)
    
    # Define component current ranges (mA)
    components = {
        'Screen (Brightness)': {
            'range': (50, 400),  # 10% to 100% brightness
            'controllable': True,
            'base': 200  # 50% brightness
        },
        'CPU (Processor Load)': {
            'range': (50, 1500),  # Idle to max load
            'controllable': True,
            'base': 300  # Moderate use
        },
        'WiFi': {
            'range': (10, 100),
            'controllable': True,
            'base': 50
        },
        '4G/LTE': {
            'range': (50, 300),
            'controllable': True,
            'base': 150
        },
        '5G': {
            'range': (100, 500),
            'controllable': True,
            'base': 250
        },
        'GPS': {
            'range': (0, 200),
            'controllable': True,
            'base': 0  # Usually off
        },
        'Bluetooth': {
            'range': (5, 50),
            'controllable': True,
            'base': 20
        },
        'Background Apps': {
            'range': (20, 200),
            'controllable': True,
            'base': 80
        },
        'Sensors (Accelerometer etc.)': {
            'range': (2, 30),
            'controllable': False,
            'base': 10
        },
        'Baseline (Standby)': {
            'range': (10, 30),
            'controllable': False,
            'base': 15
        },
    }
    
    results = {}
    
    # Calculate baseline battery life
    I_baseline = sum(c['base'] for c in components.values())
    tte_baseline = compute_time_to_empty(model, constant_current(I_baseline))
    
    print("Component Impact Analysis")
    print("="*60)
    print(f"Baseline current draw: {I_baseline} mA")
    print(f"Baseline battery life: {tte_baseline:.2f} hours")
    print("-"*60)
    
    for name, comp in components.items():
        min_current, max_current = comp['range']
        
        # Calculate TTE with component at minimum
        I_min = I_baseline - comp['base'] + min_current
        tte_min = compute_time_to_empty(model, constant_current(I_min))
        
        # Calculate TTE with component at maximum
        I_max = I_baseline - comp['base'] + max_current
        tte_max = compute_time_to_empty(model, constant_current(I_max))
        
        # Reduction potential: gain from minimizing this component
        reduction_potential = (tte_min or 24) - (tte_baseline or 0)
        
        results[name] = ComponentImpact(
            name=name,
            current_range=(min_current, max_current),
            controllable=comp['controllable'],
            reduction_potential=reduction_potential
        )
        
        print(f"{name:30s}: {min_current:4.0f}-{max_current:4.0f} mA | "
              f"Potential gain: {reduction_potential:+.2f}h")
    
    return results


def plot_component_waterfall(output_file: str = 'figure_component_impact.png'):
    """
    Create a waterfall chart showing component contributions.
    """
    components = {
        'Baseline (Standby)': 15,
        'Screen (50%)': 200,
        'CPU (Moderate)': 300,
        'Network (WiFi)': 50,
        'Background Apps': 80,
        'Sensors': 10,
        'Bluetooth': 20,
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    names = list(components.keys())
    values = list(components.values())
    
    # Calculate cumulative
    cumulative = np.cumsum(values)
    starts = np.roll(cumulative, 1)
    starts[0] = 0
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(values)))
    
    bars = ax.barh(names, values, left=starts, color=colors, edgecolor='white')
    
    # Add value labels
    for bar, val, start in zip(bars, values, starts):
        ax.text(start + val/2, bar.get_y() + bar.get_height()/2,
                f'{val} mA', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=10)
    
    # Add total line
    total = sum(values)
    ax.axvline(x=total, color='red', linestyle='--', linewidth=2)
    ax.text(total + 20, len(names)/2, f'Total: {total} mA\n({total*3.7/1000:.1f}W)',
            fontsize=11, fontweight='bold', color='red')
    
    ax.set_xlabel('Current Draw (mA)', fontsize=12)
    ax.set_title('Component-Level Power Consumption Breakdown\n(Typical Moderate Use)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, total + 100)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


# =============================================================================
# SECTION 2: USER BEHAVIOR RECOMMENDATIONS
# =============================================================================

def generate_user_recommendations() -> List[Dict]:
    """
    Generate ranked list of user behavior recommendations.
    
    Each recommendation includes:
    - Action description
    - Expected battery life improvement (%)
    - Ease of implementation (1-5)
    - Trade-off description
    """
    params = BatteryParameters()
    model = ExtendedBatteryModel(params)
    
    # Baseline: typical usage
    I_baseline = 500  # mA
    tte_baseline = compute_time_to_empty(model, constant_current(I_baseline))
    
    recommendations = []
    
    # 1. Screen brightness reduction
    # From 100% (400mA) to 50% (200mA) saves ~200mA
    I_brightness = I_baseline - 200
    tte_brightness = compute_time_to_empty(model, constant_current(I_brightness))
    improvement = ((tte_brightness - tte_baseline) / tte_baseline) * 100
    recommendations.append({
        'rank': 1,
        'action': 'Reduce screen brightness from 100% to 50%',
        'improvement_pct': improvement,
        'improvement_hours': tte_brightness - tte_baseline,
        'ease': 5,
        'tradeoff': 'Reduced visibility in bright conditions',
        'category': 'Display'
    })
    
    # 2. Disable background app refresh
    I_background = I_baseline - 80
    tte_background = compute_time_to_empty(model, constant_current(I_background))
    improvement = ((tte_background - tte_baseline) / tte_baseline) * 100
    recommendations.append({
        'rank': 2,
        'action': 'Disable background app refresh',
        'improvement_pct': improvement,
        'improvement_hours': tte_background - tte_baseline,
        'ease': 4,
        'tradeoff': 'Delayed notifications, slower app startup',
        'category': 'Software'
    })
    
    # 3. Switch from 5G to 4G/LTE
    I_network = I_baseline - 150  # 5G uses ~150mA more than LTE
    tte_network = compute_time_to_empty(model, constant_current(I_network))
    improvement = ((tte_network - tte_baseline) / tte_baseline) * 100
    recommendations.append({
        'rank': 3,
        'action': 'Switch from 5G to 4G/LTE when not needed',
        'improvement_pct': improvement,
        'improvement_hours': tte_network - tte_baseline,
        'ease': 4,
        'tradeoff': 'Slower download speeds',
        'category': 'Network'
    })
    
    # 4. Use WiFi instead of cellular
    I_wifi = I_baseline - 100  # Cellular uses ~100mA more than WiFi
    tte_wifi = compute_time_to_empty(model, constant_current(I_wifi))
    improvement = ((tte_wifi - tte_baseline) / tte_baseline) * 100
    recommendations.append({
        'rank': 4,
        'action': 'Use WiFi instead of cellular data when available',
        'improvement_pct': improvement,
        'improvement_hours': tte_wifi - tte_baseline,
        'ease': 3,
        'tradeoff': 'Need WiFi network access',
        'category': 'Network'
    })
    
    # 5. Enable dark mode (OLED screens)
    I_dark = I_baseline - 100  # Dark mode can save significant power on OLED
    tte_dark = compute_time_to_empty(model, constant_current(I_dark))
    improvement = ((tte_dark - tte_baseline) / tte_baseline) * 100
    recommendations.append({
        'rank': 5,
        'action': 'Enable dark mode (OLED screens)',
        'improvement_pct': improvement,
        'improvement_hours': tte_dark - tte_baseline,
        'ease': 5,
        'tradeoff': 'Aesthetic preference',
        'category': 'Display'
    })
    
    # 6. Disable location services
    I_gps = I_baseline - 150  # GPS can draw 100-200mA
    tte_gps = compute_time_to_empty(model, constant_current(I_gps))
    improvement = ((tte_gps - tte_baseline) / tte_baseline) * 100
    recommendations.append({
        'rank': 6,
        'action': 'Disable continuous location services',
        'improvement_pct': improvement,
        'improvement_hours': tte_gps - tte_baseline,
        'ease': 3,
        'tradeoff': 'No automatic location features',
        'category': 'Hardware'
    })
    
    # 7. Shorter screen timeout
    # Reduces average screen-on time by ~20%
    I_timeout = I_baseline - 40
    tte_timeout = compute_time_to_empty(model, constant_current(I_timeout))
    improvement = ((tte_timeout - tte_baseline) / tte_baseline) * 100
    recommendations.append({
        'rank': 7,
        'action': 'Reduce screen timeout to 30 seconds',
        'improvement_pct': improvement,
        'improvement_hours': tte_timeout - tte_baseline,
        'ease': 5,
        'tradeoff': 'More frequent unlocks needed',
        'category': 'Display'
    })
    
    # 8. Temperature management (avoid heat)
    # Operating at 35°C vs 40°C
    tte_cool = compute_time_to_empty(model, constant_current(I_baseline), T_ambient=298.15)
    tte_hot = compute_time_to_empty(model, constant_current(I_baseline), T_ambient=313.15)
    improvement = ((tte_cool - tte_hot) / tte_hot) * 100
    recommendations.append({
        'rank': 8,
        'action': 'Keep phone cool (avoid direct sunlight, remove case during heavy use)',
        'improvement_pct': improvement,
        'improvement_hours': tte_cool - tte_hot,
        'ease': 2,
        'tradeoff': 'May need to remove protective case',
        'category': 'Environmental'
    })
    
    # Sort by improvement
    recommendations.sort(key=lambda x: x['improvement_pct'], reverse=True)
    
    # Re-rank
    for i, rec in enumerate(recommendations, 1):
        rec['rank'] = i
    
    return recommendations


def print_recommendations_table(recommendations: List[Dict]):
    """Print a formatted recommendations table."""
    print("\n" + "="*80)
    print("USER RECOMMENDATIONS FOR BATTERY LIFE IMPROVEMENT")
    print("="*80)
    print(f"{'Rank':<5} {'Action':<45} {'Improvement':<12} {'Ease':<6}")
    print("-"*80)
    
    for rec in recommendations:
        print(f"{rec['rank']:<5} {rec['action'][:44]:<45} "
              f"+{rec['improvement_pct']:.1f}%{'':<6} {'★'*rec['ease']:<6}")
    
    print("-"*80)
    print("\nEase Rating: ★★★★★ = Very Easy, ★ = Requires effort")


def plot_recommendations_chart(recommendations: List[Dict],
                                output_file: str = 'figure_recommendations.png'):
    """
    Create a horizontal bar chart of recommendations.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    actions = [r['action'][:35] + '...' if len(r['action']) > 35 else r['action'] 
               for r in recommendations]
    improvements = [r['improvement_pct'] for r in recommendations]
    ease = [r['ease'] for r in recommendations]
    
    colors = plt.cm.Greens(np.array(ease) / 5)
    
    bars = ax.barh(actions, improvements, color=colors, edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar, val, e in zip(bars, improvements, ease):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'+{val:.1f}% ({"★"*e})', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('Battery Life Improvement (%)', fontsize=12)
    ax.set_title('Ranked User Recommendations for Battery Life Extension',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(improvements) * 1.3)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend for ease
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=plt.cm.Greens(i/5), label=f'Ease: {"★"*i}')
        for i in [1, 3, 5]
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


# =============================================================================
# SECTION 3: OS-LEVEL POWER MANAGEMENT STRATEGIES
# =============================================================================

def analyze_power_management_strategies() -> Dict:
    """
    Analyze effectiveness of OS-level power management strategies.
    """
    params = BatteryParameters()
    model = ExtendedBatteryModel(params)
    
    strategies = {}
    
    # Baseline: Normal operation
    tte_baseline = compute_time_to_empty(
        model, 
        periodic_usage(600, 100, period=1800, duty_cycle=0.3),
        max_hours=24
    )
    
    # Strategy 1: Aggressive CPU throttling
    # Reduces peak CPU from 600mA to 400mA
    tte_throttle = compute_time_to_empty(
        model,
        periodic_usage(450, 80, period=1800, duty_cycle=0.3),
        max_hours=24
    )
    strategies['cpu_throttling'] = {
        'name': 'Aggressive CPU Throttling',
        'improvement': ((tte_throttle - tte_baseline) / tte_baseline) * 100,
        'tradeoff': 'Reduced performance, potential UI lag'
    }
    
    # Strategy 2: Adaptive screen refresh
    # Dynamic refresh rate (120Hz -> 60Hz when static)
    tte_refresh = compute_time_to_empty(
        model,
        periodic_usage(520, 80, period=1800, duty_cycle=0.3),
        max_hours=24
    )
    strategies['adaptive_refresh'] = {
        'name': 'Adaptive Screen Refresh Rate',
        'improvement': ((tte_refresh - tte_baseline) / tte_baseline) * 100,
        'tradeoff': 'Less smooth scrolling'
    }
    
    # Strategy 3: Background task batching
    # Instead of continuous small wakes, batch to periodic intervals
    tte_batch = compute_time_to_empty(
        model,
        periodic_usage(550, 40, period=3600, duty_cycle=0.25),  # Longer idle periods
        max_hours=24
    )
    strategies['task_batching'] = {
        'name': 'Background Task Batching',
        'improvement': ((tte_batch - tte_baseline) / tte_baseline) * 100,
        'tradeoff': 'Delayed notifications'
    }
    
    # Strategy 4: Predictive battery management
    # Learn usage patterns and preemptively manage power
    tte_predictive = compute_time_to_empty(
        model,
        periodic_usage(480, 60, period=2400, duty_cycle=0.28),
        max_hours=24
    )
    strategies['predictive'] = {
        'name': 'Predictive Power Management',
        'improvement': ((tte_predictive - tte_baseline) / tte_baseline) * 100,
        'tradeoff': 'Requires ML model, privacy concerns'
    }
    
    # Strategy 5: Network optimization (coalescing)
    tte_network = compute_time_to_empty(
        model,
        periodic_usage(540, 70, period=1800, duty_cycle=0.28),
        max_hours=24
    )
    strategies['network_coalescing'] = {
        'name': 'Network Request Coalescing',
        'improvement': ((tte_network - tte_baseline) / tte_baseline) * 100,
        'tradeoff': 'Slightly delayed sync'
    }
    
    print("\nOS-Level Power Management Strategies")
    print("="*60)
    for key, strat in strategies.items():
        print(f"{strat['name']:35s}: +{strat['improvement']:.1f}%")
        print(f"   Trade-off: {strat['tradeoff']}")
    
    return strategies


# =============================================================================
# SECTION 4: BATTERY AGING RECOMMENDATIONS
# =============================================================================

def analyze_charging_behaviors() -> Dict:
    """
    Analyze how charging behaviors affect battery longevity.
    """
    print("\n" + "="*60)
    print("BATTERY LONGEVITY RECOMMENDATIONS")
    print("="*60)
    
    recommendations = {
        'avoid_extreme_soc': {
            'title': 'Avoid Extreme SOC Levels',
            'description': 'Keep battery between 20% and 80% when possible',
            'impact': 'Can extend battery lifespan by 2-3x',
            'mechanism': 'Reduces stress on cathode/anode at extreme voltages'
        },
        'avoid_fast_charging': {
            'title': 'Limit Fast Charging Usage',
            'description': 'Use slow charging overnight when possible',
            'impact': 'Reduces heat-induced degradation by ~20%',
            'mechanism': 'Lower charging current = less heat = less SEI growth'
        },
        'temperature_management': {
            'title': 'Maintain Optimal Temperature',
            'description': 'Avoid charging in hot conditions (>35°C)',
            'impact': 'Prevents accelerated capacity fade',
            'mechanism': 'Arrhenius: degradation rate doubles per 10°C increase'
        },
        'partial_cycles': {
            'title': 'Prefer Partial Cycles',
            'description': 'Multiple shallow cycles are better than deep cycles',
            'impact': '~40% more total throughput over battery life',
            'mechanism': 'Reduces mechanical stress from full expansion/contraction'
        },
    }
    
    for key, rec in recommendations.items():
        print(f"\n{rec['title']}")
        print(f"  → {rec['description']}")
        print(f"  Impact: {rec['impact']}")
        print(f"  Mechanism: {rec['mechanism']}")
    
    return recommendations


# =============================================================================
# SECTION 5: MAIN EXECUTION
# =============================================================================

def generate_all_recommendations():
    """Generate complete recommendations report."""
    print("="*70)
    print("BATTERY LIFE RECOMMENDATIONS ANALYSIS")
    print("="*70)
    
    # Component analysis
    print("\n[1] Component Impact Analysis")
    print("-"*70)
    analyze_component_impacts()
    plot_component_waterfall()
    
    # User recommendations
    print("\n[2] User Behavior Recommendations")
    print("-"*70)
    recommendations = generate_user_recommendations()
    print_recommendations_table(recommendations)
    plot_recommendations_chart(recommendations)
    
    # OS strategies
    print("\n[3] OS-Level Strategies")
    print("-"*70)
    analyze_power_management_strategies()
    
    # Aging recommendations
    print("\n[4] Battery Longevity")
    print("-"*70)
    analyze_charging_behaviors()
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS ANALYSIS COMPLETE")
    print("="*70)
    
    return recommendations


if __name__ == "__main__":
    generate_all_recommendations()
