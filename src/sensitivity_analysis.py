"""
=============================================================================
SENSITIVITY ANALYSIS MODULE
=============================================================================
Comprehensive sensitivity analysis for the battery model.
This module generates the data and visualizations for Section 6 of the paper.

Includes:
1. One-at-a-time (OAT) sensitivity analysis
2. Sobol sensitivity indices (global sensitivity)
3. Parameter importance ranking
4. Uncertainty quantification
=============================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings

from battery_model import (
    BatteryParameters, ExtendedBatteryModel,
    constant_current, compute_time_to_empty
)

# =============================================================================
# SECTION 1: PARAMETER RANGES FOR SENSITIVITY ANALYSIS
# =============================================================================

PARAM_RANGES = {
    # Battery parameters
    'Q_nom': (3000, 5000, 'mAh', 'Nominal Capacity'),
    'c': (0.4, 0.8, '-', 'KiBaM Capacity Ratio'),
    'k': (0.001, 0.005, '1/s', 'KiBaM Rate Constant'),
    
    # Peukert effect
    'n_peukert': (1.02, 1.15, '-', 'Peukert Exponent'),
    
    # Thermal parameters
    'E_a': (0.2, 0.5, 'eV', 'Activation Energy'),
    'C_th': (30, 60, 'J/K', 'Thermal Capacitance'),
    'R_th': (5, 15, 'K/W', 'Thermal Resistance'),
    'R_int_ref': (0.05, 0.20, 'Ω', 'Internal Resistance'),
    
    # External conditions
    'T_ambient': (263, 313, 'K', 'Ambient Temperature'),  # -10°C to 40°C
    'I_load': (100, 1500, 'mA', 'Load Current'),
}


@dataclass
class SensitivityResult:
    """Container for sensitivity analysis results."""
    parameter: str
    values: np.ndarray
    outputs: np.ndarray  # Time-to-empty for each parameter value
    baseline: float
    sensitivity_index: float
    elasticity: float


# =============================================================================
# SECTION 2: ONE-AT-A-TIME SENSITIVITY ANALYSIS
# =============================================================================

def oat_sensitivity(param_name: str, 
                   n_samples: int = 20,
                   base_params: BatteryParameters = None,
                   I_load: float = 500,
                   T_ambient: float = 298.15) -> SensitivityResult:
    """
    One-at-a-Time (OAT) sensitivity analysis.
    
    Varies a single parameter while holding others at baseline.
    
    Args:
        param_name: Name of parameter to vary
        n_samples: Number of samples in the range
        base_params: Baseline parameter set
        I_load: Load current [mA]
        T_ambient: Ambient temperature [K]
    
    Returns:
        SensitivityResult object
    """
    if base_params is None:
        base_params = BatteryParameters()
    
    # Get parameter range
    if param_name not in PARAM_RANGES:
        raise ValueError(f"Unknown parameter: {param_name}")
    
    p_min, p_max, unit, desc = PARAM_RANGES[param_name]
    param_values = np.linspace(p_min, p_max, n_samples)
    
    outputs = []
    
    for val in param_values:
        # Create modified parameters
        params_dict = {
            'Q_nom': base_params.Q_nom,
            'c': base_params.c,
            'k': base_params.k,
            'n_peukert': base_params.n_peukert,
            'E_a': base_params.E_a,
            'C_th': base_params.C_th,
            'R_th': base_params.R_th,
            'R_int_ref': base_params.R_int_ref,
        }
        
        # Update the varied parameter
        if param_name in params_dict:
            params_dict[param_name] = val
            params = BatteryParameters(**params_dict)
        else:
            params = base_params
        
        # Handle external conditions
        if param_name == 'T_ambient':
            T_amb = val
            I = I_load
        elif param_name == 'I_load':
            T_amb = T_ambient
            I = val
        else:
            T_amb = T_ambient
            I = I_load
        
        # Run simulation
        model = ExtendedBatteryModel(params)
        tte = compute_time_to_empty(model, constant_current(I), 
                                    T_ambient=T_amb, max_hours=24)
        outputs.append(tte if tte else 24.0)
    
    outputs = np.array(outputs)
    
    # Compute baseline (middle of range)
    baseline_idx = n_samples // 2
    baseline = outputs[baseline_idx]
    
    # Sensitivity index: normalized range of output
    sensitivity_index = (outputs.max() - outputs.min()) / baseline if baseline > 0 else 0
    
    # Elasticity: (Δoutput/output) / (Δparam/param)
    delta_out = outputs[-1] - outputs[0]
    delta_param = param_values[-1] - param_values[0]
    param_mid = param_values[baseline_idx]
    
    if param_mid != 0 and baseline != 0:
        elasticity = (delta_out / baseline) / (delta_param / param_mid)
    else:
        elasticity = 0
    
    return SensitivityResult(
        parameter=param_name,
        values=param_values,
        outputs=outputs,
        baseline=baseline,
        sensitivity_index=sensitivity_index,
        elasticity=elasticity
    )


def run_full_oat_analysis(save_results: bool = True) -> Dict[str, SensitivityResult]:
    """
    Run OAT analysis for all parameters.
    """
    results = {}
    
    print("Running One-at-a-Time Sensitivity Analysis...")
    print("-" * 50)
    
    for param in PARAM_RANGES.keys():
        print(f"  Analyzing: {param}...", end='')
        results[param] = oat_sensitivity(param)
        print(f" SI={results[param].sensitivity_index:.3f}")
    
    # Rank by sensitivity
    ranked = sorted(results.items(), 
                   key=lambda x: abs(x[1].sensitivity_index), 
                   reverse=True)
    
    print("\nParameter Sensitivity Ranking:")
    print("-" * 50)
    for i, (name, res) in enumerate(ranked, 1):
        _, _, unit, desc = PARAM_RANGES[name]
        print(f"{i:2d}. {desc:25s} SI={res.sensitivity_index:+.3f}, ε={res.elasticity:+.3f}")
    
    return results


# =============================================================================
# SECTION 3: TORNADO DIAGRAM (VISUALIZATION)
# =============================================================================

def plot_tornado_diagram(results: Dict[str, SensitivityResult], 
                         output_file: str = 'figure_tornado.png'):
    """
    Create a tornado diagram showing parameter sensitivity.
    
    This is a classic visualization showing which parameters have the 
    largest impact on the output (time-to-empty).
    """
    # Sort by absolute sensitivity
    sorted_params = sorted(results.items(),
                          key=lambda x: abs(x[1].sensitivity_index),
                          reverse=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_positions = np.arange(len(sorted_params))
    
    for i, (name, res) in enumerate(sorted_params):
        _, _, unit, desc = PARAM_RANGES[name]
        
        # Compute the change from baseline at low and high parameter values
        low_change = res.outputs[0] - res.baseline
        high_change = res.outputs[-1] - res.baseline
        
        # Ensure we have low on left, high on right
        if res.values[-1] > res.values[0]:
            left = low_change
            right = high_change
        else:
            left = high_change
            right = low_change
        
        # Plot bars
        color_left = '#E57373' if left < 0 else '#81C784'
        color_right = '#81C784' if right > 0 else '#E57373'
        
        ax.barh(i, left, color=color_left, height=0.6, align='center')
        ax.barh(i, right, color=color_right, height=0.6, align='center')
    
    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([PARAM_RANGES[name][3] for name, _ in sorted_params])
    ax.axvline(x=0, color='black', linewidth=1)
    
    ax.set_xlabel('Change in Time-to-Empty (hours)', fontsize=12)
    ax.set_title('Tornado Diagram: Parameter Sensitivity Analysis',
                 fontsize=14, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#81C784', label='Increases Battery Life'),
        Patch(facecolor='#E57373', label='Decreases Battery Life')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


# =============================================================================
# SECTION 4: SPIDER/RADAR PLOT
# =============================================================================

def plot_spider_sensitivity(results: Dict[str, SensitivityResult],
                            output_file: str = 'figure_spider.png'):
    """
    Spider (radar) plot showing relative parameter sensitivities.
    """
    # Select top 8 most sensitive parameters
    sorted_params = sorted(results.items(),
                          key=lambda x: abs(x[1].sensitivity_index),
                          reverse=True)[:8]
    
    params = [PARAM_RANGES[name][3] for name, _ in sorted_params]
    values = [abs(res.sensitivity_index) for _, res in sorted_params]
    
    # Normalize to 0-1 scale
    max_val = max(values)
    values_norm = [v / max_val for v in values]
    
    # Number of parameters
    N = len(params)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    
    # Close the plot
    values_norm += values_norm[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    ax.fill(angles, values_norm, color='#2E86AB', alpha=0.25)
    ax.plot(angles, values_norm, color='#2E86AB', linewidth=2, marker='o')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(params, fontsize=10)
    
    ax.set_ylim(0, 1.1)
    ax.set_title('Parameter Sensitivity Spider Plot\n(Normalized Sensitivity Index)',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


# =============================================================================
# SECTION 5: PARAMETER SWEEP PLOTS
# =============================================================================

def plot_parameter_sweeps(results: Dict[str, SensitivityResult],
                          output_file: str = 'figure_param_sweeps.png'):
    """
    Create a grid of parameter sweep plots showing how time-to-empty
    varies with each parameter.
    """
    # Select top 6 parameters
    sorted_params = sorted(results.items(),
                          key=lambda x: abs(x[1].sensitivity_index),
                          reverse=True)[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax, (name, res) in zip(axes, sorted_params):
        p_min, p_max, unit, desc = PARAM_RANGES[name]
        
        ax.plot(res.values, res.outputs, '-o', color='#2E86AB', 
                linewidth=2, markersize=4)
        ax.fill_between(res.values, res.outputs, alpha=0.2, color='#2E86AB')
        
        ax.axhline(y=res.baseline, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel(f'{desc} ({unit})', fontsize=10)
        ax.set_ylabel('Time-to-Empty (hours)', fontsize=10)
        ax.set_title(f'Effect of {desc}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Parameter Sweep Analysis: Time-to-Empty Sensitivity',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


# =============================================================================
# SECTION 6: UNCERTAINTY QUANTIFICATION
# =============================================================================

def monte_carlo_uncertainty(n_samples: int = 500, 
                           I_load: float = 500,
                           T_ambient: float = 298.15) -> Dict:
    """
    Monte Carlo simulation for uncertainty quantification.
    
    Samples all parameters from their uncertainty distributions
    and propagates uncertainty through the model.
    
    Returns:
        Dictionary with TTE distribution statistics
    """
    print(f"Running Monte Carlo with {n_samples} samples...")
    
    tte_samples = []
    
    # Define parameter distributions (uniform within ±10% of nominal)
    base = BatteryParameters()
    
    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"  Sample {i+1}/{n_samples}")
        
        # Sample parameters
        params = BatteryParameters(
            Q_nom=np.random.uniform(0.9, 1.1) * base.Q_nom,
            c=np.random.uniform(0.55, 0.70),
            k=np.random.uniform(0.0015, 0.0025),
            n_peukert=np.random.uniform(1.03, 1.08),
            E_a=np.random.uniform(0.30, 0.40),
            C_th=np.random.uniform(40, 50),
            R_th=np.random.uniform(6, 10),
            R_int_ref=np.random.uniform(0.08, 0.12),
        )
        
        model = ExtendedBatteryModel(params)
        tte = compute_time_to_empty(model, constant_current(I_load),
                                    T_ambient=T_ambient, max_hours=24)
        tte_samples.append(tte if tte else 24.0)
    
    tte_samples = np.array(tte_samples)
    
    results = {
        'samples': tte_samples,
        'mean': np.mean(tte_samples),
        'std': np.std(tte_samples),
        'median': np.median(tte_samples),
        'p5': np.percentile(tte_samples, 5),
        'p95': np.percentile(tte_samples, 95),
        'min': np.min(tte_samples),
        'max': np.max(tte_samples),
    }
    
    print(f"\nUncertainty Analysis Results:")
    print(f"  Mean TTE: {results['mean']:.2f} ± {results['std']:.2f} hours")
    print(f"  90% CI: [{results['p5']:.2f}, {results['p95']:.2f}] hours")
    
    return results


def plot_uncertainty_histogram(mc_results: Dict,
                               output_file: str = 'figure_uncertainty.png'):
    """
    Plot histogram of Monte Carlo time-to-empty results.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    samples = mc_results['samples']
    
    ax.hist(samples, bins=30, density=True, alpha=0.7, color='#2E86AB',
            edgecolor='white', linewidth=1)
    
    # Fit and plot normal distribution
    x = np.linspace(samples.min(), samples.max(), 100)
    from scipy.stats import norm
    pdf = norm.pdf(x, mc_results['mean'], mc_results['std'])
    ax.plot(x, pdf, 'r-', linewidth=2, label='Normal fit')
    
    # Mark statistics
    ax.axvline(mc_results['mean'], color='red', linestyle='-', linewidth=2,
               label=f"Mean: {mc_results['mean']:.2f}h")
    ax.axvline(mc_results['p5'], color='orange', linestyle='--', linewidth=1.5,
               label=f"5th percentile: {mc_results['p5']:.2f}h")
    ax.axvline(mc_results['p95'], color='orange', linestyle='--', linewidth=1.5,
               label=f"95th percentile: {mc_results['p95']:.2f}h")
    
    ax.set_xlabel('Time-to-Empty (hours)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Monte Carlo Uncertainty Quantification\n(500mA load, 25°C ambient)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = (f"N = {len(samples)}\n"
                  f"μ = {mc_results['mean']:.2f} h\n"
                  f"σ = {mc_results['std']:.2f} h\n"
                  f"90% CI: [{mc_results['p5']:.2f}, {mc_results['p95']:.2f}]")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


# =============================================================================
# SECTION 7: MAIN EXECUTION
# =============================================================================

def run_full_sensitivity_analysis():
    """Run complete sensitivity analysis suite."""
    print("="*60)
    print("SENSITIVITY ANALYSIS SUITE")
    print("="*60)
    
    # OAT analysis
    print("\n[1] One-at-a-Time (OAT) Analysis")
    print("-"*60)
    oat_results = run_full_oat_analysis()
    
    # Visualizations
    print("\n[2] Generating Visualizations")
    print("-"*60)
    plot_tornado_diagram(oat_results)
    plot_spider_sensitivity(oat_results)
    plot_parameter_sweeps(oat_results)
    
    # Monte Carlo
    print("\n[3] Monte Carlo Uncertainty Quantification")
    print("-"*60)
    mc_results = monte_carlo_uncertainty(n_samples=300)
    plot_uncertainty_histogram(mc_results)
    
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*60)
    
    return oat_results, mc_results


if __name__ == "__main__":
    run_full_sensitivity_analysis()
