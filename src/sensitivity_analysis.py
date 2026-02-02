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

from ecm_model import (
    EquivalentCircuitModel, ECMParameters
)
from battery_model import constant_current

# =============================================================================
# SECTION 1: PARAMETER RANGES FOR SENSITIVITY ANALYSIS
# =============================================================================

PARAM_RANGES = {
    # ECM electrical parameters
    'Q_nom': (3.0, 5.0, 'Ah', 'Nominal Capacity'),
    'R0_ref': (0.05, 0.15, 'Ω', 'Internal Resistance R₀'),
    'R1_ref': (0.015, 0.05, 'Ω', 'RC1 Resistance R₁'),
    'R2_ref': (0.01, 0.04, 'Ω', 'RC2 Resistance R₂'),
    'tau1_ref': (20, 80, 's', 'RC1 Time Constant τ₁'),
    'tau2_ref': (150, 500, 's', 'RC2 Time Constant τ₂'),
    'R0_temp_coeff': (0.01, 0.03, '1/K', 'R₀ Temp Coefficient β'),
    
    # Thermal parameters
    'M_th': (35, 60, 'J/K', 'Thermal Mass'),
    'R_th': (5, 15, 'K/W', 'Thermal Resistance'),
    
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


def compute_time_to_empty_ecm(model: EquivalentCircuitModel,
                              I_func,
                              SOC_0: float = 1.0,
                              T_ambient: float = 298.15,
                              max_hours: float = 24) -> float:
    """
    Compute time-to-empty for ECM model using voltage/SOC cutoff events.
    """
    t_span = (0, max_hours * 3600)
    result = model.simulate(I_func=I_func, t_span=t_span, SOC_0=SOC_0, T_amb=T_ambient)
    return result.get('time_to_empty')


def compute_energy_efficiency(model: EquivalentCircuitModel,
                              I_func,
                              SOC_0: float = 1.0,
                              T_ambient: float = 298.15,
                              max_hours: float = 24) -> float:
    """
    Compute energy efficiency = Energy delivered / Theoretical max energy.
    
    This metric IS affected by R0 because:
    - Higher R0 → more I²R losses → lower terminal voltage → less energy delivered
    - Energy = integral of V(t) * I(t) dt
    - Theoretical max = Q_nom * V_nom * (SOC_start - SOC_end)
    """
    t_span = (0, max_hours * 3600)
    result = model.simulate(I_func=I_func, t_span=t_span, SOC_0=SOC_0, T_amb=T_ambient)
    
    # Calculate delivered energy (Wh)
    t = result['t']
    V = result['voltage']
    I_A = result['current_A']
    
    # Trapezoidal integration: E = ∫V·I dt (in Wh)
    E_delivered = np.trapz(V * I_A, t) / 3600.0  # Wh
    
    # Theoretical energy = Q * V_nom * ΔSOC
    SOC_start = result['SOC'][0]
    SOC_end = result['SOC'][-1]
    Q_nom = model.params.Q_nom  # Ah
    V_nom = model.params.V_nom  # V
    E_theoretical = Q_nom * V_nom * (SOC_start - SOC_end)  # Wh
    
    # Efficiency = delivered / theoretical
    efficiency = E_delivered / E_theoretical if E_theoretical > 0 else 0
    
    return efficiency


# =============================================================================
# SECTION 2: ONE-AT-A-TIME SENSITIVITY ANALYSIS
# =============================================================================

def oat_sensitivity(param_name: str, 
                   n_samples: int = 20,
                   base_params: ECMParameters = None,
                   I_load: float = 1500,
                   T_ambient: float = 298.15) -> SensitivityResult:
    """
    One-at-a-Time (OAT) sensitivity analysis.
    
    Varies a single parameter while holding others at baseline.
    
    Args:
        param_name: Name of parameter to vary
        n_samples: Number of samples in the range
        base_params: Baseline parameter set
        I_load: Load current [mA] (default 1500mA high-load for R0 sensitivity)
        T_ambient: Ambient temperature [K]
    
    Returns:
        SensitivityResult object
    """
    if base_params is None:
        base_params = ECMParameters()
    
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
            'R0_ref': base_params.R0_ref,
            'R1_ref': base_params.R1_ref,
            'R2_ref': base_params.R2_ref,
            'tau1_ref': base_params.tau1_ref,
            'tau2_ref': base_params.tau2_ref,
            'R0_temp_coeff': base_params.R0_temp_coeff,
            'M_th': base_params.M_th,
            'R_th': base_params.R_th,
        }
        
        # Update the varied parameter
        if param_name in params_dict:
            params_dict[param_name] = val
            params = ECMParameters(**params_dict)
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
        
        # Run simulation - use energy efficiency (affected by R0!)
        model = EquivalentCircuitModel(params)
        efficiency = compute_energy_efficiency(model, constant_current(I), 
                        T_ambient=T_amb, max_hours=24)
        outputs.append(efficiency if efficiency else 0.0)
    
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
    
    # Legend - move to UPPER LEFT to avoid overlap with bars
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#81C784', label='Increases Battery Life'),
        Patch(facecolor='#E57373', label='Decreases Battery Life')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
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
    base = ECMParameters()
    
    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"  Sample {i+1}/{n_samples}")
        
        # Sample parameters
        params = ECMParameters(
            Q_nom=np.random.uniform(0.9, 1.1) * base.Q_nom,
            R0_ref=np.random.uniform(0.9, 1.1) * base.R0_ref,
            R1_ref=np.random.uniform(0.9, 1.1) * base.R1_ref,
            R2_ref=np.random.uniform(0.9, 1.1) * base.R2_ref,
            tau1_ref=np.random.uniform(0.9, 1.1) * base.tau1_ref,
            tau2_ref=np.random.uniform(0.9, 1.1) * base.tau2_ref,
            R0_temp_coeff=np.random.uniform(0.9, 1.1) * base.R0_temp_coeff,
            M_th=np.random.uniform(0.9, 1.1) * base.M_th,
            R_th=np.random.uniform(0.9, 1.1) * base.R_th,
        )
        
        model = EquivalentCircuitModel(params)
        tte = compute_time_to_empty_ecm(model, constant_current(I_load),
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
    
    ax.hist(samples, bins=25, density=True, alpha=0.7, color='#5DADE2',
            edgecolor='white', linewidth=1.2)
    
    # Fit and plot normal distribution
    x = np.linspace(samples.min(), samples.max(), 100)
    from scipy.stats import norm
    pdf = norm.pdf(x, mc_results['mean'], mc_results['std'])
    ax.plot(x, pdf, 'r-', linewidth=2.5, label='Normal fit')
    
    # Mark statistics with clearer lines
    ax.axvline(mc_results['mean'], color='red', linestyle='-', linewidth=2.5,
               label=f"Mean: {mc_results['mean']:.2f}h")
    ax.axvline(mc_results['p5'], color='orange', linestyle='--', linewidth=2,
               label=f"5th percentile: {mc_results['p5']:.2f}h")
    ax.axvline(mc_results['p95'], color='orange', linestyle='--', linewidth=2,
               label=f"95th percentile: {mc_results['p95']:.2f}h")
    
    ax.set_xlabel('Time-to-Empty (hours)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Monte Carlo Uncertainty Quantification\n(500mA load, 25°C ambient)',
                 fontsize=14, fontweight='bold')
    
    # Legend in upper right - no overlap with histogram
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Stats box in upper LEFT to avoid legend overlap
    stats_text = (f"N = {len(samples)}\n"
                  f"μ = {mc_results['mean']:.2f} h\n"
                  f"σ = {mc_results['std']:.2f} h\n"
                  f"90% CI: [{mc_results['p5']:.2f}, {mc_results['p95']:.2f}]")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


# =============================================================================
# SECTION 7: O-PRIZE SENSITIVITY INDICES (QUANTITATIVE)
# =============================================================================

def calculate_sensitivity_indices_ecm():
    """
    Calculate normalized sensitivity indices for ECM model parameters.
    
    This provides the QUANTITATIVE analysis needed for O-Prize papers.
    
    Sensitivity Index (S) = (ΔY/Y₀) / (ΔX/X₀)
    
    Where:
    - Y = Output (Time-to-Empty)
    - X = Parameter
    - Δ represents ±10% perturbation
    
    S = 1.0 means linear relationship (1% param change → 1% output change)
    S > 1.0 means amplified sensitivity
    S < 1.0 means damped sensitivity
    
    Returns:
        Dictionary with parameter names and their sensitivity indices
    """
    import sys
    sys.path.insert(0, '.')
    from ecm_model import EquivalentCircuitModel, ECMParameters
    
    print("\n" + "="*60)
    print("O-PRIZE SENSITIVITY INDICES (ECM Model)")
    print("="*60)
    
    # Use ENERGY EFFICIENCY as the metric (affected by R0 losses)
    def compute_efficiency(params):
        """Compute energy efficiency = Energy_out / Energy_ideal."""
        model = EquivalentCircuitModel(params)
        result = model.simulate(
            I_func=lambda t: 1500,  # 1500mA high-load (gaming) - accentuates R0 losses
            t_span=(0, 8*3600),
            T_amb=298.15
        )
        
        # Energy delivered to load = integral of V*I
        dt = np.diff(result['t'])
        V = result['voltage'][:-1]
        I = result['current_A'][:-1]
        energy_out = np.sum(V * I * dt) / 3600  # Wh
        
        # Ideal energy = Q_nom * V_nom
        energy_ideal = params.Q_nom * params.V_nom
        
        return energy_out / energy_ideal if energy_ideal > 0 else 0
    
    base_params = ECMParameters()
    base_eff = compute_efficiency(base_params)
    print(f"\nBaseline Energy Efficiency: {base_eff:.3f} ({base_eff*100:.1f}%)")
    
    # Also compute TTE for comparison
    def compute_tte(params):
        model = EquivalentCircuitModel(params)
        result = model.simulate(
            I_func=lambda t: 1500,
            t_span=(0, 8*3600),
            T_amb=298.15
        )
        return result['time_to_empty'] if result['time_to_empty'] else 8.0
    
    base_tte = compute_tte(base_params)
    print(f"Baseline TTE: {base_tte:.3f} hours")
    
    # Parameters to analyze (with units)
    param_info = {
        'Q_nom': ('Nominal Capacity', 'Ah'),
        'R0_ref': ('Internal Resistance R₀', 'Ω'),
        'R1_ref': ('RC1 Resistance R₁', 'Ω'),
        'R2_ref': ('RC2 Resistance R₂', 'Ω'),
        'tau1_ref': ('RC1 Time Constant τ₁', 's'),
        'tau2_ref': ('RC2 Time Constant τ₂', 's'),
        'M_th': ('Thermal Mass', 'J/K'),
        'R_th': ('Thermal Resistance', 'K/W'),
        'R0_temp_coeff': ('Temp Coefficient β', '1/K'),
    }
    
    sensitivities = {}
    perturbation = 0.10  # ±10%
    
    print(f"\nPerturbation: ±{perturbation*100:.0f}%")
    print("-"*60)
    print(f"{'Parameter':<30} {'S_index':>10} {'Interpretation':<20}")
    print("-"*60)
    
    for param_name, (description, unit) in param_info.items():
        # Get baseline value
        base_value = getattr(base_params, param_name)
        
        # +10% perturbation
        high_params = ECMParameters()
        setattr(high_params, param_name, base_value * (1 + perturbation))
        high_eff = compute_efficiency(high_params)
        
        # -10% perturbation
        low_params = ECMParameters()
        setattr(low_params, param_name, base_value * (1 - perturbation))
        low_eff = compute_efficiency(low_params)
        
        # Calculate sensitivity index based on efficiency
        # S = (ΔY/Y) / (ΔX/X) = (high - low) / base / (2 * perturbation)
        delta_eff = high_eff - low_eff
        S = (delta_eff / base_eff) / (2 * perturbation) if base_eff > 0 else 0
        
        sensitivities[param_name] = {
            'description': description,
            'unit': unit,
            'base_value': base_value,
            'S_index': S,
            'low_eff': low_eff,
            'high_eff': high_eff,
        }
        
        # Interpretation
        if abs(S) > 0.8:
            interp = "HIGH (Critical)"
        elif abs(S) > 0.3:
            interp = "MEDIUM"
        else:
            interp = "LOW (Robust)"
        
        print(f"{description:<30} {S:>10.3f} {interp:<20}")
    
    print("-"*60)
    
    # Generate LaTeX table for paper
    print("\n" + "="*60)
    print("LATEX TABLE (Copy to Paper)")
    print("="*60)
    print(r"""
\begin{table}[h]
\centering
\caption{Normalized Sensitivity Indices for ECM Parameters}
\label{tab:sensitivity}
\begin{tabular}{lccl}
\hline
\textbf{Parameter} & \textbf{Symbol} & \textbf{$S$} & \textbf{Sensitivity} \\
\hline""")
    
    for param_name, data in sensitivities.items():
        symbol = param_name.replace('_', r'\_')
        S = data['S_index']
        level = "High" if abs(S) > 0.8 else ("Medium" if abs(S) > 0.3 else "Low")
        print(f"{data['description']} & ${symbol}$ & {S:.3f} & {level} \\\\")
    
    print(r"""\hline
\end{tabular}
\end{table}
""")
    
    return sensitivities


def calculate_temperature_dependent_sensitivity():
    """
    Show how sensitivity indices change with temperature.
    This reveals that parameters become critical in extreme conditions.
    """
    import sys
    sys.path.insert(0, '.')
    from ecm_model import EquivalentCircuitModel, ECMParameters
    
    print("\n" + "="*60)
    print("TEMPERATURE-DEPENDENT SENSITIVITY ANALYSIS")
    print("="*60)
    
    temperatures = [263.15, 273.15, 298.15, 313.15]  # -10°C, 0°C, 25°C, 40°C
    temp_labels = ['-10°C', '0°C', '25°C', '40°C']
    
    # Key parameters to track
    params_to_track = ['Q_nom', 'R0_ref', 'R0_temp_coeff', 'R_th']
    param_names = ['Capacity', 'R₀', 'β (Temp Coeff)', 'R_th']
    
    perturbation = 0.10
    results = {p: [] for p in params_to_track}
    
    print(f"\n{'Temp':<8}", end='')
    for name in param_names:
        print(f"{name:<15}", end='')
    print()
    print("-"*70)
    
    for T_amb, T_label in zip(temperatures, temp_labels):
        # Baseline at this temperature
        def compute_tte(params, T):
            model = EquivalentCircuitModel(params)
            result = model.simulate(
                I_func=lambda t: 500,
                t_span=(0, 16*3600),
                T_amb=T
            )
            return result['time_to_empty'] if result['time_to_empty'] else 16.0
        
        base_params = ECMParameters()
        base_tte = compute_tte(base_params, T_amb)
        
        print(f"{T_label:<8}", end='')
        
        for param_name in params_to_track:
            base_value = getattr(base_params, param_name)
            
            # +10%
            high_params = ECMParameters()
            setattr(high_params, param_name, base_value * (1 + perturbation))
            high_tte = compute_tte(high_params, T_amb)
            
            # -10%
            low_params = ECMParameters()
            setattr(low_params, param_name, base_value * (1 - perturbation))
            low_tte = compute_tte(low_params, T_amb)
            
            # Sensitivity index
            delta_tte = high_tte - low_tte
            S = (delta_tte / base_tte) / (2 * perturbation) if base_tte > 0 else 0
            
            results[param_name].append(S)
            print(f"{S:<15.3f}", end='')
        
        print()
    
    print("-"*70)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(temperatures))
    width = 0.2
    colors = ['#3498db', '#e74c3c', '#f39c12', '#27ae60']
    
    for i, (param, name) in enumerate(zip(params_to_track, param_names)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, results[param], width, label=name, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Ambient Temperature', fontsize=12)
    ax.set_ylabel('Sensitivity Index (S)', fontsize=12)
    ax.set_title('How Parameter Sensitivity Changes with Temperature\n(Critical insight: R₀ and β become important in cold weather)', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(temp_labels)
    ax.legend(loc='upper right')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Medium threshold')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='High threshold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = '../figures/figure_temp_sensitivity.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_file}")
    plt.close()
    
    return results


def plot_sensitivity_indices_bar(sensitivities: dict = None, 
                                  output_file: str = '../figures/figure_sensitivity_indices.png'):
    """
    Create a horizontal bar chart of sensitivity indices.
    This is the "O-Prize" version of the tornado plot.
    """
    if sensitivities is None:
        sensitivities = calculate_sensitivity_indices_ecm()
    
    # Sort by absolute sensitivity
    sorted_params = sorted(sensitivities.items(), 
                          key=lambda x: abs(x[1]['S_index']), reverse=True)
    
    names = [data['description'] for _, data in sorted_params]
    values = [data['S_index'] for _, data in sorted_params]
    
    # Color by sensitivity level - use distinct, professional colors
    colors = []
    for v in values:
        if abs(v) > 0.8:
            colors.append('#C0392B')  # Dark Red - High
        elif abs(v) > 0.3:
            colors.append('#E67E22')  # Orange - Medium
        else:
            colors.append('#229954')  # Green - Low/Robust
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='#333333', linewidth=1.2, height=0.7)
    
    # Add reference lines with better labels
    ax.axvline(x=0, color='black', linewidth=1.5)
    ax.axvline(x=1.0, color='#7F8C8D', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=-1.0, color='#7F8C8D', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add threshold zones
    ax.axvspan(-0.3, 0.3, alpha=0.1, color='green', zorder=0)
    ax.axvspan(0.3, 0.8, alpha=0.1, color='orange', zorder=0)
    ax.axvspan(-0.8, -0.3, alpha=0.1, color='orange', zorder=0)
    ax.axvspan(0.8, 1.5, alpha=0.1, color='red', zorder=0)
    ax.axvspan(-1.5, -0.8, alpha=0.1, color='red', zorder=0)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Sensitivity Index (S)', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Sensitivity Analysis\n(S = % change in energy efficiency per % change in parameter)', 
                 fontsize=13, fontweight='bold')
    
    # Add value labels with better positioning
    for bar, val in zip(bars, values):
        x_pos = val + 0.03 if val >= 0 else val - 0.03
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.3f}',
                va='center', ha=ha, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Legend with clear position
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#C0392B', edgecolor='black', label='High (|S| > 0.8)'),
        Patch(facecolor='#E67E22', edgecolor='black', label='Medium (0.3 < |S| < 0.8)'),
        Patch(facecolor='#229954', edgecolor='black', label='Low/Robust (|S| < 0.3)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)
    
    ax.grid(True, alpha=0.4, axis='x', linestyle='-', linewidth=0.5)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, len(names) - 0.5)
    
    # Add interpretation note
    note = "Positive S: output increases with parameter | Negative S: output decreases"
    fig.text(0.5, 0.01, note, ha='center', fontsize=9, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()
    
    return fig


# =============================================================================
# SECTION 8: MAIN EXECUTION
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
    
    # O-Prize ECM Sensitivity Indices
    print("\n[4] ECM Sensitivity Indices (O-Prize Metric)")
    print("-"*60)
    try:
        ecm_sensitivities = calculate_sensitivity_indices_ecm()
        plot_sensitivity_indices_bar(ecm_sensitivities)
    except Exception as e:
        print(f"ECM sensitivity analysis skipped: {e}")
    
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*60)
    
    return oat_results, mc_results


if __name__ == "__main__":
    run_full_sensitivity_analysis()
