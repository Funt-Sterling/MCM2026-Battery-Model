"""
=============================================================================
MAIN EXECUTION SCRIPT
=============================================================================
Runs all components of the battery model analysis.

This script:
1. Runs the core model simulations
2. Generates all O-Prize style figures
3. Performs sensitivity analysis
4. Generates recommendations

Run this to produce all results for the MCM paper.
=============================================================================
"""

import os
import sys
import time

def main():
    """Run complete analysis pipeline."""
    
    print("="*70)
    print("   2026 MCM PROBLEM A: SMARTPHONE BATTERY DRAIN MODEL")
    print("   Complete Analysis Pipeline")
    print("="*70)
    
    start_time = time.time()
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"\nWorking directory: {script_dir}")
    print("\n" + "="*70)
    
    # =========================================================================
    # STEP 1: Core Model Demonstration
    # =========================================================================
    print("\n[STEP 1/4] Running Core Model Simulations...")
    print("-"*70)
    
    from battery_model import (
        BatteryParameters, KineticBatteryModel, ExtendedBatteryModel,
        USAGE_PROFILES, constant_current, periodic_usage, realistic_day_profile
    )
    
    params = BatteryParameters()
    print(f"Battery Parameters:")
    print(f"  - Nominal Capacity: {params.Q_nom} mAh")
    print(f"  - Nominal Voltage: {params.V_nom} V")
    print(f"  - KiBaM c (capacity ratio): {params.c}")
    print(f"  - KiBaM k (recovery rate): {params.k} /s")
    print(f"  - Peukert exponent: {params.n_peukert}")
    print(f"  - Activation energy: {params.E_a} eV")
    
    # Quick model test
    model = ExtendedBatteryModel(params)
    result = model.simulate(constant_current(500), t_span=(0, 10*3600), T_ambient=298.15)
    print(f"\nQuick Test (500mA @ 25°C): Time-to-empty = {result['time_to_empty']:.2f} hours")
    
    # =========================================================================
    # STEP 2: Generate Visualizations
    # =========================================================================
    print("\n" + "="*70)
    print("\n[STEP 2/4] Generating O-Prize Style Visualizations...")
    print("-"*70)
    
    from visualizations import generate_all_figures
    generate_all_figures()
    
    # =========================================================================
    # STEP 3: Sensitivity Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("\n[STEP 3/4] Running Sensitivity Analysis...")
    print("-"*70)
    
    from sensitivity_analysis import run_full_sensitivity_analysis
    oat_results, mc_results = run_full_sensitivity_analysis()
    
    # =========================================================================
    # STEP 4: Generate Recommendations
    # =========================================================================
    print("\n" + "="*70)
    print("\n[STEP 4/4] Generating Recommendations...")
    print("-"*70)
    
    from recommendations import generate_all_recommendations
    recommendations = generate_all_recommendations()
    
    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("   ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nTotal execution time: {elapsed:.1f} seconds")
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(script_dir)):
        if f.endswith(('.png', '.pdf')):
            print(f"  - {f}")
    
    print("\n" + "="*70)
    print("   Ready for MCM Paper Writing!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review generated figures in this directory")
    print("  2. Use figure1_kibam_schematic.png for Section 3 (Basic Model)")
    print("  3. Use figure2_thermal_feedback.png for Section 4 (Extended Model)")
    print("  4. Use figure5_sensitivity_heatmap.png for Section 6 (Sensitivity)")
    print("  5. Use figure_recommendations.png for Section 7 (Recommendations)")
    print("="*70)


if __name__ == "__main__":
    main()
