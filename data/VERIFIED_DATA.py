"""
=============================================================================
VERIFIED DATA SOURCES FOR MCM 2026 PROBLEM A
=============================================================================
This file contains ONLY data that has been verified from primary sources.
Each value includes its source, verification status, and any caveats.

Created: After comprehensive audit of all data used in the model
=============================================================================
"""

# =============================================================================
# SECTION 1: NASA BATTERY DATA (VERIFIED - Downloaded 2025-02-01)
# Source: NASA Prognostics Center of Excellence
# Dataset: "Battery Data Set" (B0005)
# URL: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
# Citation: B. Saha and K. Goebel (2007). "Battery Data Set", NASA Prognostics 
#           Data Repository, NASA Ames Research Center, Moffett Field, CA
# =============================================================================

NASA_B0005 = {
    "source": "NASA PCoE Battery Data Set",
    "url": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
    "verification": "VERIFIED - Downloaded and extracted 2025-02-01",
    "cell_type": "18650 Li-ion",
    "nominal_capacity_ah": 2.0,  # From README
    "charge_current_a": 1.5,     # CC mode at 1.5A
    "charge_voltage_v": 4.2,     # CV mode at 4.2V
    "discharge_current_a": 2.0,  # CC discharge at 2A
    "cutoff_voltage_v": 2.7,     # For B0005 specifically
    "ambient_temp_c": 24,        # Room temperature
    "eol_criterion": "30% capacity fade",
    
    # ACTUAL DATA EXTRACTED FROM .mat FILE:
    "total_discharge_cycles": 168,
    "initial_capacity_ah": 1.8565,  # Cycle 1 - NOT 2.0Ah nominal!
    "final_capacity_ah": 1.3251,    # Cycle 168
    "capacity_fade_percent": 28.6,
    
    # Key data points (cycle, capacity_ah) - REAL VALUES
    "capacity_data": [
        (1, 1.8565), (10, 1.8246), (20, 1.8470), (30, 1.7943),
        (40, 1.7751), (50, 1.7619), (60, 1.7412), (70, 1.7257),
        (80, 1.6995), (90, 1.6614), (100, 1.6235), (110, 1.5780),
        (120, 1.5286), (130, 1.4722), (140, 1.4200), (150, 1.3573),
        (160, 1.3034), (168, 1.3251),  # Note: slight recovery at end
    ],
}

# =============================================================================
# SECTION 2: 5G POWER DATA (VERIFIED - From Narayanan et al. 2021)
# Source: SIGCOMM '21 Paper
# Title: "A Variegated Look at 5G in the Wild: Performance, Power, and QoE"
# DOI: https://doi.org/10.1145/3452296.3472923
# =============================================================================

NARAYANAN_5G_POWER = {
    "source": "Narayanan et al., SIGCOMM 2021",
    "doi": "10.1145/3452296.3472923",
    "verification": "VERIFIED - Extracted from paper Tables and Figures",
    
    # TABLE 2: Power during RRC state transitions (page 557)
    # IMPORTANT: These are TAIL power values (DRX period), NOT active transmission!
    "tail_power_mw": {
        "verizon_4g": 178,           # Table 2
        "tmobile_4g": 66,            # Table 2
        "verizon_nsa_5g_lowband": 249,   # Table 2 (DSS)
        "verizon_nsa_5g_mmwave": 1092,   # Table 2 - HIGH TAIL POWER
        "tmobile_nsa_5g_lowband": 260,   # Table 2
        "tmobile_sa_5g_lowband": 593,    # Table 2
    },
    
    # 4G→5G switch power (one-time transition cost)
    "switch_power_mw": {
        "verizon_nsa_lowband": 799,      # Table 2
        "verizon_nsa_mmwave": 1494,      # Table 2
        "tmobile_nsa_lowband": 699,      # Table 2
        "tmobile_sa_lowband": 245,       # Table 2
    },
    
    # RRC timers (inactivity timer before entering IDLE)
    "tail_timer_s": {
        "verizon_nsa_5g": 10,        # Section 4.2
        "tmobile_nsa_5g": 10,        # Section 4.2
        "tmobile_sa_5g": 10,         # Section 4.2
    },
    
    # Figure 11: Throughput vs Power (approximate from graph)
    # Units: Power in WATTS, Throughput in Mbps
    # Note: These are ESTIMATED from visual inspection of Figure 11
    "throughput_power_approximate": {
        "verification": "ESTIMATED from Figure 11 - visual inspection",
        "caveat": "Exact values not provided in paper, extracted from graph",
        
        # 4G/LTE (regression line from Figure 11)
        "4g_baseline_w": 2.0,        # ~2W at zero throughput (graph y-intercept)
        "4g_crossover_mbps": 122.71, # Annotated on graph
        
        # 5G Low-band
        "5g_lowband_baseline_w": 2.5,   # Slightly higher baseline
        "5g_lowband_crossover_mbps": 188.78,  # Annotated on graph
        
        # 5G mmWave
        "5g_mmwave_baseline_w": 3.0,    # Highest baseline (~3W at zero throughput)
        "5g_mmwave_crossover_mbps": 186.97,   # Annotated on graph
        "5g_mmwave_max_power_w": 8.0,   # At ~1800 Mbps
    },
    
    # CRITICAL CORRECTION NOTE:
    "correction_note": """
    IMPORTANT: The 1092 mW value is TAIL power (DRX period after data transfer),
    NOT the power during active data transmission. During active transmission,
    power can be 2-8 WATTS depending on throughput (see Figure 11).
    
    Previous model INCORRECTLY used 1092 mW as "CONNECTED state power".
    This has been corrected.
    """,
}

# =============================================================================
# SECTION 3: DEVICE SPECIFICATIONS (PARTIALLY VERIFIED)
# =============================================================================

DEVICE_SPECS = {
    "iphone_15_pro": {
        "source": "Apple Technical Specifications",
        "url": "https://www.apple.com/iphone-15-pro/specs/",
        "verification": "VERIFIED - Official Apple specs",
        "battery_capacity_mah": 3274,  # Official
        "battery_voltage_v": 3.85,     # Typical Li-ion
        "battery_wh": 12.70,           # Calculated: 3.274Ah * 3.85V (can verify)
        "fast_charge_w": 27,           # Apple claims ~50% in 30 min
        "wireless_charge_w": 15,       # MagSafe max
        "display_size_in": 6.1,
        
        # UNVERIFIED - Apple doesn't publish these
        "tdp_w": "NOT_AVAILABLE",      # Apple doesn't publish TDP
        "thermal_mass_j_per_k": "NOT_AVAILABLE",
    },
    
    "samsung_s24_ultra": {
        "source": "Samsung Specifications",
        "url": "https://www.samsung.com/global/galaxy/galaxy-s24-ultra/specs/",
        "verification": "VERIFIED - Official Samsung specs",
        "battery_capacity_mah": 5000,  # Official
        "battery_voltage_v": 3.88,     # From spec sheet
        "battery_wh": 19.40,           # Calculated
        "fast_charge_w": 45,           # Official
        "wireless_charge_w": 15,       # Qi standard
        "display_size_in": 6.8,
        
        # UNVERIFIED
        "tdp_w": "NOT_AVAILABLE",
    },
    
    "google_pixel_8_pro": {
        "source": "Google Specifications",
        "url": "https://store.google.com/product/pixel_8_pro_specs",
        "verification": "VERIFIED - Official Google specs",
        "battery_capacity_mah": 5050,  # Official
        "battery_voltage_v": 3.87,     # Typical
        "battery_wh": 19.54,           # Calculated
        "fast_charge_w": 30,           # Official
        "wireless_charge_w": 23,       # Official
        "display_size_in": 6.7,
        
        # UNVERIFIED
        "tdp_w": "NOT_AVAILABLE",
    },
}

# =============================================================================
# SECTION 4: BATTERY PHYSICS PARAMETERS (LITERATURE VALUES)
# =============================================================================

BATTERY_PHYSICS = {
    "arrhenius_activation_energy": {
        "source": "Multiple battery literature sources",
        "verification": "GENERAL LITERATURE VALUE - ranges vary",
        "value_j_per_mol": 20000,  # Typical for Li-ion SEI growth
        "range_j_per_mol": (15000, 30000),  # Reported range in literature
        "citation": "Schuster et al., J. Power Sources, 2015",
    },
    
    "sei_growth_sqrt_n": {
        "source": "Electrochemistry literature",
        "verification": "WELL-ESTABLISHED - diffusion-limited SEI growth",
        "equation": "Qloss ~ alpha * sqrt(N)",
        "caveat": "Coefficient alpha depends on cell chemistry, temperature, C-rate",
        "citation": "Safari & Delacourt, J. Electrochem. Soc., 2011",
    },
    
    "kibam_concept": {
        "source": "Manwell & McGowan, 1993",
        "verification": "ESTABLISHED MODEL - published in peer-reviewed journal",
        "original_for": "Lead-acid batteries",
        "caveat": "Applicability to Li-ion is approximate",
        "citation": "Manwell & McGowan, Solar Energy, 1993, Vol 50(5), pp 399-405",
    },
}

# =============================================================================
# SECTION 5: CHARGING SOURCE EFFECTS (NOT VERIFIED - FABRICATED)
# =============================================================================

CHARGING_EFFECTS = {
    "verification": "NOT VERIFIED - FABRICATED VALUES",
    "caveat": """
    The sigma_aging factors (stress multipliers) used in the model were
    INVENTED without citation. Real values would require:
    1. Accelerated aging studies at different temperatures
    2. Correlation between charging source and cell temperature
    3. Statistical analysis of degradation rates
    
    For a competition paper, these should either be:
    (a) Removed and replaced with sensitivity analysis
    (b) Marked clearly as "illustrative estimates"
    (c) Sourced from actual publications
    """,
    
    # Original fabricated values (for reference)
    "fabricated_sigma_values": {
        "usb_c_slow": 1.0,       # No basis
        "usb_c_fast": 1.15,     # No basis
        "usb_pd": 1.20,          # No basis
        "car_charger": 1.50,     # No basis  
        "wireless_qi": 1.25,     # No basis
        "wireless_magsafe": 1.30,  # No basis
        "battery_pack": 1.10,    # No basis
    },
    
    # What IS known from literature:
    "literature_facts": {
        "temperature_effect": "Higher temperature accelerates degradation (Arrhenius)",
        "c_rate_effect": "Higher C-rate may increase degradation, but modern cells handle 1-2C well",
        "wireless_heating": "Wireless charging does generate more heat than wired (established)",
        "quantification": "Exact stress factors require experimental data for specific cells",
    },
}

# =============================================================================
# SECTION 6: CALCE OCV DATA (PARTIALLY VERIFIED)
# =============================================================================

CALCE_DATA = {
    "source": "CALCE Battery Research Group, University of Maryland",
    "url": "https://calce.umd.edu/battery-data",
    "verification": "SOURCE EXISTS - but specific values need re-extraction",
    
    "caveat": """
    The OCV polynomial coefficients used in the model were approximated,
    not digitized from actual CALCE dataset. For accurate values:
    1. Download actual CALCE CS2 dataset
    2. Extract OCV vs SOC curve from charging data
    3. Fit polynomial to actual data points
    """,
    
    "general_ocv_shape": {
        "soc_0": "~3.0V",      # Near cutoff voltage
        "soc_50": "~3.6-3.7V", # Mid-range
        "soc_100": "~4.2V",    # Fully charged
        "verification": "GENERAL SHAPE CORRECT - specific coefficients approximate",
    },
}

# =============================================================================
# SUMMARY: DATA QUALITY ASSESSMENT
# =============================================================================

DATA_QUALITY_SUMMARY = {
    "VERIFIED_AND_ACCURATE": [
        "NASA B0005 capacity data (downloaded and extracted)",
        "Narayanan 5G tail power values (Table 2)",
        "Device battery capacities (official specs)",
        "Physical constants (Arrhenius, etc.)",
    ],
    
    "PARTIALLY_VERIFIED": [
        "Narayanan throughput-power relationship (visual extraction from Figure 11)",
        "CALCE OCV general shape (coefficients need re-fitting)",
        "KiBaM model structure (well-established but parameters need tuning)",
    ],
    
    "FABRICATED_OR_UNVERIFIED": [
        "Charging source sigma factors (completely fabricated)",
        "Device TDP values (phones don't publish this)",
        "Thermal mass values (need measurement or literature)",
        "Temperature rise per charging source (fabricated)",
    ],
    
    "CRITICAL_CORRECTIONS_NEEDED": [
        "1092 mW is TAIL power, not CONNECTED transmission power",
        "NASA aging model should use REAL B0005 data, not fabricated curve",
        "Charging stress factors need to be removed or marked as illustrative",
    ],
}

if __name__ == "__main__":
    print("=" * 70)
    print("DATA QUALITY SUMMARY")
    print("=" * 70)
    
    for category, items in DATA_QUALITY_SUMMARY.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")
    
    print("\n" + "=" * 70)
    print("CRITICAL: Review all FABRICATED values before submission!")
    print("=" * 70)
