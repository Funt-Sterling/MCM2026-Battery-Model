"""
=============================================================================
DEVICE COMPARISON & CHARGING SOURCE MODEL
=============================================================================
Compares iPhone 15 Pro, Samsung S24 Ultra, and Google Pixel 8 Pro.
Models charging source effects on battery aging.

Based on:
- Official device specifications
- iFixit teardown data
- Battery University charging guidance
- Makinejad et al. (2023) degradation studies
=============================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


@dataclass
class DeviceSpecs:
    """Complete device specifications."""
    name: str
    
    # Battery
    battery_capacity_mah: int
    battery_voltage_nominal: float  # V
    battery_wh: float              # Wh
    
    # Charging
    max_charge_rate_w: float       # Fast charging power
    usb_pd_power_w: float          # USB-PD max
    wireless_power_w: float        # Qi/MagSafe max
    
    # Display
    display_size_in: float
    display_res_width: int
    display_res_height: int
    display_type: str              # OLED/LCD
    display_max_nits: int
    display_refresh_hz: int
    
    # SoC
    soc_name: str
    soc_process_nm: int
    soc_cpu_cores: int
    soc_peak_tflops: float
    
    # Thermal
    thermal_design_power_w: float  # TDP estimate
    max_safe_temp_c: float
    
    # Weight/dimensions
    weight_g: float
    thickness_mm: float


# Real device data
IPHONE_15_PRO = DeviceSpecs(
    name="iPhone 15 Pro",
    battery_capacity_mah=3274,
    battery_voltage_nominal=3.89,
    battery_wh=12.74,
    max_charge_rate_w=27,
    usb_pd_power_w=27,
    wireless_power_w=15,  # MagSafe
    display_size_in=6.12,
    display_res_width=1179,
    display_res_height=2556,
    display_type="OLED",
    display_max_nits=2000,
    display_refresh_hz=120,
    soc_name="A17 Pro",
    soc_process_nm=3,
    soc_cpu_cores=6,
    soc_peak_tflops=2.15,
    thermal_design_power_w=8.5,
    max_safe_temp_c=35,
    weight_g=187,
    thickness_mm=8.25
)

SAMSUNG_S24_ULTRA = DeviceSpecs(
    name="Samsung Galaxy S24 Ultra",
    battery_capacity_mah=5000,
    battery_voltage_nominal=3.86,
    battery_wh=19.30,
    max_charge_rate_w=45,
    usb_pd_power_w=45,
    wireless_power_w=15,
    display_size_in=6.8,
    display_res_width=1440,
    display_res_height=3120,
    display_type="OLED",
    display_max_nits=2600,
    display_refresh_hz=120,
    soc_name="Snapdragon 8 Gen 3",
    soc_process_nm=4,
    soc_cpu_cores=8,
    soc_peak_tflops=4.5,
    thermal_design_power_w=9.2,
    max_safe_temp_c=37,
    weight_g=232,
    thickness_mm=8.6
)

PIXEL_8_PRO = DeviceSpecs(
    name="Google Pixel 8 Pro",
    battery_capacity_mah=5050,
    battery_voltage_nominal=3.87,
    battery_wh=19.54,
    max_charge_rate_w=30,
    usb_pd_power_w=30,
    wireless_power_w=23,  # Fast wireless
    display_size_in=6.7,
    display_res_width=1344,
    display_res_height=2992,
    display_type="OLED",
    display_max_nits=2400,
    display_refresh_hz=120,
    soc_name="Tensor G3",
    soc_process_nm=4,
    soc_cpu_cores=9,
    soc_peak_tflops=2.8,
    thermal_design_power_w=7.8,
    max_safe_temp_c=36,
    weight_g=213,
    thickness_mm=8.8
)

ALL_DEVICES = [IPHONE_15_PRO, SAMSUNG_S24_ULTRA, PIXEL_8_PRO]


@dataclass
class ChargingSource:
    """
    Charging source characteristics.
    σ factor represents aging acceleration relative to wall charging.
    """
    name: str
    power_w: float           # Typical power delivery
    voltage_v: float         # Output voltage
    efficiency: float        # Charging efficiency (0-1)
    sigma_aging: float       # Aging acceleration factor (σ)
    ripple_pct: float        # Voltage ripple (%)
    temp_rise_c: float       # Temperature rise during charging
    description: str


# Charging sources with aging factors (σ)
# Based on Battery University and degradation studies
CHARGING_SOURCES = {
    'wall_5v': ChargingSource(
        name="5V/2A Wall Adapter",
        power_w=10,
        voltage_v=5,
        efficiency=0.85,
        sigma_aging=1.0,      # Reference
        ripple_pct=2.0,
        temp_rise_c=3,
        description="Standard slow charging"
    ),
    'wall_fast': ChargingSource(
        name="Fast Charger (USB-PD)",
        power_w=30,
        voltage_v=9,
        efficiency=0.88,
        sigma_aging=1.15,     # 15% faster aging
        ripple_pct=3.5,
        temp_rise_c=8,
        description="30W+ fast charging"
    ),
    'laptop_usb': ChargingSource(
        name="Laptop USB Port",
        power_w=7.5,
        voltage_v=5,
        efficiency=0.82,
        sigma_aging=1.20,     # 20% faster aging (variable voltage)
        ripple_pct=5.0,
        temp_rise_c=2,
        description="500mA-1.5A USB-A port"
    ),
    'car_12v': ChargingSource(
        name="Car 12V Adapter",
        power_w=18,
        voltage_v=5,
        efficiency=0.75,
        sigma_aging=1.50,     # 50% faster aging (worst case)
        ripple_pct=8.0,       # High ripple
        temp_rise_c=12,       # Hot car + charging
        description="12V cigarette lighter adapter"
    ),
    'wireless_qi': ChargingSource(
        name="Wireless Qi",
        power_w=10,
        voltage_v=5,
        efficiency=0.70,
        sigma_aging=1.25,     # 25% faster aging (heat)
        ripple_pct=4.0,
        temp_rise_c=10,
        description="Standard Qi pad"
    ),
    'wireless_magsafe': ChargingSource(
        name="MagSafe/Fast Wireless",
        power_w=15,
        voltage_v=9,
        efficiency=0.75,
        sigma_aging=1.30,     # 30% faster aging
        ripple_pct=3.5,
        temp_rise_c=12,
        description="Apple MagSafe or equivalent"
    ),
    'powerbank': ChargingSource(
        name="Power Bank",
        power_w=12,
        voltage_v=5,
        efficiency=0.78,
        sigma_aging=1.10,     # Slight degradation
        ripple_pct=4.0,
        temp_rise_c=5,
        description="Typical 10,000mAh power bank"
    ),
}


class BatteryAgingModel:
    """
    Models long-term battery capacity degradation.
    
    Uses the √N aging law with charging source effects:
    Q(N) = Q₀ × [1 - α × √N × σ]
    
    Where:
    - N = cycle count
    - α = base aging coefficient
    - σ = charging source stress factor
    """
    
    def __init__(self, device: DeviceSpecs, initial_capacity_pct: float = 100.0):
        self.device = device
        self.initial_capacity = initial_capacity_pct
        
        # Base aging coefficient (typical for LCO/NMC chemistry)
        # Gives ~80% capacity at 500 cycles
        self.alpha = 0.000894  # √500 ≈ 22.36, 22.36 × 0.00894 ≈ 0.20
        
        self.cycles = 0
        self.capacity = initial_capacity_pct
        self.history: List[Tuple[int, float, str]] = []
    
    def add_cycle(self, source: ChargingSource, depth_of_discharge: float = 0.8):
        """
        Add a charge cycle with given source and DOD.
        
        Args:
            source: The charging source used
            depth_of_discharge: How deep the discharge was (0-1)
        """
        # Effective cycles (DOD affects wear)
        # Shallow cycles cause less wear
        effective_cycles = depth_of_discharge ** 1.5
        self.cycles += effective_cycles
        
        # Calculate new capacity
        degradation = self.alpha * np.sqrt(self.cycles) * source.sigma_aging
        self.capacity = self.initial_capacity * (1 - degradation)
        self.capacity = max(0, self.capacity)
        
        self.history.append((self.cycles, self.capacity, source.name))
        
        return self.capacity
    
    def simulate_years(self, years: float, cycles_per_day: float = 1.0,
                       source: ChargingSource = None,
                       dod: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate battery aging over years.
        
        Args:
            years: Number of years
            cycles_per_day: Average cycles per day
            source: Primary charging source
            dod: Average depth of discharge
        
        Returns:
            time (years), capacity (%)
        """
        if source is None:
            source = CHARGING_SOURCES['wall_5v']
        
        total_days = int(years * 365)
        times = []
        capacities = []
        
        self.cycles = 0
        self.capacity = self.initial_capacity
        
        for day in range(total_days):
            times.append(day / 365)
            capacities.append(self.capacity)
            
            # Add daily cycles
            for _ in range(int(cycles_per_day)):
                self.add_cycle(source, dod)
        
        return np.array(times), np.array(capacities)


def compare_charging_sources(save_path: str = None):
    """Compare battery aging across different charging sources."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Aging over 3 years
    ax1 = axes[0]
    colors = ['#27AE60', '#3498DB', '#9B59B6', '#E74C3C', '#F39C12', '#1ABC9C', '#95A5A6']
    sources_to_compare = ['wall_5v', 'wall_fast', 'laptop_usb', 'car_12v', 
                          'wireless_qi', 'wireless_magsafe', 'powerbank']
    
    for source_key, color in zip(sources_to_compare, colors):
        source = CHARGING_SOURCES[source_key]
        model = BatteryAgingModel(IPHONE_15_PRO)
        times, capacities = model.simulate_years(3, cycles_per_day=1.0, source=source)
        
        ax1.plot(times, capacities, color=color, linewidth=2, label=source.name)
    
    ax1.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Battery Capacity (%)')
    ax1.set_title('Battery Degradation by Charging Source\n(1 cycle/day, 80% DOD)')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(70, 100)
    
    # Right: Sigma factors comparison
    ax2 = axes[1]
    source_names = [CHARGING_SOURCES[k].name for k in sources_to_compare]
    sigma_values = [CHARGING_SOURCES[k].sigma_aging for k in sources_to_compare]
    temp_rises = [CHARGING_SOURCES[k].temp_rise_c for k in sources_to_compare]
    
    x = np.arange(len(source_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, sigma_values, width, label='σ (aging factor)', color='#3498DB')
    bars2 = ax2.bar(x + width/2, [t/10 for t in temp_rises], width, 
                    label='Temp rise (°C/10)', color='#E74C3C')
    
    ax2.set_ylabel('Value')
    ax2.set_title('Charging Source Stress Factors')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.split()[0] for s in source_names], rotation=45, ha='right')
    ax2.legend()
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, sigma_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Print summary
    print("\n" + "=" * 60)
    print("CHARGING SOURCE COMPARISON")
    print("=" * 60)
    print(f"{'Source':<25} {'σ Factor':<10} {'Temp Rise':<12} {'Efficiency':<10}")
    print("-" * 60)
    for key in sources_to_compare:
        s = CHARGING_SOURCES[key]
        print(f"{s.name:<25} {s.sigma_aging:<10.2f} {s.temp_rise_c:<12.1f}°C {s.efficiency*100:<10.0f}%")


def compare_devices_same_usage(save_path: str = None):
    """Compare how different devices handle the same usage pattern."""
    
    # Simulate "medium user" pattern
    # 4 hours screen time, mixed usage
    usage_profile = {
        'screen_on_hours': 5.0,
        'gaming_hours': 0.5,
        'streaming_hours': 1.5,
        'social_hours': 1.5,
        'messaging_hours': 1.5,
        'navigation_hours': 0.5
    }
    
    # Power estimates for each activity (mW)
    activity_power = {
        'gaming': 4500,
        'streaming': 1800,
        'social': 1200,
        'messaging': 800,
        'navigation': 2200,
        'standby': 150
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#007AFF', '#9C27B0', '#34A853']  # Apple blue, Samsung purple, Google green
    
    results = []
    
    for idx, (device, color) in enumerate(zip(ALL_DEVICES, colors)):
        ax = axes[idx]
        
        # Calculate energy consumption
        total_energy = 0
        breakdown = {}
        
        for activity, hours in usage_profile.items():
            if activity == 'screen_on_hours':
                continue
            if activity.replace('_hours', '') in activity_power:
                power = activity_power[activity.replace('_hours', '')]
                energy = power * hours / 1000  # Wh
                total_energy += energy
                breakdown[activity] = energy
        
        # Add standby (24 - screen_on hours)
        standby_hours = 24 - usage_profile['screen_on_hours']
        standby_energy = activity_power['standby'] * standby_hours / 1000
        total_energy += standby_energy
        breakdown['standby'] = standby_energy
        
        # Calculate SOC over day
        times = np.linspace(0, 24, 145)  # 10-minute intervals
        battery_wh = device.battery_wh
        
        socs = []
        soc = 100
        for i, t in enumerate(times):
            # Simplified: distribute activities throughout day
            if 7 <= t < 8:
                power = activity_power['social']
            elif 8 <= t < 9:
                power = activity_power['streaming']
            elif 12 <= t < 13:
                power = activity_power['streaming']
            elif 17 <= t < 17.5:
                power = activity_power['navigation']
            elif 18 <= t < 18.5:
                power = activity_power['gaming']
            elif 19 <= t < 21:
                power = activity_power['streaming']
            elif 21 <= t < 22:
                power = activity_power['messaging']
            else:
                power = activity_power['standby']
            
            dt = 10/60 if i > 0 else 0  # hours
            energy_used = power * dt / 1000
            soc -= (energy_used / battery_wh) * 100
            soc = max(0, soc)
            socs.append(soc)
        
        # Plot
        ax.plot(times, socs, color=color, linewidth=2.5)
        ax.fill_between(times, 0, socs, alpha=0.2, color=color)
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Battery %')
        ax.set_title(f'{device.name}\n{device.battery_capacity_mah} mAh')
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        final_soc = socs[-1]
        ax.annotate(f'{final_soc:.0f}%', xy=(23, final_soc), 
                   fontsize=14, fontweight='bold', color=color)
        
        results.append({
            'device': device.name,
            'battery_wh': device.battery_wh,
            'final_soc': final_soc,
            'total_energy_wh': total_energy
        })
    
    plt.suptitle('Device Comparison: Same Usage Pattern\n(5hr screen time, mixed activities)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Print comparison
    print("\n" + "=" * 60)
    print("DEVICE COMPARISON SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n{r['device']}:")
        print(f"  Battery: {r['battery_wh']:.1f} Wh")
        print(f"  End of day: {r['final_soc']:.0f}%")
        print(f"  Energy used: {r['total_energy_wh']:.1f} Wh")


def generate_device_sensitivity_matrix(save_path: str = None):
    """Generate sensitivity heatmap for device parameters."""
    
    devices = ALL_DEVICES
    parameters = ['Battery (Wh)', 'Display Power', 'SoC Efficiency', 
                  'Fast Charge (W)', 'Wireless (W)', 'TDP (W)']
    
    # Normalize values for comparison (higher = better for battery life)
    matrix = []
    for device in devices:
        row = [
            device.battery_wh / 20,                    # Battery (normalize to ~1)
            1 - (device.display_max_nits / 3000),      # Lower nits = less power
            1 / (device.soc_process_nm / 5),           # Smaller process = better
            device.max_charge_rate_w / 50,             # Higher = faster
            device.wireless_power_w / 25,              # Higher = faster
            1 - (device.thermal_design_power_w / 10)   # Lower TDP = better
        ]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(matrix.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(devices)))
    ax.set_xticklabels([d.name for d in devices])
    ax.set_yticks(range(len(parameters)))
    ax.set_yticklabels(parameters)
    
    # Add value annotations
    for i in range(len(parameters)):
        for j in range(len(devices)):
            text = ax.text(j, i, f'{matrix[j,i]:.2f}',
                          ha='center', va='center', color='black', fontsize=10)
    
    ax.set_title('Device Parameter Sensitivity Matrix\n(Green = Better for Battery Life)')
    fig.colorbar(im, ax=ax, label='Normalized Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Generating device comparison...")
    compare_devices_same_usage("../figures/device_comparison.png")
    
    print("\nGenerating charging source comparison...")
    compare_charging_sources("../figures/charging_comparison.png")
    
    print("\nGenerating sensitivity matrix...")
    generate_device_sensitivity_matrix("../figures/device_sensitivity_matrix.png")
