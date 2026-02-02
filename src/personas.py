"""
=============================================================================
USER PERSONA SIMULATION MODULE
=============================================================================
Models 5 distinct smartphone user archetypes with realistic daily patterns.
Each persona exercises different power subsystems.

Based on MCM 2026 Problem A requirements and real usage research.
=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable
import matplotlib.pyplot as plt
from enum import Enum


@dataclass
class DeviceConfig:
    """
    Device specifications for different smartphones.
    Based on official specs and teardown data.
    """
    name: str
    battery_capacity_wh: float    # Watt-hours
    battery_capacity_mah: float   # mAh at nominal voltage
    
    # Display
    display_type: str             # "OLED" or "LCD"
    display_diagonal_in: float    # Screen size
    display_peak_power_mw: float  # Max brightness
    display_min_power_mw: float   # Minimum brightness
    
    # SoC
    soc_name: str
    soc_peak_power_mw: float
    soc_idle_power_mw: float
    
    # Thermal
    thermal_mass_j_per_k: float   # J/K
    thermal_resistance_k_per_w: float  # K/W (higher = worse cooling)
    throttle_temp_c: float        # Temperature to start throttling
    
    # Battery chemistry
    cycle_capacity_retention: float  # Capacity at 500 cycles (%)
    

# Real device specifications
IPHONE_15_PRO = DeviceConfig(
    name="iPhone 15 Pro",
    battery_capacity_wh=12.7,
    battery_capacity_mah=3274,
    display_type="OLED",
    display_diagonal_in=6.1,
    display_peak_power_mw=1800,
    display_min_power_mw=120,
    soc_name="A17 Pro",
    soc_peak_power_mw=8500,
    soc_idle_power_mw=50,
    thermal_mass_j_per_k=38,
    thermal_resistance_k_per_w=9.5,
    throttle_temp_c=40,
    cycle_capacity_retention=0.80
)

SAMSUNG_S24_ULTRA = DeviceConfig(
    name="Samsung S24 Ultra",
    battery_capacity_wh=18.9,
    battery_capacity_mah=5000,
    display_type="OLED",
    display_diagonal_in=6.8,
    display_peak_power_mw=2400,
    display_min_power_mw=140,
    soc_name="Snapdragon 8 Gen 3",
    soc_peak_power_mw=9200,
    soc_idle_power_mw=60,
    thermal_mass_j_per_k=52,
    thermal_resistance_k_per_w=8.2,
    throttle_temp_c=42,
    cycle_capacity_retention=0.82
)

PIXEL_8_PRO = DeviceConfig(
    name="Google Pixel 8 Pro",
    battery_capacity_wh=19.0,
    battery_capacity_mah=5050,
    display_type="OLED",
    display_diagonal_in=6.7,
    display_peak_power_mw=2100,
    display_min_power_mw=130,
    soc_name="Tensor G3",
    soc_peak_power_mw=7800,
    soc_idle_power_mw=55,
    thermal_mass_j_per_k=48,
    thermal_resistance_k_per_w=8.8,
    throttle_temp_c=41,
    cycle_capacity_retention=0.80
)


class Activity(Enum):
    """User activity types."""
    IDLE = "idle"
    SOCIAL_MEDIA = "social_media"
    MESSAGING = "messaging"
    GAMING = "gaming"
    VIDEO_STREAMING = "video_streaming"
    VIDEO_CALL = "video_call"
    NAVIGATION = "navigation"
    CAMERA_PHOTO = "camera_photo"
    CAMERA_VIDEO = "camera_video"
    MUSIC = "music"
    WEB_BROWSING = "web_browsing"
    EMAIL = "email"


@dataclass
class ActivityPower:
    """Power consumption for an activity."""
    display_pct: float  # Display brightness (0-100%)
    cpu_load: float     # CPU load (0-1)
    gpu_load: float     # GPU load (0-1)
    network_active: bool
    network_throughput_mbps: float  # Expected throughput
    gps_active: bool
    camera_active: bool
    speaker_active: bool
    
    
# Activity power profiles (normalized, scaled by device)
ACTIVITY_PROFILES = {
    Activity.IDLE: ActivityPower(0, 0.02, 0, False, 0, False, False, False),
    Activity.SOCIAL_MEDIA: ActivityPower(60, 0.15, 0.10, True, 5, False, False, False),
    Activity.MESSAGING: ActivityPower(50, 0.08, 0.02, True, 0.5, False, False, False),
    Activity.GAMING: ActivityPower(80, 0.60, 0.50, True, 2, False, False, True),  # CALIBRATED for 3-4h runtime (throttling simulation)
    Activity.VIDEO_STREAMING: ActivityPower(70, 0.12, 0.08, True, 25, False, False, True),
    Activity.VIDEO_CALL: ActivityPower(70, 0.35, 0.15, True, 8, False, True, True),
    Activity.NAVIGATION: ActivityPower(90, 0.25, 0.12, True, 2, True, False, True),
    Activity.CAMERA_PHOTO: ActivityPower(90, 0.40, 0.30, False, 0, False, True, False),
    Activity.CAMERA_VIDEO: ActivityPower(90, 0.60, 0.45, False, 0, False, True, False),
    Activity.MUSIC: ActivityPower(0, 0.05, 0, True, 0.3, False, False, True),
    Activity.WEB_BROWSING: ActivityPower(60, 0.18, 0.08, True, 3, False, False, False),
    Activity.EMAIL: ActivityPower(50, 0.10, 0.03, True, 1, False, False, False),
}


@dataclass
class Persona:
    """
    User persona with usage patterns.
    """
    name: str
    description: str
    device: DeviceConfig
    
    # Daily schedule: list of (start_hour, end_hour, activity, intensity)
    # intensity is a multiplier (0.5-1.5) for variability
    daily_schedule: List[Tuple[float, float, Activity, float]] = field(default_factory=list)
    
    # Key behavior characteristics
    messages_per_day: int = 0
    gaming_hours_per_day: float = 0
    streaming_hours_per_day: float = 0
    commute_hours_per_day: float = 0
    
    # Power profile modifiers
    screen_brightness_bias: float = 1.0  # 1.0 = normal, >1 = bright
    background_apps: int = 5             # Number of background apps
    
    def get_activity_at_time(self, hour: float) -> Tuple[Activity, float]:
        """Get the activity for a given hour of day."""
        for start, end, activity, intensity in self.daily_schedule:
            if start <= hour < end:
                return activity, intensity
        return Activity.IDLE, 1.0
    
    def calculate_power(self, hour: float, ambient_temp: float = 25.0) -> Dict[str, float]:
        """
        Calculate instantaneous power consumption.
        
        Returns dict with power breakdown by component.
        """
        activity, intensity = self.get_activity_at_time(hour)
        profile = ACTIVITY_PROFILES[activity]
        device = self.device
        
        # Display power (OLED scales with brightness and content)
        if profile.display_pct > 0:
            brightness = profile.display_pct * self.screen_brightness_bias / 100
            display_power = device.display_min_power_mw + \
                           (device.display_peak_power_mw - device.display_min_power_mw) * brightness
        else:
            display_power = 0
        
        # CPU power (P = C*V²*f, simplified)
        cpu_power = device.soc_idle_power_mw + \
                   (device.soc_peak_power_mw - device.soc_idle_power_mw) * profile.cpu_load * intensity
        
        # GPU power (similar model)
        gpu_power = profile.gpu_load * device.soc_peak_power_mw * 0.6 * intensity
        
        # Network power (using our RRC model concepts)
        if profile.network_active:
            if profile.network_throughput_mbps > 10:
                network_power = 800  # Efficient high-throughput
            else:
                network_power = 550  # Chatty pattern (weighted average)
        else:
            network_power = 180  # Idle
        
        # GPS power
        gps_power = 250 if profile.gps_active else 0
        
        # Camera power
        if profile.camera_active:
            camera_power = 800 if activity == Activity.CAMERA_VIDEO else 400
        else:
            camera_power = 0
        
        # Speaker power
        speaker_power = 150 if profile.speaker_active else 0
        
        # Baseline (sensors, always-on, background)
        baseline_power = 50 + self.background_apps * 10
        
        total = display_power + cpu_power + gpu_power + network_power + \
                gps_power + camera_power + speaker_power + baseline_power
        
        return {
            'total_mw': total,
            'display_mw': display_power,
            'cpu_mw': cpu_power,
            'gpu_mw': gpu_power,
            'network_mw': network_power,
            'gps_mw': gps_power,
            'camera_mw': camera_power,
            'speaker_mw': speaker_power,
            'baseline_mw': baseline_power,
            'activity': activity.value
        }


# =============================================================================
# THE FIVE PERSONAS
# =============================================================================

def create_gamer_persona() -> Persona:
    """
    Gary the Gamer (15-year-old)
    - Heavy gaming sessions (Genshin Impact, Fortnite)
    - Thermal throttling is the limiting factor
    - Often on charger while playing
    """
    schedule = [
        (0, 7, Activity.IDLE, 1.0),           # Sleep
        (7, 7.5, Activity.SOCIAL_MEDIA, 1.0), # Wake up check
        (7.5, 15, Activity.IDLE, 0.5),        # School (phone in pocket)
        (15, 15.5, Activity.MESSAGING, 1.0),  # After school chat
        (15.5, 18, Activity.GAMING, 1.3),     # GAMING SESSION 1
        (18, 19, Activity.VIDEO_STREAMING, 1.0),  # Dinner + YouTube
        (19, 22, Activity.GAMING, 1.2),       # GAMING SESSION 2
        (22, 23, Activity.SOCIAL_MEDIA, 0.8), # Wind down
        (23, 24, Activity.IDLE, 1.0),         # Sleep
    ]
    
    return Persona(
        name="Gary",
        description="The Gamer (15yo) - Thermal throttling dominates",
        device=IPHONE_15_PRO,  # Smaller battery = faster drain
        daily_schedule=schedule,
        messages_per_day=50,
        gaming_hours_per_day=5.5,
        streaming_hours_per_day=1.0,
        screen_brightness_bias=1.2,  # Bright for gaming
        background_apps=3
    )


def create_creator_persona() -> Persona:
    """
    Claire the Creator (25-year-old)
    - TikTok/Instagram content creator
    - Heavy camera use, video editing
    - Uplink-heavy (uploads drain more than downloads)
    """
    schedule = [
        (0, 8, Activity.IDLE, 1.0),
        (8, 8.5, Activity.SOCIAL_MEDIA, 1.0),    # Check notifications
        (8.5, 9, Activity.CAMERA_VIDEO, 1.2),    # Morning content
        (9, 10, Activity.CAMERA_PHOTO, 1.0),     # Photo shoot
        (10, 12, Activity.VIDEO_STREAMING, 0.8), # "Research" (watching competitors)
        (12, 13, Activity.SOCIAL_MEDIA, 1.3),    # Posting + engaging
        (13, 14, Activity.CAMERA_VIDEO, 1.2),    # More content
        (14, 16, Activity.EMAIL, 0.8),           # Brand deals
        (16, 17, Activity.CAMERA_PHOTO, 1.0),    # Golden hour shoot
        (17, 19, Activity.SOCIAL_MEDIA, 1.2),    # Prime engagement time
        (19, 21, Activity.VIDEO_CALL, 1.0),      # Collab calls
        (21, 23, Activity.SOCIAL_MEDIA, 1.0),    # Evening engagement
        (23, 24, Activity.IDLE, 1.0),
    ]
    
    return Persona(
        name="Claire",
        description="The Creator (25yo) - Camera and uplink dominate",
        device=SAMSUNG_S24_ULTRA,  # Big battery for all-day use
        daily_schedule=schedule,
        messages_per_day=200,
        streaming_hours_per_day=2.0,
        screen_brightness_bias=1.1,
        background_apps=8  # Many apps for content creation
    )


def create_commuter_persona() -> Persona:
    """
    Chris the Commuter (35-year-old)
    - Long commute with GPS navigation
    - Frequent cell handoffs (battery killer)
    - Music streaming during commute
    """
    schedule = [
        (0, 6, Activity.IDLE, 1.0),
        (6, 6.5, Activity.EMAIL, 1.0),           # Morning email check
        (6.5, 8, Activity.NAVIGATION, 1.0),      # COMMUTE TO WORK (90 min)
        (8, 12, Activity.IDLE, 0.3),             # Work (minimal phone)
        (12, 13, Activity.WEB_BROWSING, 1.0),    # Lunch break
        (13, 17, Activity.IDLE, 0.3),            # Work
        (17, 18.5, Activity.NAVIGATION, 1.0),    # COMMUTE HOME (90 min)
        (18.5, 20, Activity.VIDEO_STREAMING, 1.0),  # Evening TV
        (20, 21, Activity.MESSAGING, 0.8),       # Family chat
        (21, 22, Activity.WEB_BROWSING, 0.8),    # News
        (22, 24, Activity.IDLE, 1.0),
    ]
    
    return Persona(
        name="Chris",
        description="The Commuter (35yo) - GPS and handoffs dominate",
        device=PIXEL_8_PRO,
        daily_schedule=schedule,
        messages_per_day=30,
        commute_hours_per_day=3.0,
        streaming_hours_per_day=1.5,
        screen_brightness_bias=1.3,  # Bright for navigation
        background_apps=5
    )


def create_chatty_persona() -> Persona:
    """
    Emma the Chatter (19-year-old)
    - Constantly messaging (WhatsApp, Snapchat, Discord)
    - THE PARADOX USER: Lots of small data = radio never sleeps
    - Light data transfer but high power
    """
    schedule = [
        (0, 7, Activity.IDLE, 1.0),
        (7, 8, Activity.MESSAGING, 1.0),         # Morning texts
        (8, 12, Activity.MESSAGING, 0.8),        # Class (still texting)
        (12, 13, Activity.SOCIAL_MEDIA, 1.0),    # Lunch
        (13, 17, Activity.MESSAGING, 0.9),       # More class
        (17, 18, Activity.MESSAGING, 1.2),       # Post-class chat surge
        (18, 19, Activity.VIDEO_CALL, 1.0),      # Call with friends
        (19, 21, Activity.MESSAGING, 1.3),       # Peak messaging time
        (21, 22, Activity.SOCIAL_MEDIA, 1.0),    # TikTok
        (22, 23, Activity.MESSAGING, 1.1),       # Goodnight texts
        (23, 24, Activity.IDLE, 1.0),
    ]
    
    return Persona(
        name="Emma",
        description="The Chatter (19yo) - Radio tail energy dominates",
        device=IPHONE_15_PRO,
        daily_schedule=schedule,
        messages_per_day=500,  # THE KEY: So many messages!
        screen_brightness_bias=0.9,  # Often low brightness
        background_apps=10  # Many messaging apps
    )


def create_streamer_persona() -> Persona:
    """
    Steve the Streamer (40-year-old)
    - Netflix, YouTube, podcasts
    - Continuous high-throughput = efficient 5G use
    - THE PARADOX COMPARISON: Uses less energy despite more data
    """
    schedule = [
        (0, 7, Activity.IDLE, 1.0),
        (7, 7.5, Activity.EMAIL, 0.8),           # Morning email
        (7.5, 8.5, Activity.VIDEO_STREAMING, 1.0),  # Morning news video
        (8.5, 12, Activity.IDLE, 0.3),           # Work
        (12, 13, Activity.VIDEO_STREAMING, 1.0),    # Lunch video
        (13, 17, Activity.IDLE, 0.3),            # Work
        (17, 17.5, Activity.MUSIC, 1.0),         # Drive home music
        (17.5, 21, Activity.VIDEO_STREAMING, 1.0),  # EVENING STREAMING (3.5 hrs)
        (21, 22, Activity.WEB_BROWSING, 0.8),    # Reading
        (22, 24, Activity.MUSIC, 0.5),           # Sleep music
    ]
    
    return Persona(
        name="Steve",
        description="The Streamer (40yo) - Efficient high-throughput",
        device=SAMSUNG_S24_ULTRA,  # Big screen for streaming
        daily_schedule=schedule,
        messages_per_day=20,
        streaming_hours_per_day=6.0,  # Lots of video
        screen_brightness_bias=1.0,
        background_apps=4
    )


def get_all_personas() -> List[Persona]:
    """Get all five personas."""
    return [
        create_gamer_persona(),
        create_creator_persona(),
        create_commuter_persona(),
        create_chatty_persona(),
        create_streamer_persona()
    ]


def simulate_day(persona: Persona, dt_hours: float = 0.1) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Simulate a full day for a persona.
    
    Args:
        persona: The user persona
        dt_hours: Time step in hours
    
    Returns:
        time array (hours), SOC array (0-1), stats dict
    """
    battery_wh = persona.device.battery_capacity_wh
    soc = 1.0  # Start fully charged
    
    times = []
    socs = []
    powers = []
    activities = []
    
    t = 0
    while t < 24:
        power_data = persona.calculate_power(t)
        power_w = power_data['total_mw'] / 1000
        
        # Drain battery
        energy_used_wh = power_w * dt_hours
        soc -= energy_used_wh / battery_wh
        soc = max(0, soc)
        
        times.append(t)
        socs.append(soc)
        powers.append(power_data['total_mw'])
        activities.append(power_data['activity'])
        
        t += dt_hours
        
        if soc <= 0:
            break
    
    # Statistics
    total_energy_wh = (1 - socs[-1]) * battery_wh if socs else battery_wh
    avg_power_w = np.mean(powers) / 1000 if powers else 0
    screen_on_time = sum(1 for a in activities if a != 'idle') * dt_hours
    
    stats = {
        'final_soc': socs[-1] if socs else 0,
        'total_energy_wh': total_energy_wh,
        'avg_power_w': avg_power_w,
        'screen_on_time_hours': screen_on_time,
        'battery_life_hours': times[-1] if socs[-1] <= 0 else None
    }
    
    return np.array(times), np.array(socs), stats


def compare_all_personas(save_path: str = None):
    """Generate comparison of all 5 personas."""
    personas = get_all_personas()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = ['#E74C3C', '#9B59B6', '#3498DB', '#F39C12', '#27AE60']
    
    all_stats = []
    
    # Simulate and plot each persona
    for i, (persona, color) in enumerate(zip(personas, colors)):
        times, socs, stats = simulate_day(persona)
        all_stats.append((persona.name, stats))
        
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        ax.plot(times, socs * 100, color=color, linewidth=2)
        ax.fill_between(times, 0, socs * 100, alpha=0.3, color=color)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Battery %')
        ax.set_title(f"{persona.name}: {persona.description.split('-')[1].strip()}")
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Low battery')
        
        # Add final SOC annotation
        final_soc = stats['final_soc'] * 100
        ax.annotate(f'{final_soc:.0f}%', xy=(times[-1], final_soc), 
                   fontsize=12, fontweight='bold', color=color)
    
    # Summary comparison in last subplot
    ax = axes[1, 2]
    names = [s[0] for s in all_stats]
    final_socs = [s[1]['final_soc'] * 100 for s in all_stats]
    bars = ax.barh(names, final_socs, color=colors)
    ax.set_xlabel('Battery % at End of Day')
    ax.set_title('End-of-Day Comparison')
    ax.set_xlim(0, 100)
    ax.axvline(x=20, color='red', linestyle='--', alpha=0.5)
    
    for bar, soc in zip(bars, final_socs):
        ax.text(soc + 2, bar.get_y() + bar.get_height()/2, f'{soc:.0f}%', 
               va='center', fontweight='bold')
    
    plt.suptitle('Five User Personas: A Day in Battery Life', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Print summary
    print("\n" + "=" * 60)
    print("PERSONA COMPARISON SUMMARY")
    print("=" * 60)
    for name, stats in all_stats:
        print(f"\n{name}:")
        print(f"  Final SOC: {stats['final_soc']*100:.1f}%")
        print(f"  Energy used: {stats['total_energy_wh']:.1f} Wh")
        print(f"  Avg power: {stats['avg_power_w']*1000:.0f} mW")
        print(f"  Screen-on time: {stats['screen_on_time_hours']:.1f} hours")
    
    return all_stats


def plot_persona_power_breakdown(save_path: str = None):
    """Generate power breakdown by component for each persona."""
    personas = get_all_personas()
    
    # Get average power breakdown for each persona
    components = ['display_mw', 'cpu_mw', 'gpu_mw', 'network_mw', 'gps_mw', 
                  'camera_mw', 'speaker_mw', 'baseline_mw']
    labels = ['Display', 'CPU', 'GPU', 'Network', 'GPS', 'Camera', 'Speaker', 'Baseline']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(personas))
    width = 0.1
    
    component_colors = ['#F1C40F', '#E74C3C', '#9B59B6', '#3498DB', 
                       '#1ABC9C', '#E67E22', '#95A5A6', '#34495E']
    
    for i, (comp, label, color) in enumerate(zip(components, labels, component_colors)):
        values = []
        for persona in personas:
            # Average over day
            avg = np.mean([persona.calculate_power(h)[comp] for h in np.linspace(0, 24, 48)])
            values.append(avg)
        
        offset = (i - len(components)/2) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color)
    
    ax.set_xlabel('User Persona')
    ax.set_ylabel('Average Power (mW)')
    ax.set_title('Power Breakdown by Component for Each Persona')
    ax.set_xticks(x)
    ax.set_xticklabels([p.name for p in personas])
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_persona_radar(save_path: str = None):
    """
    Generate O-Prize radar chart comparing 5 personas across 6 dimensions.

    This is the "Visualizing the Invisible" figure (Golden Rule #3).
    Shows at a glance why the Chatter paradoxically drains more than the Streamer.

    Axes:
    1. Avg Current (mA) - Overall drain rate
    2. Peak Temp (°C) - Thermal stress
    3. Screen-On Time (hours) - Display usage
    4. Tail Energy (mAh) - 5G RRC tail drain (THE KEY INSIGHT)
    5. Data Volume (GB) - Total data transferred
    6. Battery Life (hours) - End result
    """
    personas = get_all_personas()

    # Calculate metrics for each persona
    metrics = []
    for persona in personas:
        times, socs, stats = simulate_day(persona)

        # Calculate additional metrics
        avg_current_ma = (stats['avg_power_w'] * 1000) / 3.7  # Approx at 3.7V

        # Estimate peak temperature (based on power)
        peak_power = max([persona.calculate_power(h)['total_mw'] for h in np.linspace(0, 24, 48)])
        peak_temp = 25 + (peak_power / 1000) * 5  # Rough thermal model

        # Tail energy estimate (based on messages per day)
        # Each message triggers ~15s tail at ~600mW avg
        tail_energy_mah = (persona.messages_per_day * 15 * 0.6) / 3.6  # mAh

        # Data volume (rough estimate)
        data_gb = (persona.streaming_hours_per_day * 3.0 +  # Streaming: 3 GB/hr
                   persona.gaming_hours_per_day * 0.1 +      # Gaming: 0.1 GB/hr
                   persona.messages_per_day * 0.0001)        # Messages: 0.1 MB each

        # Battery life (hours until 20%)
        battery_life = None
        for i, soc in enumerate(socs):
            if soc <= 0.20:
                battery_life = times[i]
                break
        if battery_life is None:
            battery_life = 24  # Lasted all day

        metrics.append({
            'name': persona.name,
            'avg_current': avg_current_ma,
            'peak_temp': peak_temp,
            'screen_on': stats['screen_on_time_hours'],
            'tail_energy': tail_energy_mah,
            'data_volume': data_gb,
            'battery_life': battery_life
        })

    # Normalize to 0-1 scale for radar chart
    keys = ['avg_current', 'peak_temp', 'screen_on', 'tail_energy', 'data_volume', 'battery_life']
    labels = ['Avg Current\n(mA)', 'Peak Temp\n(°C)', 'Screen-On\n(hours)',
              'Tail Energy\n(mAh)', 'Data Volume\n(GB)', 'Battery Life\n(hours)']

    # Get min/max for normalization
    normalized = []
    for m in metrics:
        norm = {}
        for key in keys:
            vals = [x[key] for x in metrics]
            min_val, max_val = min(vals), max(vals)
            if max_val > min_val:
                # Invert battery_life (higher is better, should be outer)
                if key == 'battery_life':
                    norm[key] = (m[key] - min_val) / (max_val - min_val)
                else:
                    norm[key] = (m[key] - min_val) / (max_val - min_val)
            else:
                norm[key] = 0.5
        normalized.append(norm)

    # Create radar chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    # Angles for each axis
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # Colors for each persona
    colors = ['#E74C3C', '#9B59B6', '#3498DB', '#F39C12', '#27AE60']

    for i, (m, norm, color) in enumerate(zip(metrics, normalized, colors)):
        values = [norm[k] for k in keys]
        values += values[:1]  # Close the polygon

        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=m['name'])
        ax.fill(angles, values, alpha=0.15, color=color)

    # Set axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

    # Title
    plt.title('Persona Radar: The Chatty vs Streaming Paradox\n'
              '(Chatter has highest Tail Energy despite lowest Data Volume)',
              fontsize=12, fontweight='bold', y=1.08)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()

    # Print comparison table
    print("\n" + "=" * 80)
    print("PERSONA RADAR METRICS")
    print("=" * 80)
    print(f"{'Persona':<10} {'Avg I (mA)':<12} {'Peak T (°C)':<12} {'Screen (h)':<12} "
          f"{'Tail (mAh)':<12} {'Data (GB)':<10} {'Life (h)':<10}")
    print("-" * 80)
    for m in metrics:
        print(f"{m['name']:<10} {m['avg_current']:<12.0f} {m['peak_temp']:<12.1f} "
              f"{m['screen_on']:<12.1f} {m['tail_energy']:<12.0f} {m['data_volume']:<10.1f} "
              f"{m['battery_life']:<10.1f}")

    return metrics


if __name__ == "__main__":
    print("Generating persona comparison...")
    compare_all_personas("../figures/persona_comparison.png")

    print("\nGenerating power breakdown...")
    plot_persona_power_breakdown("../figures/persona_power_breakdown.png")

    print("\nGenerating persona radar chart (O-Prize enhancement)...")
    plot_persona_radar("../figures/persona_radar.png")
