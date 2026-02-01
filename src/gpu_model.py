"""
GPU Power Model for Mobile Devices
==================================

Based on comprehensive research from:
1. APGPM (Purdue, 2024) - PMC-based power modeling achieving 9-12% MAPE
2. DVFS-aware model (INESC-ID, 2019 IEEE TPDS) - 2.4-4.6% error
3. Real-world measurements from Snapdragon 8 Elite, Apple A18 Pro, Mali G-710

Key insight: Same utilization, 33% power variance depending on workload type.
Traditional P = β·Utilization fails; workload-specific exponents required.

Core equation:
    P_GPU = P_idle + (P_max - P_idle) · u^α · thermal_factor(T)

Where α depends on workload type (rendering, compute, AI inference).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum


class GPUWorkloadType(Enum):
    """
    Different GPU workloads have distinct power signatures.
    From APGPM paper: At 70% utilization on Pixel 7 (Mali G-710):
    - Rendering: ~520 mA (texture/mesh heavy)
    - Compute: ~520 mA (FMA heavy)
    - Neural Network: ~390 mA (FMA/CVT mixed)
    """
    IDLE = "idle"
    RENDERING = "rendering"      # Games, UI, video decode
    COMPUTE = "compute"          # Image processing, filters
    AI_INFERENCE = "ai_inference"  # ML models, camera AI
    VIDEO_ENCODE = "video_encode"  # Recording, streaming
    MIXED = "mixed"              # Typical app usage


@dataclass
class GPUDeviceParams:
    """Device-specific GPU parameters from real measurements."""
    name: str
    P_idle: float       # Idle power [W]
    P_max: float        # Peak power [W]
    P_sustained: float  # Sustainable power under thermal limit [W]
    T_throttle: float   # Temperature threshold for throttling [°C]
    T_critical: float   # Critical temperature (emergency throttle) [°C]
    
    # DVFS parameters
    f_max: float        # Maximum frequency [MHz]
    f_min: float        # Minimum frequency [MHz]
    V_nom: float        # Nominal voltage [V]


# Device database from comprehensive research
DEVICE_DATABASE: Dict[str, GPUDeviceParams] = {
    'snapdragon_8_elite': GPUDeviceParams(
        name="Qualcomm Adreno 830",
        P_idle=0.140,       # W (measured)
        P_max=6.5,          # W (peak burst)
        P_sustained=3.5,    # W (Arknights 60fps sustained)
        T_throttle=45.0,    # °C
        T_critical=50.0,    # °C
        f_max=1000,         # MHz
        f_min=300,          # MHz
        V_nom=0.75,         # V
    ),
    'snapdragon_8_gen3': GPUDeviceParams(
        name="Qualcomm Adreno 750",
        P_idle=0.150,
        P_max=5.5,
        P_sustained=3.2,
        T_throttle=43.0,
        T_critical=48.0,
        f_max=900,
        f_min=280,
        V_nom=0.75,
    ),
    'apple_a18_pro': GPUDeviceParams(
        name="Apple A18 Pro GPU (6-core)",
        P_idle=0.100,       # W (Apple optimized)
        P_max=5.5,          # W (Genshin Impact peak)
        P_sustained=3.5,    # W (with vapor chamber)
        T_throttle=45.0,    # °C (improved from A17's 48°C)
        T_critical=50.0,    # °C
        f_max=1490,         # MHz
        f_min=400,          # MHz
        V_nom=0.70,         # V (estimated, Apple proprietary)
    ),
    'apple_a17_pro': GPUDeviceParams(
        name="Apple A17 Pro GPU (6-core)",
        P_idle=0.120,
        P_max=6.0,
        P_sustained=3.0,    # Throttles in Genshin
        T_throttle=42.0,    # °C (throttles early)
        T_critical=48.0,    # °C (severe throttle)
        f_max=1398,
        f_min=400,
        V_nom=0.72,
    ),
    'mali_g710': GPUDeviceParams(
        name="ARM Mali G-710 (Pixel 7)",
        P_idle=0.110,       # W (measured via power rails)
        P_max=5.0,          # W
        P_sustained=3.0,    # W
        T_throttle=42.0,    # °C
        T_critical=47.0,    # °C
        f_max=850,          # MHz
        f_min=200,          # MHz
        V_nom=0.75,         # V
    ),
    'generic_flagship_2025': GPUDeviceParams(
        name="Generic Flagship GPU (2025)",
        P_idle=0.120,
        P_max=5.5,
        P_sustained=3.2,
        T_throttle=44.0,
        T_critical=49.0,
        f_max=950,
        f_min=300,
        V_nom=0.75,
    ),
}


# Workload-specific power exponents (KEY INSIGHT from APGPM)
# These capture micro-architectural differences that utilization alone misses
WORKLOAD_EXPONENTS: Dict[GPUWorkloadType, float] = {
    GPUWorkloadType.IDLE: 1.0,          # Linear (just leakage scaling)
    GPUWorkloadType.RENDERING: 2.0,     # Quadratic (texture sampling, DRAM-bound)
    GPUWorkloadType.COMPUTE: 1.8,       # Near-quadratic (FMA-heavy, compute-bound)
    GPUWorkloadType.AI_INFERENCE: 1.5,  # Sub-quadratic (mixed ops, often NPU-assisted)
    GPUWorkloadType.VIDEO_ENCODE: 1.7,  # Mixed encode pipeline
    GPUWorkloadType.MIXED: 1.75,        # Average for typical apps
}


# Activity to workload mapping
ACTIVITY_WORKLOAD_MAP: Dict[str, GPUWorkloadType] = {
    'gaming': GPUWorkloadType.RENDERING,
    'video_streaming': GPUWorkloadType.MIXED,      # Decode is light
    'video_recording': GPUWorkloadType.VIDEO_ENCODE,
    'photography': GPUWorkloadType.AI_INFERENCE,   # Computational photography
    'social_media': GPUWorkloadType.MIXED,         # UI + image processing
    'navigation': GPUWorkloadType.RENDERING,       # Map rendering
    'messaging': GPUWorkloadType.IDLE,             # Minimal GPU
    'voice_call': GPUWorkloadType.IDLE,
    'work_apps': GPUWorkloadType.MIXED,
    'idle': GPUWorkloadType.IDLE,
    'screen_off': GPUWorkloadType.IDLE,
}


class MobileGPUPowerModel:
    """
    Physics-based GPU power model for mobile devices.
    
    Key equations from DVFS research (INESC-ID, 2019):
        P_dynamic = C·V²·f
        P_static = V·I_leak(T)
        
    Simplified for mobile (no PMC access):
        P_GPU = P_idle + (P_max - P_idle) · u^α · θ(T)
        
    Where:
        u = utilization [0, 1]
        α = workload-specific exponent
        θ(T) = thermal throttling factor
    """
    
    def __init__(self, device: str = 'generic_flagship_2025'):
        """
        Initialize GPU power model.
        
        Args:
            device: Device identifier from DEVICE_DATABASE
        """
        if device not in DEVICE_DATABASE:
            print(f"Warning: Unknown device '{device}', using generic flagship")
            device = 'generic_flagship_2025'
        
        self.params = DEVICE_DATABASE[device]
        self.device_name = device
        
        # State tracking
        self.current_temp = 25.0  # °C
        self.thermal_history = []
        
        # Thermal model parameters
        self.thermal_mass = 15.0  # J/K (GPU die + heatspreader)
        self.thermal_resistance = 8.0  # K/W (to ambient)
        
    def get_workload_type(self, activity: str) -> GPUWorkloadType:
        """Map activity string to GPU workload type."""
        return ACTIVITY_WORKLOAD_MAP.get(activity.lower(), GPUWorkloadType.MIXED)
    
    def thermal_throttle_factor(self, temperature: float) -> float:
        """
        Compute thermal throttling factor θ(T).
        
        Below T_throttle: θ = 1.0 (no throttling)
        Between T_throttle and T_critical: Linear reduction
        Above T_critical: Severe throttling (θ → 0.3)
        
        Args:
            temperature: GPU junction temperature [°C]
            
        Returns:
            Throttle factor [0.3, 1.0]
        """
        T = temperature
        T_th = self.params.T_throttle
        T_cr = self.params.T_critical
        
        if T <= T_th:
            return 1.0
        elif T >= T_cr:
            return 0.3  # Emergency throttle
        else:
            # Linear interpolation
            return 1.0 - 0.7 * (T - T_th) / (T_cr - T_th)
    
    def leakage_factor(self, temperature: float) -> float:
        """
        Temperature-dependent leakage scaling.
        
        Physics: Leakage current doubles per ~10°C rise.
        I_leak(T) ∝ exp(T/T₀) where T₀ ≈ 20°C
        
        Args:
            temperature: GPU temperature [°C]
            
        Returns:
            Leakage multiplier relative to 25°C
        """
        T_ref = 25.0  # Reference temperature
        T0 = 20.0     # Characteristic temperature
        return np.exp((temperature - T_ref) / T0)
    
    def compute_power(
        self,
        utilization: float,
        workload_type: GPUWorkloadType,
        temperature: float,
        intensity: float = 1.0
    ) -> Tuple[float, Dict]:
        """
        Compute GPU power consumption.
        
        Core equation:
            P_GPU = P_idle·leak(T) + (P_eff - P_idle)·u^α·θ(T)·intensity
            
        Where P_eff = min(P_max, P_sustained) based on thermal state.
        
        Args:
            utilization: GPU utilization [0, 1]
            workload_type: Type of GPU workload
            temperature: GPU temperature [°C]
            intensity: Activity intensity multiplier [0, 1]
            
        Returns:
            Tuple of (power_watts, breakdown_dict)
        """
        # Clamp utilization
        u = np.clip(utilization, 0.0, 1.0)
        intensity = np.clip(intensity, 0.0, 1.5)
        
        # Get workload-specific exponent
        alpha = WORKLOAD_EXPONENTS[workload_type]
        
        # Thermal factors
        theta = self.thermal_throttle_factor(temperature)
        leak_mult = self.leakage_factor(temperature)
        
        # Effective max power (sustained if hot)
        if temperature > self.params.T_throttle - 5:
            P_effective = self.params.P_sustained
        else:
            P_effective = self.params.P_max
        
        # Power components
        P_static = self.params.P_idle * leak_mult
        P_dynamic = (P_effective - self.params.P_idle) * (u ** alpha) * theta * intensity
        
        P_total = P_static + P_dynamic
        
        breakdown = {
            'P_static': P_static,
            'P_dynamic': P_dynamic,
            'P_total': P_total,
            'utilization': u,
            'alpha': alpha,
            'theta_throttle': theta,
            'leak_multiplier': leak_mult,
            'workload_type': workload_type.value,
            'temperature': temperature,
        }
        
        return P_total, breakdown
    
    def compute_power_for_activity(
        self,
        activity: str,
        intensity: float = 1.0,
        temperature: float = 35.0
    ) -> Tuple[float, Dict]:
        """
        Convenience method: compute power from activity string.
        
        Args:
            activity: Activity name (e.g., 'gaming', 'social_media')
            intensity: Activity intensity [0, 1]
            temperature: GPU temperature [°C]
            
        Returns:
            Tuple of (power_watts, breakdown_dict)
        """
        workload = self.get_workload_type(activity)
        
        # Estimate utilization from activity and intensity
        base_utilization = self._activity_base_utilization(activity)
        utilization = base_utilization * intensity
        
        return self.compute_power(utilization, workload, temperature, intensity)
    
    def _activity_base_utilization(self, activity: str) -> float:
        """Base GPU utilization for each activity type."""
        utilization_map = {
            'gaming': 0.85,           # High GPU usage
            'video_streaming': 0.25,  # Hardware decode, light
            'video_recording': 0.50,  # Encode pipeline
            'photography': 0.60,      # Computational photography bursts
            'social_media': 0.35,     # UI + image processing
            'navigation': 0.45,       # Map rendering
            'messaging': 0.10,        # Minimal
            'voice_call': 0.05,       # Almost nothing
            'work_apps': 0.20,        # Light UI
            'idle': 0.02,             # Background
            'screen_off': 0.0,        # Zero
        }
        return utilization_map.get(activity.lower(), 0.20)
    
    def update_thermal_state(
        self,
        power: float,
        dt: float,
        ambient_temp: float = 25.0
    ) -> float:
        """
        Update GPU temperature based on power dissipation.
        
        Lumped thermal model:
            M_th · dT/dt = P - (T - T_amb) / R_th
            
        Args:
            power: GPU power dissipation [W]
            dt: Time step [s]
            ambient_temp: Ambient temperature [°C]
            
        Returns:
            New GPU temperature [°C]
        """
        # Heat equation
        Q_gen = power  # W
        Q_diss = (self.current_temp - ambient_temp) / self.thermal_resistance
        
        dT_dt = (Q_gen - Q_diss) / self.thermal_mass
        self.current_temp += dT_dt * dt
        
        # Clamp to reasonable range
        self.current_temp = np.clip(self.current_temp, ambient_temp, 60.0)
        
        self.thermal_history.append(self.current_temp)
        
        return self.current_temp
    
    def simulate_gaming_session(
        self,
        duration_minutes: float,
        game_intensity: float = 0.9,
        ambient_temp: float = 25.0,
        dt: float = 1.0
    ) -> Dict:
        """
        Simulate a gaming session with thermal evolution.
        
        Args:
            duration_minutes: Gaming session length [min]
            game_intensity: Game graphics intensity [0, 1]
            ambient_temp: Room temperature [°C]
            dt: Simulation time step [s]
            
        Returns:
            Dictionary with time series and summary statistics
        """
        n_steps = int(duration_minutes * 60 / dt)
        
        times = []
        powers = []
        temps = []
        throttle_factors = []
        
        self.current_temp = ambient_temp + 5  # Start slightly warm
        
        for i in range(n_steps):
            t = i * dt
            
            # Compute power at current temperature
            P, breakdown = self.compute_power(
                utilization=0.85 * game_intensity,
                workload_type=GPUWorkloadType.RENDERING,
                temperature=self.current_temp,
                intensity=game_intensity
            )
            
            # Update thermal state
            new_temp = self.update_thermal_state(P, dt, ambient_temp)
            
            times.append(t / 60)  # Convert to minutes
            powers.append(P)
            temps.append(new_temp)
            throttle_factors.append(breakdown['theta_throttle'])
        
        return {
            'time_minutes': np.array(times),
            'power_W': np.array(powers),
            'temperature_C': np.array(temps),
            'throttle_factor': np.array(throttle_factors),
            'energy_Wh': np.trapz(powers, times) / 60,
            'avg_power_W': np.mean(powers),
            'max_temp_C': np.max(temps),
            'throttle_duration_pct': 100 * np.mean(np.array(throttle_factors) < 1.0),
        }


def get_gpu_power_for_device(
    device: str,
    activity: str,
    intensity: float = 1.0,
    temperature: float = 35.0
) -> float:
    """
    Quick utility function to get GPU power for a device/activity combination.
    
    Args:
        device: Device name from DEVICE_DATABASE
        activity: Activity type
        intensity: Activity intensity [0, 1]
        temperature: GPU temperature [°C]
        
    Returns:
        GPU power in Watts
    """
    model = MobileGPUPowerModel(device)
    power, _ = model.compute_power_for_activity(activity, intensity, temperature)
    return power


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

def validate_against_measurements():
    """
    Validate model against real-world measurements from research.
    
    Reference data from:
    - APGPM paper (Pixel 7 measurements)
    - Genshin Impact power tests
    - Arknights sustained gaming tests
    """
    print("=" * 60)
    print("GPU POWER MODEL VALIDATION")
    print("=" * 60)
    
    # Test 1: Pixel 7 (Mali G-710) at 70% utilization
    # From APGPM: Rendering ~520mA, NN ~390mA at ~4V = 2.08W vs 1.56W
    print("\n[Test 1] Mali G-710 workload variance at 70% utilization:")
    model = MobileGPUPowerModel('mali_g710')
    
    P_render, _ = model.compute_power(0.70, GPUWorkloadType.RENDERING, 35.0)
    P_ai, _ = model.compute_power(0.70, GPUWorkloadType.AI_INFERENCE, 35.0)
    
    print(f"  Rendering: {P_render:.2f} W (expected ~2.0-2.5 W)")
    print(f"  AI Inference: {P_ai:.2f} W (expected ~1.5-2.0 W)")
    print(f"  Ratio: {P_render/P_ai:.2f}x (expected ~1.3x)")
    
    # Test 2: Snapdragon 8 Elite sustained gaming (Arknights)
    # From measurements: 3.51W sustained @ 60fps
    print("\n[Test 2] Snapdragon 8 Elite gaming (Arknights reference):")
    model = MobileGPUPowerModel('snapdragon_8_elite')
    
    result = model.simulate_gaming_session(
        duration_minutes=15,
        game_intensity=0.75,  # Arknights is not max intensity
        ambient_temp=25.0
    )
    
    print(f"  Avg Power: {result['avg_power_W']:.2f} W (expected ~3.5 W)")
    print(f"  Max Temp: {result['max_temp_C']:.1f} °C")
    print(f"  Throttle Time: {result['throttle_duration_pct']:.1f}%")
    
    # Test 3: Apple A18 Pro Genshin Impact
    # From measurements: 5-5.5W peak, stable 60fps, 45°C
    print("\n[Test 3] Apple A18 Pro gaming (Genshin Impact):")
    model = MobileGPUPowerModel('apple_a18_pro')
    
    result = model.simulate_gaming_session(
        duration_minutes=20,
        game_intensity=1.0,  # Max settings
        ambient_temp=25.0
    )
    
    print(f"  Avg Power: {result['avg_power_W']:.2f} W (expected ~4-5 W)")
    print(f"  Max Temp: {result['max_temp_C']:.1f} °C (expected ~45°C)")
    print(f"  Throttle Time: {result['throttle_duration_pct']:.1f}%")
    
    # Test 4: Thermal throttling behavior
    print("\n[Test 4] Thermal throttling curve:")
    model = MobileGPUPowerModel('generic_flagship_2025')
    
    for T in [30, 35, 40, 44, 47, 50]:
        theta = model.thermal_throttle_factor(T)
        P, _ = model.compute_power(0.80, GPUWorkloadType.RENDERING, T)
        print(f"  T={T}°C: θ={theta:.2f}, P={P:.2f} W")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    validate_against_measurements()
    
    # Demo: Activity-based power lookup
    print("\n\nACTIVITY-BASED GPU POWER (Generic 2025 Flagship, 35°C):")
    print("-" * 50)
    
    model = MobileGPUPowerModel('generic_flagship_2025')
    
    activities = ['gaming', 'video_streaming', 'photography', 'social_media', 
                  'navigation', 'messaging', 'idle']
    
    for activity in activities:
        P, breakdown = model.compute_power_for_activity(activity, intensity=1.0, temperature=35.0)
        print(f"  {activity:20s}: {P*1000:6.0f} mW  (u={breakdown['utilization']:.2f}, α={breakdown['alpha']:.1f})")
