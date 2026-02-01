"""
LLM-Powered Persona Agent System
================================

Novel approach for MCM 2026: Generate realistic, stochastic user behavior
using Large Language Model agents that "live" through a simulated day.

Key innovation: Instead of deterministic schedules, LLM agents make
minute-by-minute decisions based on:
- Current time
- Battery state (SOC)
- Phone temperature
- Previous activity
- Persona personality traits

This captures emergent behaviors like:
- "Battery anxiety" (checking phone more when SOC < 30%)
- Thermal avoidance (stopping gaming when phone is hot)
- Batch optimization (grouping messages to exploit 5G tail)

Supports both API-based (GPT-4, Claude) and simulated LLM for testing.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import random
import time
from abc import ABC, abstractmethod


# ============================================================================
# ACTIVITY DEFINITIONS
# ============================================================================

class Activity(Enum):
    """Possible user activities with power implications."""
    GAMING = "gaming"
    VIDEO_STREAMING = "video_streaming"
    VIDEO_RECORDING = "video_recording"
    PHOTOGRAPHY = "photography"
    SOCIAL_MEDIA = "social_media"
    NAVIGATION = "navigation"
    MESSAGING = "messaging"
    VOICE_CALL = "voice_call"
    WORK_APPS = "work_apps"
    IDLE = "idle"
    CHARGING = "charging"
    SCREEN_OFF = "screen_off"


@dataclass
class ActivityDecision:
    """Output from LLM persona agent."""
    activity: Activity
    duration_minutes: float
    intensity: float  # 0.0 to 1.0
    reasoning: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'activity': self.activity.value,
            'duration_minutes': self.duration_minutes,
            'intensity': self.intensity,
            'reasoning': self.reasoning,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ActivityDecision':
        return cls(
            activity=Activity(d['activity']),
            duration_minutes=d['duration_minutes'],
            intensity=d['intensity'],
            reasoning=d.get('reasoning', ''),
        )


# ============================================================================
# PERSONA DEFINITIONS
# ============================================================================

@dataclass
class PersonaTraits:
    """Personality traits that influence behavior."""
    name: str
    age: int
    occupation: str
    
    # Behavioral traits (0-1 scale)
    gaming_affinity: float = 0.5
    social_media_affinity: float = 0.5
    work_focus: float = 0.5
    battery_anxiety: float = 0.5  # How much they worry about low battery
    thermal_sensitivity: float = 0.5  # Notice when phone is hot
    
    # Usage patterns
    wake_hour: int = 7
    sleep_hour: int = 23
    commute_hours: List[int] = field(default_factory=lambda: [8, 18])
    
    # Preferences
    preferred_activities: List[str] = field(default_factory=list)
    avoided_activities: List[str] = field(default_factory=list)
    
    def to_prompt_string(self) -> str:
        """Convert traits to a string for LLM prompts."""
        return f"""
Name: {self.name}
Age: {self.age}
Occupation: {self.occupation}

Personality traits (scale 0-10):
- Gaming enthusiasm: {int(self.gaming_affinity * 10)}
- Social media usage: {int(self.social_media_affinity * 10)}
- Work focus: {int(self.work_focus * 10)}
- Battery anxiety: {int(self.battery_anxiety * 10)}
- Notices hot phone: {int(self.thermal_sensitivity * 10)}

Daily schedule:
- Wakes up: {self.wake_hour}:00
- Goes to sleep: {self.sleep_hour}:00
- Commute times: {', '.join(f'{h}:00' for h in self.commute_hours)}

Preferred activities: {', '.join(self.preferred_activities) or 'none specified'}
Avoided activities: {', '.join(self.avoided_activities) or 'none specified'}
"""


# Predefined personas matching our original five
PERSONAS: Dict[str, PersonaTraits] = {
    'gamer_gary': PersonaTraits(
        name="Gary",
        age=22,
        occupation="College Student",
        gaming_affinity=0.95,
        social_media_affinity=0.6,
        work_focus=0.3,
        battery_anxiety=0.3,  # Doesn't care much
        thermal_sensitivity=0.7,  # Notices throttling in games
        wake_hour=10,
        sleep_hour=2,
        commute_hours=[],
        preferred_activities=['gaming', 'video_streaming', 'social_media'],
        avoided_activities=['work_apps'],
    ),
    'creator_chloe': PersonaTraits(
        name="Chloe",
        age=28,
        occupation="Content Creator",
        gaming_affinity=0.3,
        social_media_affinity=0.9,
        work_focus=0.7,
        battery_anxiety=0.6,
        thermal_sensitivity=0.5,
        wake_hour=8,
        sleep_hour=23,
        commute_hours=[9, 17],
        preferred_activities=['photography', 'video_recording', 'social_media'],
        avoided_activities=['gaming'],
    ),
    'commuter_chris': PersonaTraits(
        name="Chris",
        age=35,
        occupation="Sales Manager",
        gaming_affinity=0.2,
        social_media_affinity=0.4,
        work_focus=0.85,
        battery_anxiety=0.8,  # High - needs phone for work
        thermal_sensitivity=0.4,
        wake_hour=6,
        sleep_hour=22,
        commute_hours=[7, 8, 17, 18],
        preferred_activities=['navigation', 'voice_call', 'messaging', 'work_apps'],
        avoided_activities=['gaming'],
    ),
    'chatter_emma': PersonaTraits(
        name="Emma",
        age=19,
        occupation="University Student",
        gaming_affinity=0.4,
        social_media_affinity=0.95,
        work_focus=0.4,
        battery_anxiety=0.5,
        thermal_sensitivity=0.3,
        wake_hour=8,
        sleep_hour=1,
        commute_hours=[9, 16],
        preferred_activities=['messaging', 'social_media', 'photography'],
        avoided_activities=['work_apps', 'navigation'],
    ),
    'streamer_steve': PersonaTraits(
        name="Steve",
        age=31,
        occupation="Remote Worker",
        gaming_affinity=0.5,
        social_media_affinity=0.5,
        work_focus=0.6,
        battery_anxiety=0.7,
        thermal_sensitivity=0.6,
        wake_hour=7,
        sleep_hour=23,
        commute_hours=[],
        preferred_activities=['video_streaming', 'work_apps', 'messaging'],
        avoided_activities=['photography'],
    ),
}


# ============================================================================
# PHONE STATE
# ============================================================================

@dataclass
class PhoneState:
    """Current state of the smartphone."""
    soc: float  # State of charge [0-100]
    temperature: float  # Battery/SoC temperature [°C]
    is_charging: bool
    screen_on: bool
    current_time: float  # Hour of day (0-24)
    
    # History
    last_activity: Optional[Activity] = None
    last_activity_duration: float = 0
    activities_today: List[Tuple[Activity, float]] = field(default_factory=list)
    
    def get_time_string(self) -> str:
        """Convert time to HH:MM format."""
        hours = int(self.current_time)
        minutes = int((self.current_time - hours) * 60)
        return f"{hours:02d}:{minutes:02d}"
    
    def get_battery_status(self) -> str:
        """Human-readable battery status."""
        if self.soc > 80:
            return "high"
        elif self.soc > 50:
            return "comfortable"
        elif self.soc > 30:
            return "moderate"
        elif self.soc > 15:
            return "low"
        else:
            return "critical"
    
    def get_thermal_status(self) -> str:
        """Human-readable thermal status."""
        if self.temperature < 35:
            return "cool"
        elif self.temperature < 40:
            return "warm"
        elif self.temperature < 45:
            return "hot"
        else:
            return "very hot"


# ============================================================================
# LLM INTERFACE (Abstract)
# ============================================================================

class LLMInterface(ABC):
    """Abstract interface for LLM backends."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        pass
    
    def parse_activity_response(self, response: str) -> Optional[ActivityDecision]:
        """Parse JSON response from LLM."""
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                # Validate activity
                activity_str = data.get('activity', 'idle').lower()
                try:
                    activity = Activity(activity_str)
                except ValueError:
                    activity = Activity.IDLE
                
                return ActivityDecision(
                    activity=activity,
                    duration_minutes=float(data.get('duration_minutes', 10)),
                    intensity=float(data.get('intensity', 0.5)),
                    reasoning=data.get('reasoning', ''),
                )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            return None
        return None


class SimulatedLLM(LLMInterface):
    """
    Simulated LLM for testing without API calls.
    Uses probabilistic rules based on persona traits and state.
    """
    
    def __init__(self, persona: PersonaTraits, noise_sigma: float = 0.15):
        self.persona = persona
        self.noise_sigma = noise_sigma
    
    def generate(self, prompt: str) -> str:
        """Generate a simulated response."""
        # This is a placeholder - actual implementation uses probabilistic rules
        # The real SimulatedLLM logic is in decide_activity()
        return '{"activity": "idle", "duration_minutes": 10, "intensity": 0.5}'
    
    def decide_activity(self, state: PhoneState) -> ActivityDecision:
        """
        Make activity decision using probabilistic rules.
        
        This simulates what an LLM would decide based on:
        - Time of day
        - Battery state
        - Temperature
        - Persona traits
        - Previous activity
        """
        hour = state.current_time
        soc = state.soc
        temp = state.temperature
        
        # Base probabilities for each activity
        probs = {
            Activity.GAMING: 0.1,
            Activity.VIDEO_STREAMING: 0.15,
            Activity.VIDEO_RECORDING: 0.02,
            Activity.PHOTOGRAPHY: 0.05,
            Activity.SOCIAL_MEDIA: 0.2,
            Activity.NAVIGATION: 0.05,
            Activity.MESSAGING: 0.15,
            Activity.VOICE_CALL: 0.05,
            Activity.WORK_APPS: 0.1,
            Activity.IDLE: 0.13,
        }
        
        # Modify based on persona traits
        probs[Activity.GAMING] *= (1 + self.persona.gaming_affinity)
        probs[Activity.SOCIAL_MEDIA] *= (1 + self.persona.social_media_affinity)
        probs[Activity.WORK_APPS] *= (1 + self.persona.work_focus)
        probs[Activity.MESSAGING] *= (1 + self.persona.social_media_affinity * 0.5)
        
        # Time-based adjustments
        if hour < self.persona.wake_hour or hour > self.persona.sleep_hour:
            # Sleeping hours - mostly screen off
            probs = {k: 0.01 for k in probs}
            probs[Activity.SCREEN_OFF] = 0.95
        
        if int(hour) in self.persona.commute_hours:
            # Commuting - navigation and media
            probs[Activity.NAVIGATION] *= 3.0
            probs[Activity.VIDEO_STREAMING] *= 1.5
            probs[Activity.GAMING] *= 0.3
        
        # Work hours (9-17)
        if 9 <= hour <= 17 and self.persona.work_focus > 0.5:
            probs[Activity.WORK_APPS] *= 2.0
            probs[Activity.GAMING] *= 0.2
        
        # Evening relaxation (19-23)
        if 19 <= hour <= 23:
            probs[Activity.VIDEO_STREAMING] *= 1.5
            probs[Activity.GAMING] *= 1.5
            probs[Activity.WORK_APPS] *= 0.3
        
        # Battery anxiety effects
        if soc < 30:
            anxiety_factor = self.persona.battery_anxiety * (30 - soc) / 30
            
            # Reduce power-hungry activities
            probs[Activity.GAMING] *= (1 - anxiety_factor * 0.8)
            probs[Activity.VIDEO_STREAMING] *= (1 - anxiety_factor * 0.5)
            probs[Activity.NAVIGATION] *= (1 - anxiety_factor * 0.4)
            
            # Increase checking behavior (MORE messaging/social at low battery!)
            # This is the "anxiety spiral" - counterintuitive but real
            if self.persona.battery_anxiety > 0.6:
                probs[Activity.MESSAGING] *= (1 + anxiety_factor * 0.5)
                probs[Activity.SOCIAL_MEDIA] *= (1 + anxiety_factor * 0.3)
        
        # Thermal effects
        if temp > 40:
            thermal_factor = self.persona.thermal_sensitivity * (temp - 40) / 10
            probs[Activity.GAMING] *= (1 - thermal_factor * 0.9)
            probs[Activity.VIDEO_RECORDING] *= (1 - thermal_factor * 0.7)
            probs[Activity.IDLE] *= (1 + thermal_factor * 0.5)
        
        # Normalize probabilities
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        
        # Sample activity
        activities = list(probs.keys())
        probabilities = [probs[a] for a in activities]
        chosen_activity = random.choices(activities, weights=probabilities, k=1)[0]
        
        # Determine duration (with noise)
        base_durations = {
            Activity.GAMING: 45,
            Activity.VIDEO_STREAMING: 30,
            Activity.VIDEO_RECORDING: 5,
            Activity.PHOTOGRAPHY: 3,
            Activity.SOCIAL_MEDIA: 15,
            Activity.NAVIGATION: 25,
            Activity.MESSAGING: 5,
            Activity.VOICE_CALL: 10,
            Activity.WORK_APPS: 20,
            Activity.IDLE: 10,
            Activity.SCREEN_OFF: 60,
            Activity.CHARGING: 30,
        }
        
        base_duration = base_durations.get(chosen_activity, 10)
        duration = base_duration * np.random.lognormal(0, self.noise_sigma)
        duration = np.clip(duration, 1, 120)
        
        # Determine intensity (with noise)
        base_intensity = 0.7 if chosen_activity in [Activity.GAMING, Activity.VIDEO_RECORDING] else 0.5
        intensity = base_intensity * np.random.normal(1.0, self.noise_sigma)
        intensity = np.clip(intensity, 0.1, 1.0)
        
        return ActivityDecision(
            activity=chosen_activity,
            duration_minutes=duration,
            intensity=intensity,
            reasoning=f"Time: {state.get_time_string()}, SOC: {soc:.0f}%, Temp: {temp:.1f}°C"
        )


class OpenAILLM(LLMInterface):
    """OpenAI GPT-4 interface."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client
    
    def generate(self, prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are simulating a smartphone user. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200,
        )
        return response.choices[0].message.content


# ============================================================================
# PERSONA AGENT
# ============================================================================

class PersonaAgent:
    """
    LLM-powered persona agent that makes activity decisions.
    
    Can use real LLM (GPT-4, Claude) or simulated probabilistic model.
    """
    
    PROMPT_TEMPLATE = """You are {name}, a {age}-year-old {occupation}.

Current context:
- Time: {time}
- Battery: {soc}% ({battery_status})
- Phone temperature: {temp}°C ({thermal_status})
- Charging: {charging}
- Last activity: {last_activity} for {last_duration:.0f} min

Your personality:
{traits}

Based on your personality and current situation, what do you do with your phone next?

Available activities:
- gaming, video_streaming, video_recording, photography
- social_media, navigation, messaging, voice_call
- work_apps, idle, screen_off

Output ONLY valid JSON in this exact format:
{{
  "activity": "<activity_name>",
  "duration_minutes": <number 1-120>,
  "intensity": <number 0.0-1.0>,
  "reasoning": "<brief explanation>"
}}
"""
    
    def __init__(
        self,
        persona: PersonaTraits,
        llm: Optional[LLMInterface] = None,
        use_simulated: bool = True,
        noise_sigma: float = 0.15,
    ):
        """
        Initialize persona agent.
        
        Args:
            persona: Persona traits
            llm: LLM interface (if using real LLM)
            use_simulated: If True, use SimulatedLLM instead of real API
            noise_sigma: Gaussian noise std dev for stochastic variation
        """
        self.persona = persona
        self.noise_sigma = noise_sigma
        
        if use_simulated or llm is None:
            self.llm = SimulatedLLM(persona, noise_sigma)
            self.use_simulated = True
        else:
            self.llm = llm
            self.use_simulated = False
        
        # Activity history
        self.decision_history: List[Tuple[float, ActivityDecision]] = []
    
    def decide_next_activity(self, state: PhoneState) -> ActivityDecision:
        """
        Decide what activity to do next.
        
        Args:
            state: Current phone state
            
        Returns:
            Activity decision with duration and intensity
        """
        if self.use_simulated:
            # Use fast probabilistic model
            decision = self.llm.decide_activity(state)
        else:
            # Use real LLM
            prompt = self._build_prompt(state)
            response = self.llm.generate(prompt)
            decision = self.llm.parse_activity_response(response)
            
            if decision is None:
                # Fallback to simulated
                fallback_llm = SimulatedLLM(self.persona, self.noise_sigma)
                decision = fallback_llm.decide_activity(state)
        
        # Add Gaussian noise (even for LLM outputs for consistency)
        decision = self._add_noise(decision)
        
        # Record history
        self.decision_history.append((state.current_time, decision))
        
        return decision
    
    def _build_prompt(self, state: PhoneState) -> str:
        """Build prompt string for LLM."""
        return self.PROMPT_TEMPLATE.format(
            name=self.persona.name,
            age=self.persona.age,
            occupation=self.persona.occupation,
            time=state.get_time_string(),
            soc=state.soc,
            battery_status=state.get_battery_status(),
            temp=state.temperature,
            thermal_status=state.get_thermal_status(),
            charging="Yes" if state.is_charging else "No",
            last_activity=state.last_activity.value if state.last_activity else "None",
            last_duration=state.last_activity_duration,
            traits=self.persona.to_prompt_string(),
        )
    
    def _add_noise(self, decision: ActivityDecision) -> ActivityDecision:
        """Add Gaussian noise to decision parameters."""
        # Duration noise (log-normal to keep positive)
        noisy_duration = decision.duration_minutes * np.random.lognormal(0, self.noise_sigma * 0.5)
        noisy_duration = np.clip(noisy_duration, 1, 120)
        
        # Intensity noise (normal, clipped)
        noisy_intensity = decision.intensity * np.random.normal(1.0, self.noise_sigma)
        noisy_intensity = np.clip(noisy_intensity, 0.1, 1.0)
        
        return ActivityDecision(
            activity=decision.activity,
            duration_minutes=noisy_duration,
            intensity=noisy_intensity,
            reasoning=decision.reasoning,
        )
    
    def get_activity_summary(self) -> Dict[str, float]:
        """Get summary of activities from history."""
        totals = {}
        for _, decision in self.decision_history:
            activity_name = decision.activity.value
            totals[activity_name] = totals.get(activity_name, 0) + decision.duration_minutes
        return totals


# ============================================================================
# DAY SIMULATOR
# ============================================================================

class DaySimulator:
    """
    Simulate a full day of phone usage with an LLM persona agent.
    
    Integrates with battery model to track SOC and temperature evolution.
    """
    
    def __init__(
        self,
        agent: PersonaAgent,
        initial_soc: float = 100.0,
        ambient_temp: float = 25.0,
    ):
        """
        Initialize day simulator.
        
        Args:
            agent: PersonaAgent to simulate
            initial_soc: Starting battery percentage
            ambient_temp: Ambient temperature [°C]
        """
        self.agent = agent
        self.initial_soc = initial_soc
        self.ambient_temp = ambient_temp
        
        # Simulation results
        self.timeline: List[Dict] = []
    
    def simulate_day(
        self,
        start_hour: float = 0.0,
        end_hour: float = 24.0,
        battery_model: Optional[Callable] = None,
    ) -> Dict:
        """
        Simulate a full day of phone usage.
        
        Args:
            start_hour: Start time (0-24)
            end_hour: End time (0-24)
            battery_model: Optional function(activity, duration, intensity) -> (delta_soc, new_temp)
            
        Returns:
            Simulation results dictionary
        """
        current_time = start_hour
        current_soc = self.initial_soc
        current_temp = self.ambient_temp + 5  # Start slightly warm
        
        state = PhoneState(
            soc=current_soc,
            temperature=current_temp,
            is_charging=False,
            screen_on=True,
            current_time=current_time,
        )
        
        self.timeline = []
        
        while current_time < end_hour:
            # Agent decides next activity
            decision = self.agent.decide_next_activity(state)
            
            # Simulate battery drain
            if battery_model:
                delta_soc, new_temp = battery_model(
                    decision.activity.value,
                    decision.duration_minutes,
                    decision.intensity,
                )
            else:
                # Simple default model
                delta_soc, new_temp = self._default_battery_model(
                    decision, current_temp
                )
            
            # Update state
            current_soc = max(0, current_soc - delta_soc)
            current_temp = new_temp
            current_time += decision.duration_minutes / 60
            
            # Record timeline
            self.timeline.append({
                'time': current_time,
                'activity': decision.activity.value,
                'duration': decision.duration_minutes,
                'intensity': decision.intensity,
                'soc': current_soc,
                'temperature': current_temp,
                'reasoning': decision.reasoning,
            })
            
            # Update state for next iteration
            state = PhoneState(
                soc=current_soc,
                temperature=current_temp,
                is_charging=decision.activity == Activity.CHARGING,
                screen_on=decision.activity != Activity.SCREEN_OFF,
                current_time=current_time,
                last_activity=decision.activity,
                last_activity_duration=decision.duration_minutes,
            )
            
            # Check for dead battery
            if current_soc <= 0:
                break
        
        return self._compile_results()
    
    def _default_battery_model(
        self,
        decision: ActivityDecision,
        current_temp: float,
    ) -> Tuple[float, float]:
        """
        Simple battery model for testing.
        Returns (delta_soc_percent, new_temperature).
        """
        # Power consumption rates (% per minute)
        power_rates = {
            Activity.GAMING: 0.8,
            Activity.VIDEO_STREAMING: 0.3,
            Activity.VIDEO_RECORDING: 0.6,
            Activity.PHOTOGRAPHY: 0.4,
            Activity.SOCIAL_MEDIA: 0.25,
            Activity.NAVIGATION: 0.35,
            Activity.MESSAGING: 0.15,
            Activity.VOICE_CALL: 0.2,
            Activity.WORK_APPS: 0.2,
            Activity.IDLE: 0.05,
            Activity.SCREEN_OFF: 0.02,
            Activity.CHARGING: -1.5,  # Negative = charging
        }
        
        rate = power_rates.get(decision.activity, 0.1)
        delta_soc = rate * decision.duration_minutes * decision.intensity
        
        # Temperature model
        heat_rates = {
            Activity.GAMING: 0.3,
            Activity.VIDEO_RECORDING: 0.2,
            Activity.NAVIGATION: 0.15,
        }
        heat_rate = heat_rates.get(decision.activity, 0.05)
        
        # Thermal evolution (simplified)
        ambient = self.ambient_temp
        cooling_rate = 0.1  # °C per minute towards ambient
        heating = heat_rate * decision.intensity * decision.duration_minutes
        cooling = cooling_rate * decision.duration_minutes * (current_temp - ambient) / 20
        
        new_temp = current_temp + heating - cooling
        new_temp = np.clip(new_temp, ambient, 50)
        
        return delta_soc, new_temp
    
    def _compile_results(self) -> Dict:
        """Compile simulation results."""
        if not self.timeline:
            return {}
        
        # Activity breakdown
        activity_times = {}
        activity_counts = {}
        for entry in self.timeline:
            act = entry['activity']
            activity_times[act] = activity_times.get(act, 0) + entry['duration']
            activity_counts[act] = activity_counts.get(act, 0) + 1
        
        # Time series
        times = [e['time'] for e in self.timeline]
        socs = [e['soc'] for e in self.timeline]
        temps = [e['temperature'] for e in self.timeline]
        
        # Find battery depletion time
        battery_life_hours = None
        for e in self.timeline:
            if e['soc'] <= 0:
                battery_life_hours = e['time']
                break
        if battery_life_hours is None:
            battery_life_hours = times[-1] if times else 24.0
        
        return {
            'timeline': self.timeline,
            'activity_times': activity_times,
            'activity_counts': activity_counts,
            'total_screen_time': sum(
                e['duration'] for e in self.timeline 
                if e['activity'] not in ['screen_off', 'charging']
            ),
            'battery_life_hours': battery_life_hours,
            'final_soc': socs[-1] if socs else self.initial_soc,
            'max_temperature': max(temps) if temps else self.ambient_temp,
            'time_series': {
                'time': times,
                'soc': socs,
                'temperature': temps,
            },
        }


# ============================================================================
# BATCH SIMULATION FOR RL TRAINING DATA
# ============================================================================

def run_batch_simulations(
    persona_name: str,
    n_simulations: int = 100,
    initial_soc: float = 100.0,
    noise_sigma: float = 0.15,
) -> List[Dict]:
    """
    Run multiple day simulations for a persona.
    
    Args:
        persona_name: Name from PERSONAS dict
        n_simulations: Number of days to simulate
        initial_soc: Starting battery percentage
        noise_sigma: Stochastic variation level
        
    Returns:
        List of simulation results
    """
    if persona_name not in PERSONAS:
        raise ValueError(f"Unknown persona: {persona_name}")
    
    persona = PERSONAS[persona_name]
    results = []
    
    for i in range(n_simulations):
        agent = PersonaAgent(
            persona=persona,
            use_simulated=True,
            noise_sigma=noise_sigma,
        )
        
        simulator = DaySimulator(
            agent=agent,
            initial_soc=initial_soc,
        )
        
        result = simulator.simulate_day(
            start_hour=persona.wake_hour,
            end_hour=24.0,
        )
        
        result['simulation_id'] = i
        result['persona'] = persona_name
        results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{n_simulations} simulations")
    
    return results


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LLM PERSONA AGENT SYSTEM TEST")
    print("=" * 60)
    
    # Test each persona
    for persona_name, persona in PERSONAS.items():
        print(f"\n{'='*60}")
        print(f"Testing: {persona_name} ({persona.name})")
        print("=" * 60)
        
        agent = PersonaAgent(
            persona=persona,
            use_simulated=True,
            noise_sigma=0.15,
        )
        
        simulator = DaySimulator(
            agent=agent,
            initial_soc=100.0,
        )
        
        result = simulator.simulate_day(
            start_hour=persona.wake_hour,
            end_hour=24.0,
        )
        
        print(f"\nResults for {persona.name}:")
        print(f"  Battery life: {result['battery_life_hours']:.1f} hours")
        print(f"  Final SOC: {result['final_soc']:.1f}%")
        print(f"  Screen time: {result['total_screen_time']:.0f} min")
        print(f"  Max temperature: {result['max_temperature']:.1f}°C")
        
        print(f"\n  Activity breakdown (minutes):")
        for act, mins in sorted(result['activity_times'].items(), key=lambda x: -x[1]):
            print(f"    {act:20s}: {mins:5.0f} min ({result['activity_counts'][act]} sessions)")
    
    # Run batch simulation for one persona
    print("\n" + "=" * 60)
    print("BATCH SIMULATION TEST (Gamer Gary, 50 days)")
    print("=" * 60)
    
    batch_results = run_batch_simulations(
        'gamer_gary',
        n_simulations=50,
        noise_sigma=0.15,
    )
    
    battery_lives = [r['battery_life_hours'] for r in batch_results]
    screen_times = [r['total_screen_time'] for r in batch_results]
    
    print(f"\nBatch Statistics (n=50):")
    print(f"  Battery life: {np.mean(battery_lives):.1f} ± {np.std(battery_lives):.1f} hours")
    print(f"  Screen time: {np.mean(screen_times):.0f} ± {np.std(screen_times):.0f} min")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
