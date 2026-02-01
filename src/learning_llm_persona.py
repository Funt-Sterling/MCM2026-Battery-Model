"""
Learning LLM Persona Agent System
=================================

This system ACTUALLY LEARNS using the OpenAI API!

Key Innovation for MCM 2026: Instead of static rules, personas:
1. Make decisions using GPT-4
2. Observe outcomes (battery drain, temperature)
3. Reflect on what worked/didn't work
4. Update their behavior based on experience
5. Discover EMERGENT battery-saving strategies

This creates genuinely novel behaviors that emerge from learning,
not hand-coded rules - perfect for the O-Prize!

The learning loop:
    Decision → Action → Outcome → Reflection → Updated Strategy
         ↑___________________________________________|

Author: MCM 2026 Team
Date: February 2026
"""

import os
import json
import time
import random
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
from collections import deque

# Import OpenAI
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Install openai: pip install openai")

# Load .env file if exists
def load_env():
    """Load environment variables from .env file."""
    env_paths = [
        '/home/funt1kk/zjui/SPRING26/IM2C/Problem_A/.env',
        '.env',
        '../.env',
    ]
    for path in env_paths:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
            return True
    return False

load_env()


# ============================================================================
# ACTIVITY AND STATE DEFINITIONS
# ============================================================================

class Activity(Enum):
    """Possible user activities."""
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
    SCREEN_OFF = "screen_off"


# Power consumption (Watts) and heat generation for each activity
ACTIVITY_PROFILES = {
    Activity.GAMING: {'power': 5.2, 'heat': 0.4, 'satisfaction': 0.9},
    Activity.VIDEO_STREAMING: {'power': 2.1, 'heat': 0.15, 'satisfaction': 0.7},
    Activity.VIDEO_RECORDING: {'power': 4.5, 'heat': 0.35, 'satisfaction': 0.8},
    Activity.PHOTOGRAPHY: {'power': 3.2, 'heat': 0.2, 'satisfaction': 0.6},
    Activity.SOCIAL_MEDIA: {'power': 1.8, 'heat': 0.1, 'satisfaction': 0.75},
    Activity.NAVIGATION: {'power': 2.8, 'heat': 0.2, 'satisfaction': 0.5},
    Activity.MESSAGING: {'power': 0.8, 'heat': 0.05, 'satisfaction': 0.6},
    Activity.VOICE_CALL: {'power': 1.2, 'heat': 0.08, 'satisfaction': 0.5},
    Activity.WORK_APPS: {'power': 1.5, 'heat': 0.1, 'satisfaction': 0.4},
    Activity.IDLE: {'power': 0.3, 'heat': 0.02, 'satisfaction': 0.2},
    Activity.SCREEN_OFF: {'power': 0.05, 'heat': 0.0, 'satisfaction': 0.1},
}


@dataclass
class PhoneState:
    """Current state of the smartphone."""
    soc: float  # State of charge [0-100]
    temperature: float  # Battery temperature [°C]
    current_time: float  # Hour of day (0-24)
    is_charging: bool = False
    
    # Context
    last_activity: Optional[str] = None
    last_duration: float = 0
    total_satisfaction: float = 0  # Accumulated satisfaction today
    
    def to_dict(self) -> Dict:
        return {
            'soc': round(self.soc, 1),
            'temperature': round(self.temperature, 1),
            'time': f"{int(self.current_time):02d}:{int((self.current_time % 1) * 60):02d}",
            'battery_status': self._battery_status(),
            'thermal_status': self._thermal_status(),
            'last_activity': self.last_activity or 'None',
            'last_duration': round(self.last_duration, 1),
        }
    
    def _battery_status(self) -> str:
        if self.soc > 80: return "high"
        elif self.soc > 50: return "comfortable"
        elif self.soc > 30: return "moderate"
        elif self.soc > 15: return "low"
        else: return "critical"
    
    def _thermal_status(self) -> str:
        if self.temperature < 35: return "cool"
        elif self.temperature < 40: return "warm"
        elif self.temperature < 45: return "hot"
        else: return "very hot (throttling)"


@dataclass
class Experience:
    """A single experience (state, action, outcome)."""
    timestamp: float
    state_before: Dict
    action: str
    duration_minutes: float
    intensity: float
    state_after: Dict
    reward: float  # Computed reward
    reasoning: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class LearningMemory:
    """Memory system for storing and retrieving experiences."""
    experiences: List[Experience] = field(default_factory=list)
    learned_strategies: List[str] = field(default_factory=list)
    reflection_count: int = 0
    max_experiences: int = 100
    
    def add_experience(self, exp: Experience):
        """Add new experience, removing old ones if at capacity."""
        self.experiences.append(exp)
        if len(self.experiences) > self.max_experiences:
            self.experiences.pop(0)
    
    def get_recent(self, n: int = 10) -> List[Experience]:
        """Get n most recent experiences."""
        return self.experiences[-n:]
    
    def get_similar_states(self, state: PhoneState, n: int = 5) -> List[Experience]:
        """Find experiences with similar states."""
        if not self.experiences:
            return []
        
        # Compute similarity based on SOC and temperature
        similarities = []
        for exp in self.experiences:
            soc_diff = abs(exp.state_before['soc'] - state.soc)
            temp_diff = abs(exp.state_before['temperature'] - state.temperature)
            # Normalize and combine
            similarity = 1.0 / (1.0 + soc_diff/20 + temp_diff/10)
            similarities.append((similarity, exp))
        
        # Return top n
        similarities.sort(key=lambda x: -x[0])
        return [exp for _, exp in similarities[:n]]
    
    def get_stats(self) -> Dict:
        """Get learning statistics."""
        if not self.experiences:
            return {'total': 0, 'avg_reward': 0}
        
        rewards = [e.reward for e in self.experiences]
        return {
            'total': len(self.experiences),
            'avg_reward': np.mean(rewards),
            'best_reward': max(rewards),
            'worst_reward': min(rewards),
            'strategies_learned': len(self.learned_strategies),
        }


# ============================================================================
# PERSONA DEFINITIONS
# ============================================================================

@dataclass
class PersonaTraits:
    """Personality traits that influence goals and preferences."""
    name: str
    age: int
    occupation: str
    
    # Goals (what they care about 0-1)
    values_battery_life: float = 0.5
    values_entertainment: float = 0.5
    values_productivity: float = 0.5
    values_social_connection: float = 0.5
    
    # Schedule
    wake_hour: int = 7
    sleep_hour: int = 23
    
    # Personality description for prompts
    personality: str = ""
    
    def to_prompt_string(self) -> str:
        return f"""
Name: {self.name}
Age: {self.age}
Occupation: {self.occupation}
Personality: {self.personality}

What {self.name} cares about (0-10 scale):
- Battery life: {int(self.values_battery_life * 10)}
- Entertainment: {int(self.values_entertainment * 10)}  
- Productivity: {int(self.values_productivity * 10)}
- Staying connected: {int(self.values_social_connection * 10)}

Schedule: Wakes at {self.wake_hour}:00, sleeps at {self.sleep_hour}:00
"""


# Predefined personas
PERSONAS = {
    'gamer_gary': PersonaTraits(
        name="Gary",
        age=22,
        occupation="College Student",
        values_battery_life=0.3,
        values_entertainment=0.95,
        values_productivity=0.2,
        values_social_connection=0.6,
        wake_hour=10,
        sleep_hour=2,
        personality="Gaming enthusiast who doesn't care much about battery until it dies. "
                   "Loves mobile games, watches streams, stays up late. Will game until "
                   "phone overheats then complain about it.",
    ),
    'anxious_anna': PersonaTraits(
        name="Anna",
        age=28,
        occupation="Marketing Manager",
        values_battery_life=0.9,
        values_entertainment=0.4,
        values_productivity=0.8,
        values_social_connection=0.7,
        wake_hour=6,
        sleep_hour=22,
        personality="Always worried about battery dying. Checks battery percentage constantly. "
                   "Uses dark mode, closes apps obsessively. Actually uses MORE battery due to "
                   "anxiety-driven phone checking.",
    ),
    'balanced_ben': PersonaTraits(
        name="Ben",
        age=35,
        occupation="Software Engineer",
        values_battery_life=0.6,
        values_entertainment=0.5,
        values_productivity=0.7,
        values_social_connection=0.5,
        wake_hour=7,
        sleep_hour=23,
        personality="Rational and efficient. Balances phone usage with battery conservation. "
                   "Will batch notifications, use low power mode proactively, and plan "
                   "heavy activities when charging is available soon.",
    ),
    'social_sara': PersonaTraits(
        name="Sara",
        age=19,
        occupation="University Student",
        values_battery_life=0.3,
        values_entertainment=0.7,
        values_productivity=0.3,
        values_social_connection=0.95,
        wake_hour=8,
        sleep_hour=1,
        personality="Lives on social media. Constantly messaging, posting, checking notifications. "
                   "Phone is always in hand. Gets anxious if can't respond immediately. "
                   "Will find a charger rather than reduce usage.",
    ),
}


# ============================================================================
# LEARNING LLM PERSONA AGENT
# ============================================================================

class LearningLLMPersona:
    """
    An LLM-powered persona that ACTUALLY LEARNS from experience!
    
    Uses OpenAI GPT-4 for:
    1. Making activity decisions based on personality
    2. Reflecting on past experiences
    3. Updating strategies based on what worked
    
    The key insight: The LLM doesn't just follow rules - it DISCOVERS
    emergent battery-saving behaviors through experience.
    """
    
    DECISION_PROMPT = """You are {name}, {personality}

CURRENT SITUATION:
{state}

YOUR PAST EXPERIENCES:
{experiences}

STRATEGIES YOU'VE LEARNED:
{strategies}

What do you do with your phone next? Consider:
1. Your personality and what you care about
2. Current battery/temperature situation
3. What you've learned from past experiences
4. Time of day and what makes sense

Output ONLY valid JSON:
{{
    "activity": "<one of: gaming, video_streaming, video_recording, photography, social_media, navigation, messaging, voice_call, work_apps, idle, screen_off>",
    "duration_minutes": <number 1-60>,
    "intensity": <0.0 to 1.0 how intensely you do it>,
    "reasoning": "<1-2 sentences explaining your decision>"
}}"""

    REFLECTION_PROMPT = """You are {name}. Analyze these recent experiences and extract insights.

RECENT EXPERIENCES:
{experiences}

YOUR PERSONALITY:
{personality}

YOUR CURRENT STRATEGIES:
{strategies}

Questions to consider:
1. What activities drained battery faster/slower than expected?
2. Did high temperatures affect anything?
3. Were there times you wished you had more battery?
4. What patterns do you notice?

Based on this analysis, what's ONE new strategy you should adopt?
The strategy should be specific and actionable, like:
- "When battery drops below 30%, avoid gaming"
- "If phone is hot, wait before video recording"
- "Batch social media checks instead of constant checking"

Output ONLY valid JSON:
{{
    "insight": "<what you noticed>",
    "new_strategy": "<specific strategy to adopt>",
    "confidence": <0.0 to 1.0 how confident you are this will help>
}}"""

    def __init__(
        self,
        persona: PersonaTraits,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        enable_learning: bool = True,
        reflection_interval: int = 10,  # Reflect every N decisions
    ):
        """
        Initialize learning persona.
        
        Args:
            persona: Personality traits
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (gpt-4o-mini is fast/cheap, gpt-4o for better quality)
            enable_learning: Whether to do reflection and learning
            reflection_interval: How often to reflect
        """
        self.persona = persona
        self.model = model
        self.enable_learning = enable_learning
        self.reflection_interval = reflection_interval
        
        # Initialize OpenAI client
        api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key required! Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = OpenAI(api_key=api_key)
        
        # Learning memory
        self.memory = LearningMemory()
        self.decision_count = 0
        
        # Tracking
        self.api_calls = 0
        self.total_tokens = 0
        
        print(f"🧠 Initialized Learning LLM Persona: {persona.name}")
        print(f"   Model: {model}")
        print(f"   Learning: {'Enabled' if enable_learning else 'Disabled'}")
    
    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call OpenAI API and return response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are simulating a smartphone user. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=300,
            )
            
            self.api_calls += 1
            self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️ API Error: {e}")
            return '{"activity": "idle", "duration_minutes": 5, "intensity": 0.3, "reasoning": "API error fallback"}'
    
    def _parse_json(self, text: str) -> Optional[Dict]:
        """Parse JSON from LLM response."""
        try:
            # Find JSON in response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error: {e}")
        return None
    
    def _format_experiences(self, experiences: List[Experience]) -> str:
        """Format experiences for prompt."""
        if not experiences:
            return "No experiences yet - this is your first time!"
        
        lines = []
        for i, exp in enumerate(experiences[-5:], 1):  # Last 5 experiences
            reward_emoji = "✅" if exp.reward > 0.5 else "⚠️" if exp.reward > 0 else "❌"
            lines.append(
                f"{i}. {exp.action} for {exp.duration_minutes:.0f}min: "
                f"Battery {exp.state_before['soc']:.0f}%→{exp.state_after['soc']:.0f}%, "
                f"Temp {exp.state_before['temperature']:.0f}°C→{exp.state_after['temperature']:.0f}°C "
                f"{reward_emoji}"
            )
        return "\n".join(lines)
    
    def _format_strategies(self) -> str:
        """Format learned strategies for prompt."""
        if not self.memory.learned_strategies:
            return "No strategies learned yet - you're still figuring things out!"
        
        return "\n".join(f"- {s}" for s in self.memory.learned_strategies[-5:])
    
    def decide_activity(self, state: PhoneState) -> Tuple[Activity, float, float, str]:
        """
        Decide what activity to do next using GPT-4.
        
        Returns:
            (activity, duration_minutes, intensity, reasoning)
        """
        # Build prompt
        prompt = self.DECISION_PROMPT.format(
            name=self.persona.name,
            personality=self.persona.personality,
            state=json.dumps(state.to_dict(), indent=2),
            experiences=self._format_experiences(self.memory.get_recent(5)),
            strategies=self._format_strategies(),
        )
        
        # Call LLM
        response = self._call_llm(prompt)
        data = self._parse_json(response)
        
        if data is None:
            # Fallback
            return Activity.IDLE, 5.0, 0.3, "Parse error fallback"
        
        # Parse activity
        try:
            activity = Activity(data.get('activity', 'idle').lower())
        except ValueError:
            activity = Activity.IDLE
        
        duration = float(data.get('duration_minutes', 10))
        duration = np.clip(duration, 1, 60)
        
        intensity = float(data.get('intensity', 0.5))
        intensity = np.clip(intensity, 0.1, 1.0)
        
        reasoning = data.get('reasoning', '')
        
        self.decision_count += 1
        
        return activity, duration, intensity, reasoning
    
    def record_experience(
        self,
        state_before: PhoneState,
        activity: Activity,
        duration: float,
        intensity: float,
        state_after: PhoneState,
        reasoning: str = "",
    ):
        """Record an experience and optionally trigger reflection."""
        # Compute reward
        reward = self._compute_reward(state_before, activity, duration, state_after)
        
        exp = Experience(
            timestamp=state_before.current_time,
            state_before=state_before.to_dict(),
            action=activity.value,
            duration_minutes=duration,
            intensity=intensity,
            state_after=state_after.to_dict(),
            reward=reward,
            reasoning=reasoning,
        )
        
        self.memory.add_experience(exp)
        
        # Periodic reflection
        if (self.enable_learning and 
            len(self.memory.experiences) >= 5 and 
            self.decision_count % self.reflection_interval == 0):
            self._reflect()
    
    def _compute_reward(
        self,
        state_before: PhoneState,
        activity: Activity,
        duration: float,
        state_after: PhoneState,
    ) -> float:
        """
        Compute reward based on persona values and outcome.
        
        Reward balances:
        - Satisfaction from activity
        - Battery conservation
        - Avoiding overheating
        """
        profile = ACTIVITY_PROFILES[activity]
        
        # Base satisfaction from activity
        satisfaction = profile['satisfaction'] * duration / 60
        
        # Battery cost (penalize heavy drain)
        battery_drain = state_before.soc - state_after.soc
        battery_penalty = battery_drain * self.persona.values_battery_life * 0.1
        
        # Temperature penalty
        temp_penalty = 0
        if state_after.temperature > 40:
            temp_penalty = (state_after.temperature - 40) * 0.05
        
        # Critical battery penalty
        critical_penalty = 0
        if state_after.soc < 20:
            critical_penalty = (20 - state_after.soc) * 0.02 * self.persona.values_battery_life
        
        # Persona-specific bonuses
        activity_bonus = 0
        if activity in [Activity.GAMING, Activity.VIDEO_STREAMING]:
            activity_bonus = self.persona.values_entertainment * 0.2
        elif activity in [Activity.WORK_APPS]:
            activity_bonus = self.persona.values_productivity * 0.2
        elif activity in [Activity.MESSAGING, Activity.SOCIAL_MEDIA]:
            activity_bonus = self.persona.values_social_connection * 0.2
        
        reward = satisfaction + activity_bonus - battery_penalty - temp_penalty - critical_penalty
        return np.clip(reward, -1, 1)
    
    def _reflect(self):
        """Use LLM to reflect on experiences and learn new strategies."""
        print(f"\n🔄 {self.persona.name} is reflecting on experiences...")
        
        # Build reflection prompt
        prompt = self.REFLECTION_PROMPT.format(
            name=self.persona.name,
            experiences=self._format_experiences(self.memory.get_recent(10)),
            personality=self.persona.to_prompt_string(),
            strategies=self._format_strategies(),
        )
        
        # Call LLM
        response = self._call_llm(prompt, temperature=0.5)
        data = self._parse_json(response)
        
        if data and 'new_strategy' in data:
            new_strategy = data['new_strategy']
            confidence = data.get('confidence', 0.5)
            
            # Only add high-confidence strategies
            if confidence >= 0.6 and new_strategy not in self.memory.learned_strategies:
                self.memory.learned_strategies.append(new_strategy)
                self.memory.reflection_count += 1
                print(f"   💡 New strategy learned: {new_strategy}")
                print(f"   📊 Confidence: {confidence:.0%}")
            else:
                print(f"   🤔 Insight (low confidence): {data.get('insight', 'N/A')}")
    
    def get_stats(self) -> Dict:
        """Get learning statistics."""
        return {
            'persona': self.persona.name,
            'decisions': self.decision_count,
            'api_calls': self.api_calls,
            'total_tokens': self.total_tokens,
            'memory': self.memory.get_stats(),
            'strategies': self.memory.learned_strategies,
        }


# ============================================================================
# SIMULATION WITH LEARNING
# ============================================================================

class LearningSimulator:
    """
    Simulate a day with a learning LLM persona.
    
    Tracks battery drain, temperature, and learning progress.
    """
    
    def __init__(
        self,
        persona: LearningLLMPersona,
        initial_soc: float = 100.0,
        ambient_temp: float = 25.0,
        battery_capacity_wh: float = 17.1,  # 4500mAh @ 3.8V
    ):
        self.persona = persona
        self.initial_soc = initial_soc
        self.ambient_temp = ambient_temp
        self.battery_capacity_wh = battery_capacity_wh
        
        self.timeline = []
    
    def _simulate_physics(
        self,
        activity: Activity,
        duration_minutes: float,
        intensity: float,
        current_soc: float,
        current_temp: float,
    ) -> Tuple[float, float]:
        """
        Simulate battery drain and temperature change.
        
        Returns:
            (new_soc, new_temperature)
        """
        profile = ACTIVITY_PROFILES[activity]
        
        # Power consumption (scaled by intensity)
        power_w = profile['power'] * intensity
        
        # Energy consumed
        energy_wh = power_w * duration_minutes / 60
        soc_drain = (energy_wh / self.battery_capacity_wh) * 100
        
        new_soc = max(0, current_soc - soc_drain)
        
        # Temperature model (simplified)
        heat_rate = profile['heat'] * intensity
        cooling_rate = 0.05  # Natural cooling towards ambient
        
        temp_change = heat_rate * duration_minutes - cooling_rate * (current_temp - self.ambient_temp)
        new_temp = np.clip(current_temp + temp_change, self.ambient_temp, 50)
        
        return new_soc, new_temp
    
    def simulate_day(
        self,
        start_hour: Optional[float] = None,
        end_hour: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Simulate a full day with the learning persona.
        
        Returns:
            Simulation results dictionary
        """
        start = start_hour or self.persona.persona.wake_hour
        end = end_hour or min(self.persona.persona.sleep_hour + 24 
                              if self.persona.persona.sleep_hour < self.persona.persona.wake_hour 
                              else self.persona.persona.sleep_hour, 24)
        
        current_time = start
        current_soc = self.initial_soc
        current_temp = self.ambient_temp + 3  # Slightly warm
        total_satisfaction = 0
        
        self.timeline = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🌅 Starting day simulation for {self.persona.persona.name}")
            print(f"   Time: {start:.0f}:00 to {end:.0f}:00")
            print(f"   Battery: {current_soc:.0f}%")
            print(f"{'='*60}\n")
        
        step = 0
        while current_time < end and current_soc > 0:
            step += 1
            
            # Current state
            state = PhoneState(
                soc=current_soc,
                temperature=current_temp,
                current_time=current_time,
                total_satisfaction=total_satisfaction,
            )
            
            # LLM decides
            activity, duration, intensity, reasoning = self.persona.decide_activity(state)
            
            # Simulate physics
            new_soc, new_temp = self._simulate_physics(
                activity, duration, intensity, current_soc, current_temp
            )
            
            # Create new state
            new_state = PhoneState(
                soc=new_soc,
                temperature=new_temp,
                current_time=current_time + duration/60,
                last_activity=activity.value,
                last_duration=duration,
            )
            
            # Record experience (this may trigger learning)
            self.persona.record_experience(
                state, activity, duration, intensity, new_state, reasoning
            )
            
            # Update satisfaction
            satisfaction = ACTIVITY_PROFILES[activity]['satisfaction'] * duration / 60
            total_satisfaction += satisfaction
            
            # Log
            if verbose:
                time_str = f"{int(current_time):02d}:{int((current_time % 1)*60):02d}"
                print(f"[{time_str}] {activity.value:15s} {duration:4.0f}min | "
                      f"SOC: {current_soc:5.1f}%→{new_soc:5.1f}% | "
                      f"Temp: {current_temp:4.1f}°C→{new_temp:4.1f}°C")
                if reasoning:
                    print(f"         └─ {reasoning[:60]}...")
            
            # Record timeline
            self.timeline.append({
                'step': step,
                'time': current_time,
                'activity': activity.value,
                'duration': duration,
                'intensity': intensity,
                'soc_before': current_soc,
                'soc_after': new_soc,
                'temp_before': current_temp,
                'temp_after': new_temp,
                'reasoning': reasoning,
            })
            
            # Update state
            current_time += duration / 60
            current_soc = new_soc
            current_temp = new_temp
            
            # Check for dead battery
            if current_soc <= 0:
                if verbose:
                    print(f"\n⚠️ Battery depleted at {current_time:.1f}:00!")
                break
        
        # Compile results
        results = self._compile_results(
            start, current_time, current_soc, total_satisfaction
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 Day Complete for {self.persona.persona.name}")
            print(f"   Battery life: {results['battery_life_hours']:.1f} hours")
            print(f"   Final SOC: {results['final_soc']:.1f}%")
            print(f"   Total satisfaction: {total_satisfaction:.1f}")
            print(f"   Strategies learned: {len(self.persona.memory.learned_strategies)}")
            print(f"   API calls: {self.persona.api_calls}")
            print(f"{'='*60}")
        
        return results
    
    def _compile_results(
        self,
        start_time: float,
        end_time: float,
        final_soc: float,
        total_satisfaction: float,
    ) -> Dict:
        """Compile simulation results."""
        # Activity breakdown
        activity_times = {}
        for entry in self.timeline:
            act = entry['activity']
            activity_times[act] = activity_times.get(act, 0) + entry['duration']
        
        return {
            'timeline': self.timeline,
            'activity_breakdown': activity_times,
            'battery_life_hours': end_time - start_time,
            'final_soc': final_soc,
            'total_satisfaction': total_satisfaction,
            'max_temperature': max(e['temp_after'] for e in self.timeline) if self.timeline else 0,
            'strategies_learned': self.persona.memory.learned_strategies.copy(),
            'learning_stats': self.persona.get_stats(),
        }


# ============================================================================
# MULTI-DAY LEARNING EXPERIMENT
# ============================================================================

def run_learning_experiment(
    persona_name: str,
    n_days: int = 5,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> Dict:
    """
    Run a multi-day learning experiment.
    
    The persona learns across multiple simulated days, discovering
    emergent battery-saving strategies through experience.
    
    Args:
        persona_name: Name from PERSONAS dict
        n_days: Number of days to simulate
        api_key: OpenAI API key
        model: Model to use
        verbose: Print progress
        
    Returns:
        Experiment results with learning progression
    """
    if persona_name not in PERSONAS:
        raise ValueError(f"Unknown persona: {persona_name}. Choose from: {list(PERSONAS.keys())}")
    
    persona_traits = PERSONAS[persona_name]
    
    # Create learning persona (persists across days)
    learning_persona = LearningLLMPersona(
        persona=persona_traits,
        api_key=api_key,
        model=model,
        enable_learning=True,
        reflection_interval=8,  # Reflect every 8 decisions
    )
    
    # Track results across days
    daily_results = []
    
    print(f"\n{'='*70}")
    print(f"🧪 LEARNING EXPERIMENT: {persona_traits.name}")
    print(f"   Simulating {n_days} days with continuous learning")
    print(f"   Model: {model}")
    print(f"{'='*70}")
    
    for day in range(1, n_days + 1):
        print(f"\n📅 DAY {day}/{n_days}")
        print("-" * 40)
        
        # Create simulator (fresh battery each day)
        simulator = LearningSimulator(
            persona=learning_persona,
            initial_soc=100.0,
        )
        
        # Run simulation
        results = simulator.simulate_day(verbose=verbose)
        results['day'] = day
        daily_results.append(results)
        
        # Brief pause to avoid rate limiting
        time.sleep(0.5)
    
    # Analyze learning progression
    battery_lives = [r['battery_life_hours'] for r in daily_results]
    satisfactions = [r['total_satisfaction'] for r in daily_results]
    
    print(f"\n{'='*70}")
    print(f"📈 LEARNING PROGRESSION SUMMARY")
    print(f"{'='*70}")
    print(f"\nBattery Life Over Days:")
    for i, bl in enumerate(battery_lives, 1):
        bar = '█' * int(bl * 2)
        print(f"  Day {i}: {bl:5.1f}h |{bar}")
    
    print(f"\nLearned Strategies:")
    for i, strategy in enumerate(learning_persona.memory.learned_strategies, 1):
        print(f"  {i}. {strategy}")
    
    print(f"\nFinal Stats:")
    print(f"  Total API calls: {learning_persona.api_calls}")
    print(f"  Total tokens used: {learning_persona.total_tokens:,}")
    print(f"  Reflections performed: {learning_persona.memory.reflection_count}")
    
    # Did battery life improve?
    if len(battery_lives) >= 2:
        improvement = battery_lives[-1] - battery_lives[0]
        print(f"\n  Battery life improvement: {improvement:+.1f}h "
              f"({battery_lives[0]:.1f}h → {battery_lives[-1]:.1f}h)")
    
    return {
        'persona': persona_name,
        'daily_results': daily_results,
        'strategies_learned': learning_persona.memory.learned_strategies,
        'final_stats': learning_persona.get_stats(),
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_learning_progression(results: Dict, save_path: Optional[str] = None):
    """
    Create visualization of learning progression.
    
    Shows:
    1. Battery life improvement over days
    2. Activity breakdown comparison
    3. Strategies learned timeline
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Learning Progression: {results['persona']}", fontsize=16, fontweight='bold')
    
    daily = results['daily_results']
    n_days = len(daily)
    
    # 1. Battery Life Improvement
    ax1 = fig.add_subplot(2, 2, 1)
    battery_lives = [d['battery_life_hours'] for d in daily]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_days))
    bars = ax1.bar(range(1, n_days + 1), battery_lives, color=colors, edgecolor='black')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Battery Life (hours)')
    ax1.set_title('Battery Life Improvement Over Days')
    ax1.set_xticks(range(1, n_days + 1))
    
    # Add value labels on bars
    for bar, val in zip(bars, battery_lives):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.1f}h', ha='center', va='bottom', fontsize=10)
    
    # Add improvement annotation
    if n_days >= 2:
        improvement = battery_lives[-1] - battery_lives[0]
        ax1.annotate(f'+{improvement:.1f}h\nimprovement!',
                    xy=(n_days, battery_lives[-1]),
                    xytext=(n_days - 0.5, battery_lives[-1] - 2),
                    fontsize=12, color='green', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    # 2. Satisfaction vs Battery Trade-off
    ax2 = fig.add_subplot(2, 2, 2)
    satisfactions = [d['total_satisfaction'] for d in daily]
    final_socs = [d['final_soc'] for d in daily]
    
    ax2.scatter(satisfactions, battery_lives, c=range(n_days), cmap='viridis', 
               s=200, edgecolor='black', zorder=3)
    for i, (sat, bl) in enumerate(zip(satisfactions, battery_lives)):
        ax2.annotate(f'Day {i+1}', (sat, bl), textcoords="offset points", 
                    xytext=(10, 5), fontsize=10)
    
    ax2.set_xlabel('Total Satisfaction')
    ax2.set_ylabel('Battery Life (hours)')
    ax2.set_title('Satisfaction vs Battery Life Trade-off')
    ax2.grid(True, alpha=0.3)
    
    # 3. Activity Breakdown Comparison (Day 1 vs Last Day)
    ax3 = fig.add_subplot(2, 2, 3)
    
    activity_colors = {
        'gaming': '#FF4136',
        'video_streaming': '#FF851B', 
        'video_recording': '#FFDC00',
        'photography': '#01FF70',
        'social_media': '#7FDBFF',
        'navigation': '#B10DC9',
        'messaging': '#0074D9',
        'voice_call': '#85144b',
        'work_apps': '#3D9970',
        'idle': '#AAAAAA',
        'screen_off': '#DDDDDD',
    }
    
    # Compare first and last day
    day1 = daily[0]['activity_breakdown']
    day_last = daily[-1]['activity_breakdown']
    
    all_activities = sorted(set(day1.keys()) | set(day_last.keys()))
    
    x = np.arange(len(all_activities))
    width = 0.35
    
    day1_vals = [day1.get(a, 0) for a in all_activities]
    day_last_vals = [day_last.get(a, 0) for a in all_activities]
    
    bars1 = ax3.bar(x - width/2, day1_vals, width, label=f'Day 1', 
                   color=[activity_colors.get(a, '#666') for a in all_activities], alpha=0.6)
    bars2 = ax3.bar(x + width/2, day_last_vals, width, label=f'Day {n_days}',
                   color=[activity_colors.get(a, '#666') for a in all_activities], edgecolor='black')
    
    ax3.set_xlabel('Activity')
    ax3.set_ylabel('Time (minutes)')
    ax3.set_title(f'Activity Breakdown: Day 1 vs Day {n_days}')
    ax3.set_xticks(x)
    ax3.set_xticklabels([a.replace('_', '\n') for a in all_activities], fontsize=8, rotation=0)
    ax3.legend()
    
    # 4. Learned Strategies
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    strategies = results.get('strategies_learned', [])
    text = "🧠 EMERGENT STRATEGIES LEARNED:\n\n"
    for i, s in enumerate(strategies, 1):
        # Wrap text
        wrapped = s[:60] + '...' if len(s) > 60 else s
        text += f"  {i}. {wrapped}\n\n"
    
    if not strategies:
        text += "  No strategies learned yet.\n"
    
    stats = results.get('final_stats', {})
    text += f"\n📊 LEARNING STATS:\n"
    text += f"  • API calls: {stats.get('api_calls', 0)}\n"
    text += f"  • Tokens used: {stats.get('total_tokens', 0):,}\n"
    text += f"  • Reflections: {stats.get('memory', {}).get('strategies_learned', 0)}\n"
    
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    ax4.set_title('Learned Strategies & Stats')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Figure saved to: {save_path}")
    
    plt.show()
    return fig


# ============================================================================
# MAIN - DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           LEARNING LLM PERSONA SYSTEM - MCM 2026                     ║
║                                                                       ║
║  This system uses GPT-4 to create personas that ACTUALLY LEARN       ║
║  battery-saving strategies through experience!                        ║
║                                                                       ║
║  Set your API key:                                                    ║
║    export OPENAI_API_KEY='your-key-here'                             ║
║                                                                       ║
║  Or pass it directly to the functions.                               ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ No API key found!")
        print("   Set OPENAI_API_KEY environment variable or pass api_key parameter")
        print("\n   Example:")
        print("   export OPENAI_API_KEY='sk-...'")
        print("   python learning_llm_persona.py")
        exit(1)
    
    print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Run experiment with one persona
    print("\n" + "="*70)
    print("Starting learning experiment with 'balanced_ben'")
    print("This will make ~50 API calls over 3 simulated days")
    print("="*70)
    
    results = run_learning_experiment(
        persona_name='balanced_ben',
        n_days=3,
        api_key=api_key,
        model='gpt-4o-mini',  # Fast and cheap
        verbose=True,
    )
    
    # Save results
    output_file = '/home/funt1kk/zjui/SPRING26/IM2C/Problem_A/figures/learning_experiment_results.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print("\n🎉 Learning experiment complete!")
