"""
Reinforcement Learning Training for Persona Agents
===================================================

Novel contribution for MCM 2026: Train LLM persona agents to discover
emergent battery-usage behaviors through reinforcement learning.

Key innovation: Agents learn to balance:
- Activity satisfaction (doing what they want)
- Battery anxiety (worry about low SOC)
- Thermal discomfort (phone getting hot)

This reveals behaviors like:
- "Anxiety spiral" - checking phone MORE when battery is low
- "Thermal avoidance" - stopping gaming when hot
- "Batch optimization" - grouping activities smartly

Uses Contextual Bandit approach (simpler than full RL, still publishable).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
import random
import json
from pathlib import Path

from llm_persona import (
    Activity, ActivityDecision, PersonaTraits, PhoneState,
    PersonaAgent, PERSONAS
)

# Activity to workload mapping (GPU-related)
ACTIVITY_WORKLOAD_MAP = {
    'gaming': 'rendering',
    'video_streaming': 'general',
    'social_media': 'general',
    'messaging': 'idle',
    'browsing': 'general',
    'photo_editing': 'ai',
    'video_creation': 'rendering',
    'music': 'idle',
    'navigation': 'compute',
    'idle': 'idle',
}


# ============================================================================
# REWARD FUNCTION
# ============================================================================

@dataclass
class RewardWeights:
    """Weights for the multi-objective reward function."""
    satisfaction: float = 1.0      # Weight for activity satisfaction
    battery_cost: float = 0.3      # Weight for battery drain penalty
    thermal_penalty: float = 0.1   # Weight for thermal discomfort
    anxiety_bonus: float = 0.2     # Bonus for managing low-battery anxiety
    
    # Activity-specific satisfaction multipliers
    activity_satisfaction: Dict[str, float] = field(default_factory=lambda: {
        'gaming': 1.5,
        'video_streaming': 1.2,
        'social_media': 1.1,
        'messaging': 1.0,
        'photography': 1.3,
        'video_recording': 1.4,
        'navigation': 0.8,  # Necessary but not "fun"
        'work_apps': 0.7,   # Work is work
        'voice_call': 0.9,
        'idle': 0.3,
        'screen_off': 0.1,
        'charging': 0.5,
    })


def compute_activity_satisfaction(
    activity: Activity,
    duration: float,
    intensity: float,
    persona: PersonaTraits,
    weights: RewardWeights,
) -> float:
    """
    Compute satisfaction from an activity.
    
    Satisfaction depends on:
    - Activity type (gaming more satisfying than work for gamers)
    - Duration (diminishing returns after ~30 min)
    - Intensity (higher intensity = more engagement)
    - Persona preferences
    """
    base_satisfaction = weights.activity_satisfaction.get(activity.value, 0.5)
    
    # Persona affinity boost
    if activity == Activity.GAMING:
        base_satisfaction *= (1 + persona.gaming_affinity * 0.5)
    elif activity == Activity.SOCIAL_MEDIA:
        base_satisfaction *= (1 + persona.social_media_affinity * 0.5)
    elif activity == Activity.WORK_APPS:
        base_satisfaction *= (1 + persona.work_focus * 0.3)
    
    # Preferred activities get bonus
    if activity.value in persona.preferred_activities:
        base_satisfaction *= 1.3
    
    # Avoided activities get penalty
    if activity.value in persona.avoided_activities:
        base_satisfaction *= 0.5
    
    # Duration scaling (diminishing returns)
    # Peak satisfaction around 30 minutes, then diminishing
    duration_factor = 1 - np.exp(-duration / 15)  # Quick ramp-up
    duration_factor *= np.exp(-max(0, duration - 30) / 60)  # Slow decay after 30 min
    
    # Intensity boost
    intensity_factor = 0.5 + 0.5 * intensity
    
    return base_satisfaction * duration_factor * intensity_factor * duration


def compute_battery_anxiety(
    soc: float,
    persona: PersonaTraits,
) -> float:
    """
    Compute battery anxiety factor.
    
    Users become irrational below 30% battery:
    - Check phone more frequently
    - Experience stress/discomfort
    
    This is the KEY insight: low battery → MORE usage → faster drain
    """
    if soc > 50:
        return 0.0
    elif soc > 30:
        # Mild concern
        return persona.battery_anxiety * (50 - soc) / 40
    else:
        # Exponential anxiety below 30%
        base_anxiety = (30 - soc) / 30
        return persona.battery_anxiety * (1 + np.exp(base_anxiety * 2) - 1)


def compute_thermal_discomfort(
    temperature: float,
    persona: PersonaTraits,
) -> float:
    """
    Compute discomfort from hot phone.
    
    Above 40°C, users notice and may change behavior.
    Above 45°C, significant discomfort.
    """
    if temperature < 38:
        return 0.0
    elif temperature < 42:
        return persona.thermal_sensitivity * (temperature - 38) / 8
    else:
        # Strong discomfort above 42°C
        return persona.thermal_sensitivity * (1 + (temperature - 42) / 5)


def compute_reward(
    state_before: PhoneState,
    action: ActivityDecision,
    state_after: PhoneState,
    persona: PersonaTraits,
    weights: RewardWeights = None,
) -> Tuple[float, Dict]:
    """
    Compute reward for a state-action-state transition.
    
    R = w_sat * Satisfaction 
        - w_bat * Battery_Cost * (1 + Anxiety_Factor)
        - w_th * Thermal_Penalty
    
    Args:
        state_before: Phone state before action
        action: Activity decision taken
        state_after: Phone state after action
        persona: Persona traits
        weights: Reward weights
        
    Returns:
        Tuple of (total_reward, component_breakdown)
    """
    if weights is None:
        weights = RewardWeights()
    
    # Satisfaction component
    satisfaction = compute_activity_satisfaction(
        action.activity,
        action.duration_minutes,
        action.intensity,
        persona,
        weights,
    )
    
    # Battery cost
    delta_soc = state_before.soc - state_after.soc
    anxiety = compute_battery_anxiety(state_after.soc, persona)
    battery_cost = delta_soc * (1 + anxiety)
    
    # If charging, battery cost becomes negative (reward)
    if action.activity == Activity.CHARGING:
        battery_cost = -abs(delta_soc) * 0.5  # Reward for charging
    
    # Thermal penalty
    thermal_penalty = compute_thermal_discomfort(state_after.temperature, persona)
    
    # Total reward
    reward = (
        weights.satisfaction * satisfaction
        - weights.battery_cost * battery_cost
        - weights.thermal_penalty * thermal_penalty
    )
    
    breakdown = {
        'satisfaction': satisfaction,
        'battery_cost': battery_cost,
        'anxiety_factor': anxiety,
        'thermal_penalty': thermal_penalty,
        'delta_soc': delta_soc,
        'total_reward': reward,
    }
    
    return reward, breakdown


# ============================================================================
# CONTEXTUAL BANDIT AGENT
# ============================================================================

@dataclass
class Context:
    """Context features for the bandit."""
    hour_sin: float  # sin(2π * hour / 24)
    hour_cos: float  # cos(2π * hour / 24)
    soc_normalized: float  # SOC / 100
    temp_normalized: float  # (temp - 25) / 25
    is_commute: float  # 1 if commute hour, 0 otherwise
    is_work_hours: float  # 1 if 9-17, 0 otherwise
    last_activity_idx: int  # Index of last activity
    
    @classmethod
    def from_state(cls, state: PhoneState, persona: PersonaTraits) -> 'Context':
        hour = state.current_time
        return cls(
            hour_sin=np.sin(2 * np.pi * hour / 24),
            hour_cos=np.cos(2 * np.pi * hour / 24),
            soc_normalized=state.soc / 100,
            temp_normalized=(state.temperature - 25) / 25,
            is_commute=1.0 if int(hour) in persona.commute_hours else 0.0,
            is_work_hours=1.0 if 9 <= hour <= 17 else 0.0,
            last_activity_idx=list(Activity).index(state.last_activity) if state.last_activity else 0,
        )
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.hour_sin,
            self.hour_cos,
            self.soc_normalized,
            self.temp_normalized,
            self.is_commute,
            self.is_work_hours,
            self.last_activity_idx / len(Activity),
        ])


class ContextualBanditAgent:
    """
    Contextual bandit agent using Thompson Sampling.
    
    Learns which activities are best in different contexts
    (time of day, battery level, temperature).
    
    Simpler than full RL but captures the key dynamics.
    """
    
    def __init__(
        self,
        persona: PersonaTraits,
        n_arms: int = len(Activity),
        context_dim: int = 7,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
    ):
        """
        Initialize bandit agent.
        
        Args:
            persona: Persona traits
            n_arms: Number of actions (activities)
            context_dim: Dimensionality of context features
            prior_mean: Prior mean for reward
            prior_std: Prior std for reward
        """
        self.persona = persona
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.activities = list(Activity)
        
        # Thompson Sampling parameters (Bayesian linear regression)
        self.B = [np.eye(context_dim) for _ in range(n_arms)]  # Precision matrices
        self.mu = [np.zeros(context_dim) for _ in range(n_arms)]  # Mean vectors
        self.f = [np.zeros(context_dim) for _ in range(n_arms)]  # Accumulated features
        
        # Hyperparameters
        self.lambda_prior = 1.0  # Regularization
        self.sigma = 1.0  # Noise std
        
        # Training history
        self.history: List[Dict] = []
        self.cumulative_reward = 0.0
    
    def select_action(
        self,
        state: PhoneState,
        exploration_rate: float = 0.1,
    ) -> Tuple[Activity, int]:
        """
        Select action using Thompson Sampling.
        
        Args:
            state: Current phone state
            exploration_rate: Probability of random exploration
            
        Returns:
            Tuple of (selected_activity, arm_index)
        """
        context = Context.from_state(state, self.persona)
        x = context.to_vector()
        
        # Exploration
        if random.random() < exploration_rate:
            arm = random.randint(0, self.n_arms - 1)
            return self.activities[arm], arm
        
        # Thompson Sampling: sample from posterior and pick best
        sampled_rewards = []
        for arm in range(self.n_arms):
            # Sample weight vector from posterior
            try:
                B_inv = np.linalg.inv(self.B[arm])
                w_sample = np.random.multivariate_normal(self.mu[arm], self.sigma**2 * B_inv)
            except np.linalg.LinAlgError:
                w_sample = self.mu[arm]
            
            # Predicted reward
            reward = np.dot(w_sample, x)
            sampled_rewards.append(reward)
        
        # Select best arm
        best_arm = np.argmax(sampled_rewards)
        return self.activities[best_arm], best_arm
    
    def update(
        self,
        state: PhoneState,
        arm: int,
        reward: float,
    ):
        """
        Update posterior after observing reward.
        
        Args:
            state: State where action was taken
            arm: Index of action taken
            reward: Observed reward
        """
        context = Context.from_state(state, self.persona)
        x = context.to_vector()
        
        # Bayesian update
        self.B[arm] += np.outer(x, x)
        self.f[arm] += reward * x
        
        try:
            self.mu[arm] = np.linalg.solve(self.B[arm], self.f[arm])
        except np.linalg.LinAlgError:
            pass  # Keep previous mu if singular
        
        self.cumulative_reward += reward
    
    def get_policy_summary(self) -> Dict[str, Dict]:
        """
        Get summary of learned policy.
        
        Returns which activities the agent prefers in different contexts.
        """
        summary = {}
        
        # Test different contexts
        test_contexts = [
            ("Morning (8am), full battery", 8.0, 90.0, 30.0),
            ("Afternoon (14pm), half battery", 14.0, 50.0, 35.0),
            ("Evening (20pm), low battery", 20.0, 25.0, 38.0),
            ("Night (23pm), critical battery", 23.0, 10.0, 32.0),
        ]
        
        for name, hour, soc, temp in test_contexts:
            state = PhoneState(
                soc=soc,
                temperature=temp,
                is_charging=False,
                screen_on=True,
                current_time=hour,
            )
            context = Context.from_state(state, self.persona)
            x = context.to_vector()
            
            # Get expected rewards for each action
            expected_rewards = {}
            for arm, activity in enumerate(self.activities):
                expected_rewards[activity.value] = np.dot(self.mu[arm], x)
            
            # Sort by expected reward
            sorted_activities = sorted(
                expected_rewards.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            summary[name] = {
                'top_3': sorted_activities[:3],
                'context': {'hour': hour, 'soc': soc, 'temp': temp},
            }
        
        return summary


# ============================================================================
# TRAINING LOOP
# ============================================================================

class RLTrainer:
    """
    Trainer for RL persona agents.
    
    Runs episodes, computes rewards, updates agent.
    """
    
    def __init__(
        self,
        persona_name: str,
        reward_weights: RewardWeights = None,
    ):
        """
        Initialize trainer.
        
        Args:
            persona_name: Name from PERSONAS dict
            reward_weights: Custom reward weights
        """
        if persona_name not in PERSONAS:
            raise ValueError(f"Unknown persona: {persona_name}")
        
        self.persona = PERSONAS[persona_name]
        self.persona_name = persona_name
        self.weights = reward_weights or RewardWeights()
        
        # Create agent
        self.agent = ContextualBanditAgent(self.persona)
        
        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_battery_lives: List[float] = []
        self.discovered_behaviors: List[Dict] = []
    
    def run_episode(
        self,
        initial_soc: float = 100.0,
        exploration_rate: float = 0.1,
    ) -> Dict:
        """
        Run a single training episode (one day).
        
        Args:
            initial_soc: Starting battery percentage
            exploration_rate: Exploration probability
            
        Returns:
            Episode statistics
        """
        current_time = float(self.persona.wake_hour)
        current_soc = initial_soc
        current_temp = 30.0
        
        state = PhoneState(
            soc=current_soc,
            temperature=current_temp,
            is_charging=False,
            screen_on=True,
            current_time=current_time,
        )
        
        episode_reward = 0.0
        timeline = []
        
        while current_time < 24.0 and current_soc > 0:
            # Select action
            activity, arm = self.agent.select_action(state, exploration_rate)
            
            # Generate duration and intensity (could also be learned)
            duration = self._sample_duration(activity)
            intensity = self._sample_intensity(activity)
            
            action = ActivityDecision(
                activity=activity,
                duration_minutes=duration,
                intensity=intensity,
            )
            
            # Simulate transition
            delta_soc, new_temp = self._simulate_transition(action, current_temp)
            
            # New state
            new_soc = max(0, current_soc - delta_soc)
            new_time = current_time + duration / 60
            
            state_after = PhoneState(
                soc=new_soc,
                temperature=new_temp,
                is_charging=activity == Activity.CHARGING,
                screen_on=activity != Activity.SCREEN_OFF,
                current_time=new_time,
                last_activity=activity,
                last_activity_duration=duration,
            )
            
            # Compute reward
            reward, breakdown = compute_reward(
                state, action, state_after, self.persona, self.weights
            )
            
            # Update agent
            self.agent.update(state, arm, reward)
            
            # Record
            timeline.append({
                'time': current_time,
                'activity': activity.value,
                'duration': duration,
                'intensity': intensity,
                'soc': current_soc,
                'temperature': current_temp,
                'reward': reward,
                'breakdown': breakdown,
            })
            
            episode_reward += reward
            
            # Advance state
            state = state_after
            current_soc = new_soc
            current_temp = new_temp
            current_time = new_time
        
        # Episode statistics
        battery_life = current_time - self.persona.wake_hour
        
        self.episode_rewards.append(episode_reward)
        self.episode_battery_lives.append(battery_life)
        
        return {
            'total_reward': episode_reward,
            'battery_life_hours': battery_life,
            'final_soc': current_soc,
            'n_activities': len(timeline),
            'timeline': timeline,
        }
    
    def _sample_duration(self, activity: Activity) -> float:
        """Sample activity duration."""
        base_durations = {
            Activity.GAMING: 40,
            Activity.VIDEO_STREAMING: 25,
            Activity.VIDEO_RECORDING: 5,
            Activity.PHOTOGRAPHY: 3,
            Activity.SOCIAL_MEDIA: 12,
            Activity.NAVIGATION: 20,
            Activity.MESSAGING: 5,
            Activity.VOICE_CALL: 8,
            Activity.WORK_APPS: 15,
            Activity.IDLE: 10,
            Activity.SCREEN_OFF: 45,
            Activity.CHARGING: 30,
        }
        base = base_durations.get(activity, 10)
        return base * np.random.lognormal(0, 0.3)
    
    def _sample_intensity(self, activity: Activity) -> float:
        """Sample activity intensity."""
        if activity in [Activity.GAMING, Activity.VIDEO_RECORDING]:
            return np.clip(np.random.normal(0.8, 0.15), 0.3, 1.0)
        else:
            return np.clip(np.random.normal(0.5, 0.2), 0.2, 0.9)
    
    def _simulate_transition(
        self,
        action: ActivityDecision,
        current_temp: float,
    ) -> Tuple[float, float]:
        """Simulate state transition."""
        # Power consumption (% per minute)
        power_rates = {
            Activity.GAMING: 0.7,
            Activity.VIDEO_STREAMING: 0.25,
            Activity.VIDEO_RECORDING: 0.5,
            Activity.PHOTOGRAPHY: 0.35,
            Activity.SOCIAL_MEDIA: 0.2,
            Activity.NAVIGATION: 0.3,
            Activity.MESSAGING: 0.12,
            Activity.VOICE_CALL: 0.15,
            Activity.WORK_APPS: 0.18,
            Activity.IDLE: 0.04,
            Activity.SCREEN_OFF: 0.015,
            Activity.CHARGING: -1.2,
        }
        
        rate = power_rates.get(action.activity, 0.1)
        delta_soc = rate * action.duration_minutes * action.intensity
        
        # Temperature
        heat_gen = {
            Activity.GAMING: 0.25,
            Activity.VIDEO_RECORDING: 0.15,
            Activity.NAVIGATION: 0.1,
        }.get(action.activity, 0.03)
        
        ambient = 25.0
        new_temp = current_temp + heat_gen * action.intensity * action.duration_minutes
        new_temp -= 0.05 * action.duration_minutes * (current_temp - ambient) / 15
        new_temp = np.clip(new_temp, ambient, 48)
        
        return delta_soc, new_temp
    
    def train(
        self,
        n_episodes: int = 500,
        exploration_decay: float = 0.995,
        initial_exploration: float = 0.3,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the agent for multiple episodes.
        
        Args:
            n_episodes: Number of training episodes
            exploration_decay: Decay rate for exploration
            initial_exploration: Starting exploration rate
            verbose: Print progress
            
        Returns:
            Training statistics
        """
        exploration_rate = initial_exploration
        
        if verbose:
            print(f"Training {self.persona_name} for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            result = self.run_episode(
                initial_soc=100.0,
                exploration_rate=exploration_rate,
            )
            
            # Decay exploration
            exploration_rate *= exploration_decay
            exploration_rate = max(exploration_rate, 0.05)
            
            # Progress
            if verbose and (episode + 1) % 100 == 0:
                recent_rewards = self.episode_rewards[-100:]
                recent_battery = self.episode_battery_lives[-100:]
                print(f"  Episode {episode + 1}: "
                      f"Avg Reward = {np.mean(recent_rewards):.1f}, "
                      f"Avg Battery Life = {np.mean(recent_battery):.1f}h, "
                      f"ε = {exploration_rate:.3f}")
        
        # Analyze learned policy
        policy_summary = self.agent.get_policy_summary()
        
        # Detect emergent behaviors
        behaviors = self._detect_emergent_behaviors()
        
        return {
            'n_episodes': n_episodes,
            'final_exploration': exploration_rate,
            'mean_reward': np.mean(self.episode_rewards[-100:]),
            'mean_battery_life': np.mean(self.episode_battery_lives[-100:]),
            'policy_summary': policy_summary,
            'emergent_behaviors': behaviors,
            'reward_history': self.episode_rewards,
            'battery_life_history': self.episode_battery_lives,
        }
    
    def _detect_emergent_behaviors(self) -> List[Dict]:
        """
        Analyze learned policy to detect emergent behaviors.
        """
        behaviors = []
        summary = self.agent.get_policy_summary()
        
        # Check for anxiety spiral (more activity at low battery)
        low_battery_context = summary.get("Evening (20pm), low battery", {})
        critical_battery_context = summary.get("Night (23pm), critical battery", {})
        
        low_top = low_battery_context.get('top_3', [])
        critical_top = critical_battery_context.get('top_3', [])
        
        # If messaging/social_media rank higher at low battery → anxiety spiral
        low_battery_social = any(
            act in ['messaging', 'social_media'] 
            for act, _ in low_top[:2]
        )
        
        if low_battery_social and self.persona.battery_anxiety > 0.5:
            behaviors.append({
                'name': 'Battery Anxiety Spiral',
                'description': 'Agent increases phone checking when battery is low, '
                               'causing faster drain - a feedback loop.',
                'evidence': f"Top activities at 25% SOC: {[a for a, _ in low_top[:3]]}",
            })
        
        # Check for thermal avoidance
        morning_context = summary.get("Morning (8am), full battery", {})
        morning_top = morning_context.get('top_3', [])
        
        if 'gaming' in [a for a, _ in morning_top[:2]]:
            # Gaming preferred in morning (cool phone)
            evening_context = summary.get("Evening (20pm), low battery", {})
            evening_top = evening_context.get('top_3', [])
            
            if 'gaming' not in [a for a, _ in evening_top[:3]]:
                behaviors.append({
                    'name': 'Thermal Avoidance',
                    'description': 'Agent avoids heavy gaming when phone is warm, '
                                   'preferring lighter activities.',
                    'evidence': f"Gaming preferred in morning, avoided in evening",
                })
        
        return behaviors


# ============================================================================
# BATCH TRAINING AND ANALYSIS
# ============================================================================

def train_all_personas(
    n_episodes: int = 500,
    save_results: bool = True,
) -> Dict[str, Dict]:
    """
    Train all personas and compare results.
    
    Args:
        n_episodes: Episodes per persona
        save_results: Save results to file
        
    Returns:
        Results for all personas
    """
    results = {}
    
    for persona_name in PERSONAS:
        print(f"\n{'='*60}")
        print(f"Training: {persona_name}")
        print("=" * 60)
        
        trainer = RLTrainer(persona_name)
        result = trainer.train(n_episodes=n_episodes, verbose=True)
        results[persona_name] = result
        
        # Print summary
        print(f"\nResults for {persona_name}:")
        print(f"  Mean Reward: {result['mean_reward']:.1f}")
        print(f"  Mean Battery Life: {result['mean_battery_life']:.1f}h")
        print(f"\n  Emergent Behaviors Detected:")
        for behavior in result['emergent_behaviors']:
            print(f"    - {behavior['name']}: {behavior['description']}")
    
    if save_results:
        # Save to JSON (excluding numpy arrays)
        save_path = Path(__file__).parent.parent / 'data' / 'rl_training_results.json'
        
        serializable_results = {}
        for name, res in results.items():
            serializable_results[name] = {
                'n_episodes': res['n_episodes'],
                'mean_reward': float(res['mean_reward']),
                'mean_battery_life': float(res['mean_battery_life']),
                'emergent_behaviors': res['emergent_behaviors'],
            }
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {save_path}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("REINFORCEMENT LEARNING PERSONA TRAINING")
    print("=" * 60)
    
    # Train single persona for demonstration
    trainer = RLTrainer('gamer_gary')
    result = trainer.train(n_episodes=300, verbose=True)
    
    print("\n" + "=" * 60)
    print("LEARNED POLICY SUMMARY")
    print("=" * 60)
    
    for context_name, data in result['policy_summary'].items():
        print(f"\n{context_name}:")
        print(f"  Context: SOC={data['context']['soc']}%, "
              f"Temp={data['context']['temp']}°C, "
              f"Hour={data['context']['hour']}")
        print(f"  Top 3 activities:")
        for act, score in data['top_3'][:3]:
            print(f"    {act}: {score:.2f}")
    
    print("\n" + "=" * 60)
    print("EMERGENT BEHAVIORS")
    print("=" * 60)
    
    for behavior in result['emergent_behaviors']:
        print(f"\n{behavior['name']}:")
        print(f"  {behavior['description']}")
        print(f"  Evidence: {behavior['evidence']}")
    
    # Uncomment to train all personas
    # all_results = train_all_personas(n_episodes=500)
