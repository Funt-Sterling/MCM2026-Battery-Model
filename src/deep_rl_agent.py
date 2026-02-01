"""
=============================================================================
DEEP REINFORCEMENT LEARNING AGENT WITH LLM INTEGRATION
=============================================================================
A REAL learning system that:
1. Uses GPT-4 for intelligent decision reasoning
2. Trains a neural network to predict optimal actions
3. Learns from experience and IMPROVES over time
4. Runs on GPU (RTX 5070 Ti)

This is the NOVEL contribution - LLM-guided Deep RL for user behavior modeling.

Authors: MCM 2026 Team
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
import random
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
from pathlib import Path
import time

# Check for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# ACTIVITY DEFINITIONS
# =============================================================================

class Activity(Enum):
    GAMING = 0
    VIDEO_STREAMING = 1
    VIDEO_RECORDING = 2
    PHOTOGRAPHY = 3
    SOCIAL_MEDIA = 4
    NAVIGATION = 5
    MESSAGING = 6
    VOICE_CALL = 7
    WORK_APPS = 8
    IDLE = 9
    CHARGING = 10
    SCREEN_OFF = 11

NUM_ACTIONS = len(Activity)
ACTIVITY_NAMES = [a.name.lower() for a in Activity]


# =============================================================================
# EXPERIENCE REPLAY BUFFER
# =============================================================================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Experience(*args))
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


# =============================================================================
# NEURAL NETWORK (Deep Q-Network)
# =============================================================================

class DQN(nn.Module):
    """
    Deep Q-Network for action value estimation.
    
    Input: State vector (SOC, temp, time, persona traits, etc.)
    Output: Q-values for each action
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN - separates value and advantage streams.
    Better for learning state values vs action advantages.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Value stream (how good is this state?)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream (how good is each action relative to others?)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


# =============================================================================
# OPENAI LLM INTEGRATION
# =============================================================================

class GPT4Advisor:
    """
    GPT-4 provides reasoning and guidance for the RL agent.
    
    The LLM doesn't make decisions directly - it provides:
    1. Reasoning about current situation
    2. Suggested action with explanation
    3. Learning signal adjustments
    """
    
    SYSTEM_PROMPT = """You are simulating a smartphone user's decision-making process.
You will be given the current phone state and user personality.
Analyze the situation and suggest the most realistic action.

Output ONLY valid JSON:
{
    "reasoning": "Brief analysis of the situation",
    "suggested_action": "activity_name",
    "confidence": 0.0-1.0,
    "energy_priority": 0.0-1.0,
    "satisfaction_priority": 0.0-1.0
}

Available activities: gaming, video_streaming, video_recording, photography,
social_media, navigation, messaging, voice_call, work_apps, idle, charging, screen_off
"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self._client = None
        self.call_count = 0
        self.cache = {}  # Cache responses to save API calls
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        return self._client
    
    def get_advice(self, state_dict: Dict, persona_dict: Dict) -> Dict:
        """Get LLM advice for current situation."""
        
        # Create cache key
        cache_key = json.dumps({
            'soc_bucket': int(state_dict['soc'] // 10) * 10,
            'temp_bucket': int(state_dict['temperature'] // 5) * 5,
            'hour': int(state_dict['time']),
            'persona': persona_dict['name'],
        }, sort_keys=True)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        prompt = f"""Current phone state:
- Battery: {state_dict['soc']:.0f}%
- Temperature: {state_dict['temperature']:.1f}°C
- Time: {int(state_dict['time'])}:{int((state_dict['time'] % 1) * 60):02d}
- Charging: {state_dict.get('charging', False)}
- Last activity: {state_dict.get('last_activity', 'none')}

User personality:
- Name: {persona_dict['name']}
- Occupation: {persona_dict['occupation']}
- Gaming affinity: {persona_dict['gaming_affinity']}/10
- Social media: {persona_dict['social_media']}/10
- Battery anxiety: {persona_dict['battery_anxiety']}/10
- Work focus: {persona_dict['work_focus']}/10

What would this user do next?"""
        
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200,
            )
            self.call_count += 1
            
            # Parse response
            content = response.choices[0].message.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                result = json.loads(content[start:end])
                self.cache[cache_key] = result
                return result
        except Exception as e:
            print(f"GPT-4 error: {e}")
        
        # Fallback
        return {
            "reasoning": "API error, using default",
            "suggested_action": "idle",
            "confidence": 0.5,
            "energy_priority": 0.5,
            "satisfaction_priority": 0.5,
        }
    
    def get_action_index(self, action_name: str) -> int:
        """Convert action name to index."""
        action_name = action_name.lower().replace(" ", "_")
        try:
            return ACTIVITY_NAMES.index(action_name)
        except ValueError:
            return Activity.IDLE.value


# =============================================================================
# ENVIRONMENT
# =============================================================================

@dataclass
class PhoneEnvironment:
    """
    Smartphone battery environment for RL training.
    
    State: [soc, temperature, time_sin, time_cos, charging, 
            persona_gaming, persona_social, persona_work, persona_anxiety]
    """
    
    # Persona traits (normalized 0-1)
    gaming_affinity: float = 0.5
    social_media_affinity: float = 0.5
    work_focus: float = 0.5
    battery_anxiety: float = 0.5
    
    # Power consumption rates (% per minute)
    POWER_RATES = {
        Activity.GAMING: 0.7,
        Activity.VIDEO_STREAMING: 0.3,
        Activity.VIDEO_RECORDING: 0.5,
        Activity.PHOTOGRAPHY: 0.35,
        Activity.SOCIAL_MEDIA: 0.2,
        Activity.NAVIGATION: 0.3,
        Activity.MESSAGING: 0.12,
        Activity.VOICE_CALL: 0.18,
        Activity.WORK_APPS: 0.15,
        Activity.IDLE: 0.03,
        Activity.CHARGING: -1.0,
        Activity.SCREEN_OFF: 0.01,
    }
    
    # Heat generation rates (°C per minute)
    HEAT_RATES = {
        Activity.GAMING: 0.25,
        Activity.VIDEO_RECORDING: 0.18,
        Activity.NAVIGATION: 0.12,
        Activity.VIDEO_STREAMING: 0.08,
        Activity.CHARGING: 0.1,
    }
    
    def __init__(self, persona_name: str = "default"):
        self.reset()
        self._set_persona(persona_name)
    
    def _set_persona(self, name: str):
        personas = {
            'gamer': (0.95, 0.6, 0.3, 0.3),
            'creator': (0.3, 0.9, 0.7, 0.6),
            'commuter': (0.2, 0.4, 0.85, 0.8),
            'chatter': (0.4, 0.95, 0.4, 0.5),
            'streamer': (0.5, 0.5, 0.6, 0.7),
            'default': (0.5, 0.5, 0.5, 0.5),
        }
        traits = personas.get(name, personas['default'])
        self.gaming_affinity = traits[0]
        self.social_media_affinity = traits[1]
        self.work_focus = traits[2]
        self.battery_anxiety = traits[3]
        self.persona_name = name
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.soc = 100.0
        self.temperature = 25.0 + np.random.uniform(-2, 5)
        self.time = np.random.uniform(7, 10)  # Start morning
        self.charging = False
        self.last_action = Activity.IDLE
        self.total_reward = 0
        self.steps = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state as numpy array."""
        # Encode time as sin/cos for cyclical nature
        time_sin = np.sin(2 * np.pi * self.time / 24)
        time_cos = np.cos(2 * np.pi * self.time / 24)
        
        state = np.array([
            self.soc / 100.0,  # Normalize to 0-1
            self.temperature / 50.0,  # Normalize
            time_sin,
            time_cos,
            float(self.charging),
            self.gaming_affinity,
            self.social_media_affinity,
            self.work_focus,
            self.battery_anxiety,
        ], dtype=np.float32)
        
        return state
    
    def step(self, action: int, duration_minutes: float = 15.0) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info).
        """
        activity = Activity(action)
        
        # Battery drain
        drain_rate = self.POWER_RATES.get(activity, 0.1)
        drain = drain_rate * duration_minutes
        
        # Temperature change
        ambient = 25.0
        heat_rate = self.HEAT_RATES.get(activity, 0.02)
        heat = heat_rate * duration_minutes
        cooling = 0.05 * duration_minutes * (self.temperature - ambient) / 10
        
        # Update state
        old_soc = self.soc
        self.soc = np.clip(self.soc - drain, 0, 100)
        self.temperature = np.clip(self.temperature + heat - cooling, ambient, 55)
        self.time = (self.time + duration_minutes / 60) % 24
        self.charging = (activity == Activity.CHARGING)
        self.last_action = activity
        self.steps += 1
        
        # Calculate reward
        reward = self._calculate_reward(activity, old_soc, duration_minutes)
        self.total_reward += reward
        
        # Check if done
        done = (self.soc <= 0) or (self.steps >= 96)  # 96 steps = 24 hours at 15min
        
        info = {
            'activity': activity.name,
            'drain': drain,
            'soc': self.soc,
            'temperature': self.temperature,
            'time': self.time,
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self, activity: Activity, old_soc: float, duration: float) -> float:
        """
        Multi-objective reward function.
        
        Reward = Satisfaction - Battery_Cost - Thermal_Penalty + Context_Bonus
        """
        # Satisfaction from activity (persona-weighted)
        satisfaction_map = {
            Activity.GAMING: 1.5 * self.gaming_affinity,
            Activity.VIDEO_STREAMING: 1.0,
            Activity.VIDEO_RECORDING: 1.2,
            Activity.PHOTOGRAPHY: 1.1,
            Activity.SOCIAL_MEDIA: 1.3 * self.social_media_affinity,
            Activity.NAVIGATION: 0.8,
            Activity.MESSAGING: 1.0 * self.social_media_affinity,
            Activity.VOICE_CALL: 0.9,
            Activity.WORK_APPS: 1.2 * self.work_focus,
            Activity.IDLE: 0.1,
            Activity.CHARGING: 0.3,
            Activity.SCREEN_OFF: 0.0,
        }
        satisfaction = satisfaction_map.get(activity, 0.5)
        
        # Battery cost (higher when SOC is low)
        drain = old_soc - self.soc
        battery_cost = 0.3 * drain
        if self.soc < 30:
            battery_cost *= (1 + self.battery_anxiety * (30 - self.soc) / 30)
        if self.soc < 10:
            battery_cost *= 3  # Critical penalty
        
        # Thermal penalty
        thermal_penalty = 0.0
        if self.temperature > 40:
            thermal_penalty = 0.1 * (self.temperature - 40)
        if self.temperature > 50:
            thermal_penalty += 0.5  # Severe
        
        # Context bonuses
        context_bonus = 0.0
        hour = self.time
        
        # Gaming in evening = bonus
        if activity == Activity.GAMING and 19 <= hour <= 23:
            context_bonus += 0.3
        
        # Work during work hours = bonus
        if activity == Activity.WORK_APPS and 9 <= hour <= 17:
            context_bonus += 0.2 * self.work_focus
        
        # Charging when low = smart
        if activity == Activity.CHARGING and self.soc < 30:
            context_bonus += 0.5
        
        # Screen off at night = good
        if activity == Activity.SCREEN_OFF and (hour >= 23 or hour < 7):
            context_bonus += 0.2
        
        reward = satisfaction - battery_cost - thermal_penalty + context_bonus
        return reward


# =============================================================================
# DEEP RL AGENT
# =============================================================================

class DeepRLAgent:
    """
    Deep Reinforcement Learning agent with optional LLM guidance.
    
    Uses Double DQN with:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration (or LLM-guided)
    """
    
    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = NUM_ACTIONS,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        llm_advisor: Optional[GPT4Advisor] = None,
        use_dueling: bool = True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.llm_advisor = llm_advisor
        
        # Epsilon for exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        NetworkClass = DuelingDQN if use_dueling else DQN
        self.policy_net = NetworkClass(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.target_net = NetworkClass(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Tracking
        self.training_steps = 0
        self.episode_rewards = []
    
    def select_action(self, state: np.ndarray, env: PhoneEnvironment = None) -> int:
        """Select action using epsilon-greedy or LLM guidance."""
        
        # LLM-guided exploration (if available and early in training)
        if self.llm_advisor and self.epsilon > 0.3 and random.random() < 0.2:
            state_dict = {
                'soc': state[0] * 100,
                'temperature': state[1] * 50,
                'time': (np.arctan2(state[2], state[3]) / (2 * np.pi) * 24) % 24,
                'charging': bool(state[4]),
            }
            persona_dict = {
                'name': env.persona_name if env else 'default',
                'occupation': 'user',
                'gaming_affinity': int(state[5] * 10),
                'social_media': int(state[6] * 10),
                'work_focus': int(state[7] * 10),
                'battery_anxiety': int(state[8] * 10),
            }
            advice = self.llm_advisor.get_advice(state_dict, persona_dict)
            return self.llm_advisor.get_action_index(advice['suggested_action'])
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        # Greedy action from network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self) -> Optional[float]:
        """Perform one step of learning."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(DEVICE)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(DEVICE)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(DEVICE)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(DEVICE)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(DEVICE)
        
        # Current Q values
        current_q = self.policy_net(state_batch).gather(1, action_batch)
        
        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            next_q = self.target_net(next_state_batch).gather(1, next_actions)
            target_q = reward_batch + (1 - done_batch) * self.gamma * next_q
        
        # Huber loss (more robust than MSE)
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update()
        
        self.training_steps += 1
        return loss.item()
    
    def _soft_update(self):
        """Soft update of target network."""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episode_rewards': self.episode_rewards,
        }, path)
        print(f"💾 Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.episode_rewards = checkpoint['episode_rewards']
        print(f"📂 Model loaded from {path}")


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_agent(
    agent: DeepRLAgent,
    env: PhoneEnvironment,
    n_episodes: int = 1000,
    log_interval: int = 50,
    save_interval: int = 200,
    save_dir: str = "../models",
):
    """
    Train the Deep RL agent.
    """
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=f"{save_dir}/runs")
    
    print("=" * 60)
    print(f"🎮 TRAINING DEEP RL AGENT")
    print(f"   Persona: {env.persona_name}")
    print(f"   Device: {DEVICE}")
    print(f"   Episodes: {n_episodes}")
    print("=" * 60)
    
    best_reward = float('-inf')
    
    for episode in range(1, n_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        
        while True:
            # Select and perform action
            action = agent.select_action(state, env)
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Learn
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Decay exploration
        agent.decay_epsilon()
        agent.episode_rewards.append(episode_reward)
        
        # Logging
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        writer.add_scalar('Reward/episode', episode_reward, episode)
        writer.add_scalar('Loss/episode', avg_loss, episode)
        writer.add_scalar('Epsilon', agent.epsilon, episode)
        writer.add_scalar('SOC_final', env.soc, episode)
        
        if episode % log_interval == 0:
            avg_reward = np.mean(agent.episode_rewards[-log_interval:])
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg: {avg_reward:7.2f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"SOC: {env.soc:5.1f}%")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(f"{save_dir}/best_model_{env.persona_name}.pt")
        
        # Periodic save
        if episode % save_interval == 0:
            agent.save(f"{save_dir}/checkpoint_{env.persona_name}_{episode}.pt")
    
    writer.close()
    print("\n✅ Training complete!")
    print(f"   Best reward: {best_reward:.2f}")
    print(f"   Final epsilon: {agent.epsilon:.3f}")
    
    return agent


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_agent(agent: DeepRLAgent, env: PhoneEnvironment, n_episodes: int = 10):
    """Evaluate trained agent."""
    agent.policy_net.eval()
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # No exploration
    
    results = []
    
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        actions_taken = []
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            actions_taken.append(Activity(action).name)
            state = next_state
            
            if done:
                break
        
        results.append({
            'reward': total_reward,
            'final_soc': env.soc,
            'final_temp': env.temperature,
            'steps': env.steps,
            'actions': actions_taken,
        })
    
    agent.epsilon = original_epsilon
    
    avg_reward = np.mean([r['reward'] for r in results])
    avg_soc = np.mean([r['final_soc'] for r in results])
    
    print(f"\n📊 EVALUATION RESULTS ({n_episodes} episodes)")
    print(f"   Average reward: {avg_reward:.2f}")
    print(f"   Average final SOC: {avg_soc:.1f}%")
    
    # Action distribution
    all_actions = []
    for r in results:
        all_actions.extend(r['actions'])
    
    print(f"\n   Action distribution:")
    action_counts = {}
    for a in all_actions:
        action_counts[a] = action_counts.get(a, 0) + 1
    
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[:5]:
        pct = 100 * count / len(all_actions)
        print(f"   {action:20s}: {pct:5.1f}%")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Deep RL Agent")
    parser.add_argument('--persona', type=str, default='gamer', 
                        choices=['gamer', 'creator', 'commuter', 'chatter', 'streamer'])
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--api-key', type=str, default=None, help='OpenAI API key')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--load', type=str, default=None, help='Path to checkpoint')
    args = parser.parse_args()
    
    # Create environment
    env = PhoneEnvironment(persona_name=args.persona)
    
    # Create LLM advisor (optional)
    llm_advisor = None
    if args.api_key:
        llm_advisor = GPT4Advisor(api_key=args.api_key)
        print("🤖 GPT-4 advisor enabled!")
    
    # Create agent
    agent = DeepRLAgent(
        state_dim=9,
        action_dim=NUM_ACTIONS,
        hidden_dim=256,
        llm_advisor=llm_advisor,
        use_dueling=True,
    )
    
    if args.load:
        agent.load(args.load)
    
    if args.eval_only:
        evaluate_agent(agent, env, n_episodes=20)
    else:
        # Train
        agent = train_agent(
            agent, env,
            n_episodes=args.episodes,
            log_interval=50,
            save_dir="../models",
        )
        
        # Evaluate
        evaluate_agent(agent, env, n_episodes=20)
        
        # Final save
        agent.save(f"../models/final_{args.persona}.pt")
        
        if llm_advisor:
            print(f"\n📞 GPT-4 API calls made: {llm_advisor.call_count}")
