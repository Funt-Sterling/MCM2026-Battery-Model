"""
=============================================================================
🚀 ULTIMATE TRAINING PIPELINE - GPU-ACCELERATED LEARNING
=============================================================================
This script trains Deep RL agents for ALL personas with:
- GPU acceleration (RTX 5070 Ti)
- Optional GPT-4 guidance
- TensorBoard logging
- Automatic model saving
- Comprehensive evaluation

Run: python train_all.py --api-key YOUR_KEY

Authors: MCM 2026 Team
=============================================================================
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent))

from deep_rl_agent import (
    DeepRLAgent, PhoneEnvironment, GPT4Advisor,
    train_agent, evaluate_agent, Activity, DEVICE
)


def train_all_personas(
    api_key: str = None,
    episodes_per_persona: int = 500,
    output_dir: str = "../models"
):
    """Train agents for all personas."""
    
    personas = ['gamer', 'creator', 'commuter', 'chatter', 'streamer']
    results = {}
    
    # Create LLM advisor if API key provided
    llm_advisor = None
    if api_key:
        llm_advisor = GPT4Advisor(api_key=api_key, model="gpt-4")
        print("🤖 GPT-4 Advisor ENABLED - Using LLM-guided exploration!")
    else:
        print("⚠️  No API key provided - Using pure epsilon-greedy exploration")
    
    print("\n" + "=" * 70)
    print("🎯 ULTIMATE TRAINING PIPELINE")
    print(f"   Device: {DEVICE}")
    print(f"   Personas: {personas}")
    print(f"   Episodes per persona: {episodes_per_persona}")
    print(f"   Output directory: {output_dir}")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    for i, persona in enumerate(personas):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(personas)}] Training: {persona.upper()}")
        print("=" * 70)
        
        # Create environment
        env = PhoneEnvironment(persona_name=persona)
        
        # Create fresh agent for each persona
        agent = DeepRLAgent(
            state_dim=9,
            action_dim=len(Activity),
            hidden_dim=256,
            lr=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            llm_advisor=llm_advisor,
            use_dueling=True,
        )
        
        # Train
        persona_start = time.time()
        agent = train_agent(
            agent, env,
            n_episodes=episodes_per_persona,
            log_interval=50,
            save_interval=200,
            save_dir=f"{output_dir}/{persona}",
        )
        persona_time = time.time() - persona_start
        
        # Evaluate
        eval_results = evaluate_agent(agent, env, n_episodes=20)
        
        # Store results
        results[persona] = {
            'training_time': persona_time,
            'final_epsilon': agent.epsilon,
            'training_steps': agent.training_steps,
            'best_reward': max(agent.episode_rewards),
            'avg_final_reward': np.mean(agent.episode_rewards[-50:]),
            'eval_avg_reward': np.mean([r['reward'] for r in eval_results]),
            'eval_avg_soc': np.mean([r['final_soc'] for r in eval_results]),
        }
        
        print(f"\n✅ {persona.upper()} complete in {persona_time:.1f}s")
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    
    for persona, stats in results.items():
        print(f"\n{persona.upper()}:")
        print(f"   Best reward: {stats['best_reward']:.2f}")
        print(f"   Eval avg reward: {stats['eval_avg_reward']:.2f}")
        print(f"   Eval avg final SOC: {stats['eval_avg_soc']:.1f}%")
        print(f"   Training time: {stats['training_time']:.1f}s")
    
    print(f"\n⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    if llm_advisor:
        print(f"📞 Total GPT-4 API calls: {llm_advisor.call_count}")
    
    # Save results
    with open(f"{output_dir}/training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def compare_with_without_llm(api_key: str, episodes: int = 300):
    """
    Compare training with and without LLM guidance.
    This demonstrates the value of LLM-guided exploration!
    """
    
    print("\n" + "=" * 70)
    print("🔬 EXPERIMENT: LLM-GUIDED vs PURE RL")
    print("=" * 70)
    
    persona = 'gamer'
    env = PhoneEnvironment(persona_name=persona)
    
    # Train WITHOUT LLM
    print("\n📊 Training WITHOUT LLM guidance...")
    agent_no_llm = DeepRLAgent(
        state_dim=9,
        action_dim=len(Activity),
        llm_advisor=None,
        use_dueling=True,
    )
    
    no_llm_start = time.time()
    agent_no_llm = train_agent(
        agent_no_llm, env,
        n_episodes=episodes,
        log_interval=50,
        save_dir="../models/no_llm",
    )
    no_llm_time = time.time() - no_llm_start
    no_llm_results = evaluate_agent(agent_no_llm, env, n_episodes=20)
    
    # Train WITH LLM
    print("\n📊 Training WITH LLM guidance...")
    llm_advisor = GPT4Advisor(api_key=api_key, model="gpt-4")
    
    agent_with_llm = DeepRLAgent(
        state_dim=9,
        action_dim=len(Activity),
        llm_advisor=llm_advisor,
        use_dueling=True,
    )
    
    env.reset()  # Reset environment
    with_llm_start = time.time()
    agent_with_llm = train_agent(
        agent_with_llm, env,
        n_episodes=episodes,
        log_interval=50,
        save_dir="../models/with_llm",
    )
    with_llm_time = time.time() - with_llm_start
    with_llm_results = evaluate_agent(agent_with_llm, env, n_episodes=20)
    
    # Compare
    print("\n" + "=" * 70)
    print("📊 COMPARISON RESULTS")
    print("=" * 70)
    
    no_llm_reward = np.mean([r['reward'] for r in no_llm_results])
    with_llm_reward = np.mean([r['reward'] for r in with_llm_results])
    
    print(f"\n{'Metric':<30} {'Without LLM':>15} {'With LLM':>15}")
    print("-" * 60)
    print(f"{'Avg Eval Reward':<30} {no_llm_reward:>15.2f} {with_llm_reward:>15.2f}")
    print(f"{'Training Time (s)':<30} {no_llm_time:>15.1f} {with_llm_time:>15.1f}")
    print(f"{'Best Training Reward':<30} {max(agent_no_llm.episode_rewards):>15.2f} {max(agent_with_llm.episode_rewards):>15.2f}")
    print(f"{'GPT-4 API Calls':<30} {'0':>15} {llm_advisor.call_count:>15}")
    
    improvement = (with_llm_reward - no_llm_reward) / abs(no_llm_reward) * 100
    print(f"\n🎯 LLM guidance improved reward by {improvement:+.1f}%")
    
    return {
        'no_llm': no_llm_results,
        'with_llm': with_llm_results,
    }


def visualize_learned_policy(agent: DeepRLAgent, env: PhoneEnvironment):
    """Visualize what the agent learned."""
    
    import matplotlib.pyplot as plt
    
    # Create grid of states and get Q-values
    soc_range = np.linspace(0, 1, 20)
    time_range = np.linspace(0, 1, 24)
    
    q_values_map = np.zeros((len(soc_range), len(time_range), len(Activity)))
    
    agent.policy_net.eval()
    with torch.no_grad():
        for i, soc in enumerate(soc_range):
            for j, t in enumerate(time_range):
                # Create state
                time_hours = t * 24
                state = np.array([
                    soc,  # SOC
                    0.7,  # temperature (35°C normalized)
                    np.sin(2 * np.pi * time_hours / 24),
                    np.cos(2 * np.pi * time_hours / 24),
                    0.0,  # not charging
                    env.gaming_affinity,
                    env.social_media_affinity,
                    env.work_focus,
                    env.battery_anxiety,
                ], dtype=np.float32)
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q = agent.policy_net(state_tensor).cpu().numpy()[0]
                q_values_map[i, j] = q
    
    # Get best actions
    best_actions = q_values_map.argmax(axis=-1)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Action policy heatmap
    im = axes[0].imshow(best_actions, aspect='auto', cmap='tab20',
                         extent=[0, 24, 0, 100], origin='lower')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Battery SOC (%)')
    axes[0].set_title(f'Learned Policy: {env.persona_name.title()}')
    
    # Add colorbar with action names
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_ticks(range(len(Activity)))
    cbar.set_ticklabels([a.name[:8] for a in Activity])
    
    # Q-value for top actions over time
    soc_idx = 15  # ~75% SOC
    top_actions = [Activity.GAMING, Activity.SOCIAL_MEDIA, Activity.IDLE, Activity.CHARGING]
    
    for action in top_actions:
        q_over_time = q_values_map[soc_idx, :, action.value]
        axes[1].plot(range(24), q_over_time, label=action.name.lower(), linewidth=2)
    
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Q-Value')
    axes[1].set_title(f'Q-Values at 75% SOC: {env.persona_name.title()}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../figures/policy_{env.persona_name}.png', dpi=150)
    plt.show()
    
    print(f"💾 Saved: ../figures/policy_{env.persona_name}.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultimate Training Pipeline")
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API key for GPT-4 guidance')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Episodes per persona')
    parser.add_argument('--persona', type=str, default=None,
                        help='Train single persona (default: all)')
    parser.add_argument('--compare', action='store_true',
                        help='Run LLM vs no-LLM comparison experiment')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize learned policy')
    parser.add_argument('--load', type=str, default=None,
                        help='Load existing model')
    args = parser.parse_args()
    
    # Try to get API key from environment if not provided
    if args.api_key is None:
        args.api_key = os.environ.get('OPENAI_API_KEY')
    
    if args.compare and args.api_key:
        # Run comparison experiment
        compare_with_without_llm(args.api_key, episodes=args.episodes)
    
    elif args.visualize and args.load:
        # Visualize existing model
        persona = args.persona or 'gamer'
        env = PhoneEnvironment(persona_name=persona)
        agent = DeepRLAgent(state_dim=9, action_dim=len(Activity))
        agent.load(args.load)
        visualize_learned_policy(agent, env)
    
    elif args.persona:
        # Train single persona
        env = PhoneEnvironment(persona_name=args.persona)
        
        llm_advisor = None
        if args.api_key:
            llm_advisor = GPT4Advisor(api_key=args.api_key)
        
        agent = DeepRLAgent(
            state_dim=9,
            action_dim=len(Activity),
            llm_advisor=llm_advisor,
        )
        
        if args.load:
            agent.load(args.load)
        
        agent = train_agent(agent, env, n_episodes=args.episodes)
        evaluate_agent(agent, env, n_episodes=20)
        agent.save(f"../models/final_{args.persona}.pt")
    
    else:
        # Train all personas
        train_all_personas(
            api_key=args.api_key,
            episodes_per_persona=args.episodes,
            output_dir="../models",
        )
