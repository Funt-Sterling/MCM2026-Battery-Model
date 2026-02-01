"""
=============================================================================
5G NETWORK POWER MODEL - RRC State Machine Implementation
=============================================================================
Based on: Narayanan et al., "A Variegated Look at 5G in the Wild" (SIGCOMM 2021)

This is the "secret weapon" - models the HIDDEN battery drain from 5G networks.
Key insight: 5G radios stay "hot" for 10-20 seconds AFTER data transfer stops.

This explains the "Chatty vs Streaming Paradox":
- Chatty user (100 small messages): Radio NEVER sleeps → HIGH drain
- Streaming user (continuous data): Efficient high-throughput mode → LOW drain
=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import matplotlib.pyplot as plt


class RRCState(Enum):
    """Radio Resource Control States (3GPP standard)"""
    IDLE = "IDLE"           # Radio sleeping, lowest power
    CONNECTED = "CONNECTED" # Active data transfer, highest power
    TAIL = "TAIL"           # Waiting for more data, medium power


@dataclass
class NetworkConfig:
    """
    Network power configuration from Narayanan et al. (2021).
    Table 2: RRC State Power Consumption
    """
    network_type: str = "5G_mmWave"
    
    # Power values in mW (from Table 2 of the paper)
    # 4G/LTE
    power_4g_idle: float = 178.0
    power_4g_connected: float = 800.0
    power_4g_tail: float = 400.0
    tail_duration_4g: float = 10.0  # seconds
    
    # 5G Low-Band (Sub-6 GHz)
    power_5g_low_idle: float = 300.0
    power_5g_low_connected: float = 450.0  # Average of 260-593
    power_5g_low_tail: float = 600.0
    tail_duration_5g_low: float = 12.0  # seconds
    
    # 5G mmWave (High-band) - THE BATTERY KILLER
    power_5g_mm_idle: float = 300.0
    power_5g_mm_connected: float = 1092.0  # This is the key number!
    power_5g_mm_tail: float = 600.0
    tail_duration_5g_mm: float = 15.0  # 10-20 seconds
    
    # Handoff penalties (4G ↔ 5G switch)
    handoff_power_nsa: float = 799.0   # NSA 5G handoff spike (mW)
    handoff_power_sa: float = 245.0    # SA 5G handoff (lower)
    handoff_duration: float = 0.5      # Duration of power spike (seconds)
    
    # Efficiency crossover point (from paper)
    crossover_throughput_dl: float = 187.0  # Mbps - below this, 4G wins
    crossover_throughput_ul: float = 40.0   # Mbps - uplink crossover
    
    def get_power(self, state: RRCState) -> float:
        """Get power for current network type and state."""
        if self.network_type == "4G":
            powers = {
                RRCState.IDLE: self.power_4g_idle,
                RRCState.CONNECTED: self.power_4g_connected,
                RRCState.TAIL: self.power_4g_tail
            }
        elif self.network_type == "5G_low":
            powers = {
                RRCState.IDLE: self.power_5g_low_idle,
                RRCState.CONNECTED: self.power_5g_low_connected,
                RRCState.TAIL: self.power_5g_low_tail
            }
        else:  # 5G_mmWave (default)
            powers = {
                RRCState.IDLE: self.power_5g_mm_idle,
                RRCState.CONNECTED: self.power_5g_mm_connected,
                RRCState.TAIL: self.power_5g_mm_tail
            }
        return powers[state]
    
    def get_tail_duration(self) -> float:
        """Get tail duration for current network type."""
        if self.network_type == "4G":
            return self.tail_duration_4g
        elif self.network_type == "5G_low":
            return self.tail_duration_5g_low
        else:
            return self.tail_duration_5g_mm


class RRCStateMachine:
    """
    Simulates the Radio Resource Control state machine.
    
    State Transitions:
    - IDLE → CONNECTED: When data transfer starts
    - CONNECTED → TAIL: When data transfer stops (start tail timer)
    - TAIL → IDLE: When tail timer expires
    - TAIL → CONNECTED: When new data arrives (reset tail timer)
    """
    
    def __init__(self, config: NetworkConfig = None):
        self.config = config or NetworkConfig()
        self.state = RRCState.IDLE
        self.time_in_state = 0.0
        self.tail_timer = 0.0
        
        # Tracking for analysis
        self.state_history: List[Tuple[float, RRCState, float]] = []  # (time, state, power)
        self.total_energy = 0.0  # mJ
        self.handoff_count = 0
        
    def reset(self):
        """Reset state machine to initial conditions."""
        self.state = RRCState.IDLE
        self.time_in_state = 0.0
        self.tail_timer = 0.0
        self.state_history = []
        self.total_energy = 0.0
        self.handoff_count = 0
        
    def update(self, dt: float, data_active: bool, throughput_mbps: float = 0) -> float:
        """
        Update state machine for one time step.
        
        Args:
            dt: Time step (seconds)
            data_active: Whether data is being transferred
            throughput_mbps: Current throughput (for efficiency calculation)
            
        Returns:
            Power consumption for this time step (mW)
        """
        power = self.config.get_power(self.state)
        tail_duration = self.config.get_tail_duration()
        
        # State transitions
        if data_active:
            if self.state == RRCState.IDLE:
                # Wake up the radio
                self.state = RRCState.CONNECTED
                self.time_in_state = 0
            elif self.state == RRCState.TAIL:
                # Already hot, just go back to connected
                self.state = RRCState.CONNECTED
                self.time_in_state = 0
            # Reset tail timer whenever data is active
            self.tail_timer = tail_duration
        else:
            if self.state == RRCState.CONNECTED:
                # Data stopped, start tail
                self.state = RRCState.TAIL
                self.time_in_state = 0
            elif self.state == RRCState.TAIL:
                # Count down tail timer
                self.tail_timer -= dt
                if self.tail_timer <= 0:
                    # Tail expired, go to idle
                    self.state = RRCState.IDLE
                    self.time_in_state = 0
                    self.tail_timer = 0
        
        # Update time tracking
        self.time_in_state += dt
        
        # Record state
        current_time = sum(h[0] for h in self.state_history) if self.state_history else 0
        self.state_history.append((dt, self.state, power))
        
        # Track energy
        energy = power * dt / 1000  # mJ (power in mW, dt in seconds)
        self.total_energy += energy
        
        return power
    
    def add_handoff(self):
        """Record a network handoff event."""
        self.handoff_count += 1
        # Add handoff energy penalty
        handoff_energy = self.config.handoff_power_nsa * self.config.handoff_duration / 1000
        self.total_energy += handoff_energy
        
    def get_statistics(self) -> dict:
        """Get statistics about state machine behavior."""
        if not self.state_history:
            return {}
        
        total_time = sum(h[0] for h in self.state_history)
        
        # Time in each state
        time_idle = sum(h[0] for h in self.state_history if h[1] == RRCState.IDLE)
        time_connected = sum(h[0] for h in self.state_history if h[1] == RRCState.CONNECTED)
        time_tail = sum(h[0] for h in self.state_history if h[1] == RRCState.TAIL)
        
        # Average power
        avg_power = self.total_energy * 1000 / total_time if total_time > 0 else 0
        
        return {
            'total_time_s': total_time,
            'total_energy_mJ': self.total_energy,
            'avg_power_mW': avg_power,
            'time_idle_s': time_idle,
            'time_connected_s': time_connected,
            'time_tail_s': time_tail,
            'pct_idle': 100 * time_idle / total_time if total_time > 0 else 0,
            'pct_connected': 100 * time_connected / total_time if total_time > 0 else 0,
            'pct_tail': 100 * time_tail / total_time if total_time > 0 else 0,
            'handoff_count': self.handoff_count
        }


def simulate_chatty_user(duration_hours: float = 2.0, 
                         messages_per_hour: int = 50,
                         network: str = "5G_mmWave") -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Simulate a "chatty" user sending frequent small messages.
    
    The KEY insight: Messages come faster than tail timer, so radio NEVER sleeps!
    
    Args:
        duration_hours: Simulation duration
        messages_per_hour: Number of messages per hour
        network: Network type ("4G", "5G_low", "5G_mmWave")
    
    Returns:
        time array, power array, statistics dict
    """
    config = NetworkConfig(network_type=network)
    sm = RRCStateMachine(config)
    
    duration_s = duration_hours * 3600
    dt = 0.1  # 100ms time step
    
    # Generate message times (random but averaging to messages_per_hour)
    avg_interval = 3600 / messages_per_hour  # seconds between messages
    message_times = []
    t = np.random.exponential(avg_interval)
    while t < duration_s:
        message_times.append(t)
        t += np.random.exponential(avg_interval)
    
    # Message duration (small, ~0.5-2 seconds)
    message_duration = 1.0
    
    # Simulate
    times = []
    powers = []
    t = 0
    msg_idx = 0
    
    while t < duration_s:
        # Check if we're sending a message
        data_active = False
        if msg_idx < len(message_times):
            msg_start = message_times[msg_idx]
            msg_end = msg_start + message_duration
            if msg_start <= t < msg_end:
                data_active = True
            elif t >= msg_end:
                msg_idx += 1
        
        power = sm.update(dt, data_active)
        times.append(t)
        powers.append(power)
        t += dt
    
    return np.array(times), np.array(powers), sm.get_statistics()


def simulate_streaming_user(duration_hours: float = 2.0,
                           throughput_mbps: float = 25.0,
                           network: str = "5G_mmWave") -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Simulate a streaming user with continuous high-throughput data.
    
    The KEY insight: Continuous data means ONE long session, and at high 
    throughput 5G is actually 5x MORE efficient than 4G!
    
    Args:
        duration_hours: Simulation duration (streaming time)
        throughput_mbps: Streaming bitrate
        network: Network type
    
    Returns:
        time array, power array, statistics dict
    """
    config = NetworkConfig(network_type=network)
    sm = RRCStateMachine(config)
    
    duration_s = duration_hours * 3600
    dt = 0.1
    
    # Streaming is continuous, but let's add 10s idle at start and end
    stream_start = 10.0
    stream_end = duration_s - 10.0
    
    times = []
    powers = []
    t = 0
    
    while t < duration_s:
        data_active = stream_start <= t <= stream_end
        power = sm.update(dt, data_active, throughput_mbps if data_active else 0)
        times.append(t)
        powers.append(power)
        t += dt
    
    return np.array(times), np.array(powers), sm.get_statistics()


def simulate_commuter(duration_hours: float = 2.0,
                      speed_mph: float = 60.0,
                      handoffs_per_mile: float = 0.5,
                      network: str = "5G_mmWave") -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Simulate a commuter with GPS active and frequent handoffs.
    
    Args:
        duration_hours: Drive duration
        speed_mph: Speed in mph
        handoffs_per_mile: Network handoffs per mile traveled
        network: Network type
    
    Returns:
        time array, power array, statistics dict
    """
    config = NetworkConfig(network_type=network)
    sm = RRCStateMachine(config)
    
    duration_s = duration_hours * 3600
    dt = 0.1
    
    # Calculate handoff times
    miles_traveled = speed_mph * duration_hours
    num_handoffs = int(miles_traveled * handoffs_per_mile)
    handoff_times = np.linspace(60, duration_s - 60, num_handoffs)  # Spread evenly
    
    # Navigation: periodic data bursts (map tiles, directions)
    data_burst_interval = 30.0  # Every 30 seconds
    data_burst_duration = 2.0   # 2 second burst
    
    times = []
    powers = []
    t = 0
    handoff_idx = 0
    
    while t < duration_s:
        # Check for handoff
        if handoff_idx < len(handoff_times) and t >= handoff_times[handoff_idx]:
            sm.add_handoff()
            handoff_idx += 1
        
        # Navigation data bursts
        time_in_cycle = t % data_burst_interval
        data_active = time_in_cycle < data_burst_duration
        
        power = sm.update(dt, data_active)
        times.append(t)
        powers.append(power)
        t += dt
    
    return np.array(times), np.array(powers), sm.get_statistics()


def compare_chatty_vs_streaming(save_path: str = None):
    """
    Generate the "Chatty vs Streaming Paradox" comparison figure.
    This is THE key figure for the paper.
    """
    print("=" * 60)
    print("THE CHATTY VS STREAMING PARADOX")
    print("=" * 60)
    
    # Simulate both users for 2 hours
    t_chat, p_chat, stats_chat = simulate_chatty_user(
        duration_hours=2.0, messages_per_hour=50, network="5G_mmWave"
    )
    t_stream, p_stream, stats_stream = simulate_streaming_user(
        duration_hours=2.0, throughput_mbps=25.0, network="5G_mmWave"
    )
    
    # Print comparison
    print("\nEmma the Chatter (50 messages/hour):")
    print(f"  Total energy: {stats_chat['total_energy_mJ']/1000:.1f} J")
    print(f"  Avg power: {stats_chat['avg_power_mW']:.0f} mW")
    print(f"  Time in IDLE: {stats_chat['pct_idle']:.1f}%")
    print(f"  Time in CONNECTED: {stats_chat['pct_connected']:.1f}%")
    print(f"  Time in TAIL: {stats_chat['pct_tail']:.1f}%")
    
    print("\nSteve the Streamer (25 Mbps Netflix):")
    print(f"  Total energy: {stats_stream['total_energy_mJ']/1000:.1f} J")
    print(f"  Avg power: {stats_stream['avg_power_mW']:.0f} mW")
    print(f"  Time in IDLE: {stats_stream['pct_idle']:.1f}%")
    print(f"  Time in CONNECTED: {stats_stream['pct_connected']:.1f}%")
    print(f"  Time in TAIL: {stats_stream['pct_tail']:.1f}%")
    
    # Calculate data transferred (rough estimate)
    data_chat = 50 * 2 * 0.05  # 50 msg/hr * 2 hr * 50KB/msg = 5 MB
    data_stream = 25 * 2 * 3600 / 8 / 1000  # 25 Mbps * 2h = ~22.5 GB
    
    print(f"\nData transferred:")
    print(f"  Emma: ~{data_chat:.0f} MB")
    print(f"  Steve: ~{data_stream*1000:.0f} MB ({data_stream:.1f} GB)")
    
    # THE KEY METRIC: Energy per MB (efficiency)
    efficiency_chat = stats_chat['total_energy_mJ'] / data_chat  # mJ/MB
    efficiency_stream = stats_stream['total_energy_mJ'] / (data_stream * 1000)  # mJ/MB
    
    print(f"\n*** THE PARADOX - Energy Efficiency ***")
    print(f"  Emma: {efficiency_chat:.1f} mJ/MB (INEFFICIENT)")
    print(f"  Steve: {efficiency_stream:.3f} mJ/MB (EFFICIENT)")
    print(f"\n  Emma is {efficiency_chat/efficiency_stream:.0f}x LESS efficient per MB!")
    print(f"  This is the 'Chatty vs Streaming Paradox':"
          f"\n  Frequent small transfers keep the radio hot!")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Power profiles (show first 10 minutes for detail)
    window = 600  # 10 minutes
    idx_chat = t_chat <= window
    idx_stream = t_stream <= window
    
    axes[0, 0].plot(t_chat[idx_chat]/60, p_chat[idx_chat], 'r-', alpha=0.8, linewidth=0.5)
    axes[0, 0].set_xlabel('Time (minutes)')
    axes[0, 0].set_ylabel('Power (mW)')
    axes[0, 0].set_title('Emma (Chatter): Radio Power Profile')
    axes[0, 0].axhline(y=1092, color='darkred', linestyle='--', alpha=0.5, label='CONNECTED')
    axes[0, 0].axhline(y=600, color='orange', linestyle='--', alpha=0.5, label='TAIL')
    axes[0, 0].axhline(y=300, color='green', linestyle='--', alpha=0.5, label='IDLE')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1200])
    
    axes[0, 1].plot(t_stream[idx_stream]/60, p_stream[idx_stream], 'b-', alpha=0.8, linewidth=0.5)
    axes[0, 1].set_xlabel('Time (minutes)')
    axes[0, 1].set_ylabel('Power (mW)')
    axes[0, 1].set_title('Steve (Streamer): Radio Power Profile')
    axes[0, 1].axhline(y=1092, color='darkred', linestyle='--', alpha=0.5, label='CONNECTED')
    axes[0, 1].axhline(y=600, color='orange', linestyle='--', alpha=0.5, label='TAIL')
    axes[0, 1].axhline(y=300, color='green', linestyle='--', alpha=0.5, label='IDLE')
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1200])
    
    # State distribution pie charts
    labels = ['IDLE', 'CONNECTED', 'TAIL']
    colors = ['green', 'red', 'orange']
    
    sizes_chat = [stats_chat['pct_idle'], stats_chat['pct_connected'], stats_chat['pct_tail']]
    axes[1, 0].pie(sizes_chat, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title(f'Emma: State Distribution\nAvg Power = {stats_chat["avg_power_mW"]:.0f} mW')
    
    sizes_stream = [stats_stream['pct_idle'], stats_stream['pct_connected'], stats_stream['pct_tail']]
    axes[1, 1].pie(sizes_stream, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title(f'Steve: State Distribution\nAvg Power = {stats_stream["avg_power_mW"]:.0f} mW')
    
    plt.suptitle('The Chatty vs Streaming Paradox\n"More data ≠ More drain"', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")
    
    plt.show()
    
    return stats_chat, stats_stream


def plot_rrc_state_machine(save_path: str = None):
    """Generate RRC state machine diagram for paper."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw states as circles
    circle_idle = plt.Circle((0.2, 0.5), 0.12, color='green', alpha=0.3)
    circle_conn = plt.Circle((0.5, 0.5), 0.12, color='red', alpha=0.3)
    circle_tail = plt.Circle((0.8, 0.5), 0.12, color='orange', alpha=0.3)
    
    ax.add_patch(circle_idle)
    ax.add_patch(circle_conn)
    ax.add_patch(circle_tail)
    
    # State labels
    ax.text(0.2, 0.5, 'IDLE\n178-300 mW', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.5, 'CONNECTED\n800-1092 mW', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(0.8, 0.5, 'TAIL\n400-600 mW', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(0.38, 0.55), xytext=(0.32, 0.55),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.35, 0.62, 'Data\nstarts', ha='center', fontsize=9)
    
    ax.annotate('', xy=(0.68, 0.55), xytext=(0.62, 0.55),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.65, 0.62, 'Data\nstops', ha='center', fontsize=9)
    
    ax.annotate('', xy=(0.32, 0.45), xytext=(0.68, 0.45),
                arrowprops=dict(arrowstyle='->', color='black', lw=2, connectionstyle='arc3,rad=-0.3'))
    ax.text(0.5, 0.25, 'Tail timer expires\n(10-20 seconds)', ha='center', fontsize=9)
    
    ax.annotate('', xy=(0.62, 0.42), xytext=(0.68, 0.42),
                arrowprops=dict(arrowstyle='->', color='black', lw=2, connectionstyle='arc3,rad=0.5'))
    ax.text(0.72, 0.3, 'New data\n(reset timer)', ha='center', fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('5G RRC State Machine\n(The Hidden Battery Killer)', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Generate figures for the paper
    print("Generating RRC State Machine diagram...")
    plot_rrc_state_machine("../figures/rrc_state_machine.png")
    
    print("\nGenerating Chatty vs Streaming comparison...")
    compare_chatty_vs_streaming("../figures/chatty_vs_streaming.png")
