import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def count_victories(log_dir):
    event_files = [f for f in os.listdir(log_dir) if "tfevents" in f]
    if not event_files:
        print("No event files found.")
        return
    
    event_file = os.path.join(log_dir, event_files[0])
    # Set window_size to a high number to load all events
    acc = EventAccumulator(event_file, size_guidance={'scalars': 0})
    acc.Reload()
    
    tags = acc.Tags().get('scalars', [])
    target_tag = 'charts/episodic_return'
    
    if target_tag not in tags:
        print(f"Target tag '{target_tag}' not found. Available: {tags[:10]}...")
        return
        
    returns = acc.Scalars(target_tag)
    # A win adds +5.0 (scaled) to the episode return.
    # We count episodes where return > 5.0.
    victories = [r.value for r in returns if r.value > 5.0]
    
    print(f"--- V17 Training Run: {os.path.basename(log_dir)} ---")
    print(f"Total Episodes Logged: {len(returns)}")
    print(f"Total Signpost Victories (Reward > 5.0): {len(victories)}")
    if victories:
        print(f"Peak Victory Reward: {max(victories):.2f}")
        # Show recent victory stats
        recent = victories[-5:] if len(victories) > 5 else victories
        print(f"Recent Victory Rewards: {[round(v, 2) for v in recent]}")

if __name__ == "__main__":
    log_path = "logs/Sonic2_PPO_finetune_1770226417"
    if os.path.exists(log_path):
        count_victories(log_path)
    else:
        print(f"Log path {log_path} not found.")
