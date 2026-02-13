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
    target_tag = 'charts/victory'
    
    if target_tag not in tags:
        # Fallback to older runs if necessary, but warn.
        print(f"Warning: '{target_tag}' not found. Falling back to return threshold (UNCERTAIN).")
        target_tag = 'charts/episodic_return'
        
    returns = acc.Scalars(target_tag)
    # If using charts/victory, values are 1.0 for win, 0.0 for loss.
    if target_tag == 'charts/victory':
        victories = [r.value for r in returns if r.value > 0.5]
    else:
        # Old unreliable threshold
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, help="Path to the TensorBoard log directory")
    args = parser.parse_args()
    
    if args.log_dir:
        log_path = args.log_dir
    else:
        # Check latest logs if none provided
        log_root = "logs"
        runs = sorted([os.path.join(log_root, d) for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))], key=os.path.getmtime)
        log_path = runs[-1] if runs else None
        
    if log_path and os.path.exists(log_path):
        count_victories(log_path)
    else:
        print(f"Log path {log_path} not found.")
