import os
from tensorboard.backend.event_processing import event_accumulator

def extract_metrics(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    results = {}
    
    for tag in tags:
        events = ea.Scalars(tag)
        if events:
            # Get the last value and the max value if relevant
            last_val = events[-1].value
            max_val = max([e.value for e in events])
            mean_val = sum([e.value for e in events]) / len(events)
            results[tag] = {
                'last': last_val,
                'max': max_val,
                'mean': mean_val,
                'steps': events[-1].step
            }
            
    return results

if __name__ == "__main__":
    log_path = r"d:\programming\Sonic2_Prototype\logs\Sonic2_PPO__1769874931"
    metrics = extract_metrics(log_path)
    
    print("--- Training Metrics Summary ---")
    for tag, data in metrics.items():
        print(f"{tag}:")
        print(f"  Last: {data['last']:.4f}")
        print(f"  Max:  {data['max']:.4f}")
        print(f"  Mean: {data['mean']:.4f}")
        print(f"  Step: {data['steps']}")
