def main():
    """
    Entry point for the project.
    Directs new users to the correct scripts for training and evaluation.
    """
    print("="*60)
    print("Welcome to the Sonic 2 RL Project!")
    print("="*60)
    print("To start training the agent:")
    print("  python src/train.py")
    print("\nTo watch a trained agent play:")
    print("  python src/evaluate.py --model models/checkpoints/latest_checkpoint.pth")
    print("\nTo record a victory video:")
    print("  python src/record_victory.py models/checkpoints/latest_checkpoint.pth")
    print("="*60)

if __name__ == "__main__":
    main()
