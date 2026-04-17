import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def plot_latest_result():
    # Find all .npy files in Results directory
    result_files = glob.glob('./Results/**/*.npy', recursive=True)
    
    if not result_files:
        print("No result files found in ./Results")
        return

    # Sort by modification time to get the latest
    latest_file = max(result_files, key=os.path.getmtime)
    print(f"Plotting results from: {latest_file}")

    # Load data
    rewards = np.load(latest_file)
    
    # Calculate moving average for smoother plot
    window_size = 100
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label='Instant Rewards')
    plt.plot(range(window_size-1, len(rewards)), moving_avg, color='red', label=f'Moving Average (n={window_size})')
    
    plt.title(f'Learning Curve\n{os.path.basename(latest_file)}')
    plt.xlabel('Time Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    output_file = 'learning_curve.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_latest_result()
