import argparse
import os
import multiprocessing

# Calculate exactly 70% CPU usage limit
cpu_count = multiprocessing.cpu_count()
num_workers = max(1, int(cpu_count * 0.70))

# We need to dynamically restrict math thread spawning mathematically 
# so 3 parallel PyTorch jobs don't spawn 3 * 16 threads and cause 100% thrashing.
# Assume roughly maximum `num_workers` concurrent jobs will run.
# Determine how many threads EACH individual job should spawn natively.
max_jobs = min(num_workers, 3) # we default test at 3 L elements realistically
threads_per_job = max(1, num_workers // max_jobs)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = str(threads_per_job)
os.environ["MKL_NUM_THREADS"] = str(threads_per_job)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_job)

import csv
import numpy as np
import torch
# explicitly restrict PyTorch
torch.set_num_threads(threads_per_job)
import gym

import multiprocessing

import Beta_Space_Exp_SAC
import utils

# Register the custom environment if not
from gym.envs.registration import register
try:
    register(
        id='RIS_MISO_PDA-v0',
        entry_point='environment:RIS_MISO_PDA',
    )
except Exception as e:
    pass

def whiten(state):
    return (state - np.mean(state)) / np.std(state)


def run_experiment(L, K, M, max_time_steps, seed=0):
    env_kwargs = {
        "num_antennas": M,
        "num_RIS_elements": L,
        "num_users": K,
        "mismatch": False,
        "channel_est_error": True,
        "cascaded_channels": True,
        "beta_min": 0.6,
        "theta_bar": 0.0,
        "kappa_bar": 1.5,
        "AWGN_var": 1e-2,
        "channel_noise_var": 1e-2,
        "seed": seed,
    }
    
    env = gym.make("RIS_MISO_PDA-v0", **env_kwargs)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    device = torch.device("cpu")
    
    power_t_val = 30 # Base reference 30dBm
    
    agent_kwargs = {
        "state_dim": state_dim,
        "action_space": env.action_space,
        "M": M,
        "N": L,
        "K": K,
        "power_t": power_t_val,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "policy_type": "Gaussian",
        "alpha": 0.2,
        "target_update_interval": 1,
        "automatic_entropy_tuning": True,
        "device": device,
        "discount": 1.0,
        "tau": 1e-3
    }
    
    agent = Beta_Space_Exp_SAC.Beta_Space_Exp_SAC(**agent_kwargs, beta_min=0.6)
    replay_buffer = utils.BetaExperienceReplayBuffer(state_dim, action_dim, L, int(2e4))
    
    state = env.reset()
    state = whiten(state)
    
    exp_regularization = 0.3
    batch_size = 16
    
    final_rates = None
    final_sum_rate = 0.0
    
    max_reward = 0
    max_mismatch_reward = 0
    time_logs = []

    print(f"Starting Training for L={L}, K={K}, Steps={max_time_steps} on {device}...")
    for t in range(max_time_steps):
        action, beta = agent.select_action(state, exp_regularization)
        next_state, reward, done, info = env.step(action, beta)
        
        mismatch_reward = info["true reward"]
        ind_rates = info.get("individual_rates", [])
        
        if reward > max_reward:
            max_reward = reward
        if mismatch_reward > max_mismatch_reward:
            max_mismatch_reward = mismatch_reward
            
        if (t + 1) % 100 == 0:
            time_logs.append({
                "Time step": t + 1,
                "Max. Reward": round(max_reward, 3),
                "Max. Mismatch Reward": round(max_mismatch_reward, 3)
            })
            
        next_state = whiten(next_state)
        
        # In later stages, we might want to track this directly
        # reward here is 1 step delay in mismatch since it returns true_reward anyway
        replay_buffer.add(state, action, beta, next_state, reward, float(done))
        
        state = next_state
        
        # Update
        if t > batch_size:
            agent.update_parameters(replay_buffer, exp_regularization, batch_size)
            
        # Linear schedule
        exp_regularization = 0.3 - (0.3 * (t / max_time_steps))
        
        # Save last few best or simply last step
        # Here we just take the last step as an approximation for final metrics
        if t == max_time_steps - 1:
            final_rates = ind_rates
            final_sum_rate = mismatch_reward

    # Determine outages (rate < 0.5 can be an outage, you can customize this)
    outages = sum([1 for r in final_rates if r < 0.5])
    
    if outages == 0:
        outage_str = "0 outages!"
    else:
        outage_str = f"{outages} users starved"
        
    print(f"-> Completed L={L}: System Total Rate={final_sum_rate:.2f}, Outages={outage_str}")
    return L, final_sum_rate, final_rates, outage_str, time_logs

def worker_wrapper(args_tuple):
    # args_tuple: (L, users, max_steps)
    L, users, max_steps = args_tuple
    return run_experiment(L, users, users, max_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analysis Pipeline")
    parser.add_argument("--users", type=int, default=4, help="Number of users (K) and antennas (M)")
    parser.add_argument("--steps", type=int, default=500, help="Number of time steps to train per config")
    parser.add_argument("--ris_elements", type=int, nargs="+", default=[4, 16, 64], help="List of RIS elements to evaluate")
    parser.add_argument("--out_dir", type=str, default="analysis_logs", help="Directory to save the CSV")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    csv_file = os.path.join(args.out_dir, "analysis_results.csv")
    
    # Calculate exactly 70% CPU usage limit
    cpu_count = multiprocessing.cpu_count()
    num_workers = max(1, int(cpu_count * 0.70))
    # Don't create unnecessarily many workers if there are fewer tasks
    num_workers = min(num_workers, len(args.ris_elements))
    
    print(f"Running analysis for: K={args.users}, L={args.ris_elements}, max_steps={args.steps}")
    print(f"Multiprocessing initialized: using {num_workers} out of {cpu_count} CPU cores (Max 70%).")
    
    results = []
    
    pool_args = [(L, args.users, args.steps) for L in args.ris_elements]
    
    try:
        # Avoid hanging issues natively with spawn if cuda is initialized early
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
        
    with multiprocessing.Pool(processes=num_workers) as pool:
        parallel_results = pool.map(worker_wrapper, pool_args)
        
    for L, sum_rate, ind_rates, outage_str, time_logs in parallel_results:
        result = {
            "L (RIS Elements)": f"L = {L}",
            "System Total Rate": f"{sum_rate:.2f} bps/Hz",
        }
        for idx, rate in enumerate(ind_rates):
            result[f"User {idx+1}"] = f"{rate:.2f}"
            
        result["Outages"] = outage_str
        results.append(result)
        
        if time_logs:
            time_log_file = os.path.join(args.out_dir, f"time_log_L{L}.csv")
            log_fields = list(time_logs[0].keys())
            with open(time_log_file, 'w', newline='') as tcsv:
                twriter = csv.DictWriter(tcsv, fieldnames=log_fields)
                twriter.writeheader()
                for log_row in time_logs:
                    twriter.writerow(log_row)
        
    # Write to CSV
    if len(results) > 0:
        fields = list(results[0].keys())
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
                
        print(f"\nAnalysis complete! Results saved to '{args.out_dir}'")
