import argparse
import os
import multiprocessing
import csv
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from torch_env import RIS_MISO_Env

class PhysicalCheckpointCallback(BaseCallback):
    def __init__(self, L=0, max_steps=1, verbose=0):
        super(PhysicalCheckpointCallback, self).__init__(verbose)
        self.L_config = L
        self.max_steps = max_steps
        self.max_mismatch_reward = -float('inf')
        self.best_action = None
        self.time_logs = []

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        reward = self.locals['rewards'][0]
        
        if reward > self.max_mismatch_reward:
            self.max_mismatch_reward = reward
            # Copy action to avoid memory overwrites
            self.best_action = self.locals['actions'][0].copy()
            
        if self.num_timesteps % 100 == 0:
            self.time_logs.append({
                "Time step": self.num_timesteps,
                "Max. Min-Max Reward": round(float(self.max_mismatch_reward), 3)
            })
            
            # Export progress silently for the Tkinter Dashboard
            try:
                with open(f"progress_L{self.L_config}.txt", "w") as f:
                    progress_pct = min((self.num_timesteps / self.max_steps) * 100, 100)
                    f.write(f"{progress_pct:.1f},{self.num_timesteps},{self.max_mismatch_reward:.3f}")
            except:
                pass
                
        return True

def run_experiment(L, K, M, max_time_steps, seed=0):
    device = torch.device("cpu")
    # Setting number of threads explicitly to avoid thread explosion thrashing
    torch.set_num_threads(1)
    
    env = RIS_MISO_Env(
        num_users=K,
        num_BS_antennas=M,
        num_RIS_elements=L,
        beta_min=0.95,             # HWI Impairments softened (5% degradation max instead of 10%)
        mu_PDA=0.21,
        kappa_PDA=3.4,
        location_mu=0.6*np.pi,
        concentration_kappa=1.2,   
        uncertainty_factor=0.05,   # CSI Error softened (5% channel fog instead of 10%)
        AWGN_var=0.000001,
        Tx_power=1,
        bits=1,
        max_episode_steps=max_time_steps,
        seed=seed,
        L=L,
    )
    
    layers = [L*2**i for i in range(5, 0, -1)]
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(pi=layers, vf=layers)
    )
    
    print(f"Starting PPO Min-Max MSE Training for L={L}, K={K}, Steps={max_time_steps} on {device}...", flush=True)
    for _ in range(0): env.reset()
    
    # Force cpu locally to avoid multiprocessing GPU lock contention natively
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, ent_coef=0.01, verbose=0, device="cpu")
    
    chkpt_callback = PhysicalCheckpointCallback(L=L, max_steps=max_time_steps)
    model.learn(total_timesteps=max_time_steps, reset_num_timesteps=False, callback=chkpt_callback)
    
    # Clean up GUI progress file
    try: os.remove(f"progress_L{L}.txt")
    except: pass
    
    final_rates = []
    final_reward = 0.0
    
    # --- INFERENCE PHASE: PHYSICAL ACTION CHECKPOINTING ---
    if chkpt_callback.best_action is not None:
        obs, reward, done, truncated, info = env.step(chkpt_callback.best_action)
        final_rates = info.get('individual_rates', [])
        final_reward = reward

    outages = sum([1 for r in final_rates if r < 1.0])
    outage_str = "0 outages!" if outages == 0 else f"{outages} users starved (<1.0 bps/Hz)"
    
    print(f"-> Completed PPO L={L}: Peak Min-Max Reward={final_reward:.3f}, Outages={outage_str}", flush=True)
    return L, final_reward, final_rates, outage_str, chkpt_callback.time_logs

def worker_wrapper(args_tuple):
    # args_tuple: (L, users, max_steps)
    L, users, max_steps = args_tuple
    return run_experiment(L, users, 16, max_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PPO Min-Max-MSE Analysis Pipeline")
    parser.add_argument("--users", type=int, default=4, help="Number of users (K)")
    parser.add_argument("--steps", type=int, default=20480, help="Number of time steps to train per config")
    parser.add_argument("--ris_elements", type=int, nargs="+", default=[4], help="List of RIS elements (L) to evaluate")
    parser.add_argument("--out_dir", type=str, default="analysis_logs_PPO", help="Directory to save analytical logs")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    csv_file = os.path.join(args.out_dir, "analysis_results.csv")
    
    # Dynamically cap CPU footprint
    cpu_count = multiprocessing.cpu_count()
    num_workers = max(1, int(cpu_count * 0.70))
    # Cap extra processes so threads aren't wasted natively
    num_workers = min(num_workers, len(args.ris_elements))
    
    print(f"Running PPO Analysis for: K={args.users}, L={args.ris_elements}, max_steps={args.steps}")
    print(f"Multitasking gracefully utilizing {num_workers} out of {cpu_count} CPU cores (70% Max).")
    
    try:
        # Prevents hanging on Linux/CUDA native fork logic
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
        
    pool_args = [(L, args.users, args.steps) for L in args.ris_elements]
    
    # Bypass Multiprocessing Pool if only 1 config is run to allow perfect live terminal streaming
    if len(args.ris_elements) == 1:
        parallel_results = [worker_wrapper(arg) for arg in pool_args]
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            parallel_results = pool.map(worker_wrapper, pool_args)
        
    results = []
    for L, final_reward, ind_rates, outage_str, time_logs in parallel_results:
        result = {
            "L (RIS Elements)": f"L = {L}",
            "Min-Max Peak Reward": f"{final_reward:.3f}",
        }
        for idx, rate in enumerate(ind_rates):
            result[f"User {idx+1}"] = f"{rate:.2f}"
            
        result["Outages"] = outage_str
        results.append(result)
        
        # Save independent time traces specifically
        if time_logs:
            time_log_file = os.path.join(args.out_dir, f"time_log_L{L}.csv")
            log_fields = list(time_logs[0].keys())
            with open(time_log_file, 'w', newline='') as tcsv:
                twriter = csv.DictWriter(tcsv, fieldnames=log_fields)
                twriter.writeheader()
                for log_row in time_logs:
                    twriter.writerow(log_row)
                    
    # Export definitive global comparison
    if len(results) > 0:
        fields = list(results[0].keys())
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
                
        print(f"\nAnalysis completely synchronized! Results successfully backed up to '{args.out_dir}'")
