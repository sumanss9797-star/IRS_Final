import tkinter as tk
from tkinter import ttk
import os
import glob

class ProgressDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("PPO Physics Execution Dashboard")
        self.root.geometry("600x300")
        self.root.configure(bg="#1E1E1E")
        
        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TProgressbar", thickness=25, background="#00FF00", troughcolor="#333333")
        self.style.configure("TLabel", background="#1E1E1E", foreground="#FFFFFF", font=("Courier", 12))
        
        self.title_label = ttk.Label(self.root, text="Real-Time Matrix Simulation", font=("Courier", 14, "bold"))
        self.title_label.pack(pady=15)
        
        self.frames = {}
        self.update_progress()

    def update_progress(self):
        # Look for the dynamically generated progress tracker files
        files = glob.glob("progress_L*.txt")
        
        # If no files found, just wait
        if not files:
            self.root.after(500, self.update_progress)
            return
            
        for file in files:
            try:
                l_config = file.split("_L")[1].replace(".txt", "")
                with open(file, "r") as f:
                    content = f.read().strip()
                
                if not content:
                    continue
                    
                parts = content.split(",")
                if len(parts) == 3:
                    pct, step, reward = parts
                    pct_float = float(pct)
                    
                    if l_config not in self.frames:
                        frame = tk.Frame(self.root, bg="#1E1E1E")
                        frame.pack(fill=tk.X, padx=20, pady=10)
                        
                        lbl = ttk.Label(frame, text=f"L={l_config} | Reward: {reward}")
                        lbl.pack(anchor="w")
                        
                        bar = ttk.Progressbar(frame, orient="horizontal", length=560, mode="determinate", style="TProgressbar")
                        bar.pack(pady=5)
                        
                        self.frames[l_config] = {"frame": frame, "lbl": lbl, "bar": bar}
                    
                    # Update existing elements
                    self.frames[l_config]["lbl"].config(text=f"Array L={l_config} | Step: {step} | Reward: {reward} ({pct_float}%)")
                    self.frames[l_config]["bar"]["value"] = pct_float
            except Exception as e:
                pass
                
        self.root.after(200, self.update_progress) # Auto-refresh every 200 ms!

if __name__ == "__main__":
    root = tk.Tk()
    app = ProgressDashboard(root)
    root.mainloop()
