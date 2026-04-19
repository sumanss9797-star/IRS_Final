"""
generate_report_figures.py
Generates all figures for the RIS-MISO DRL project report.
Run with: python generate_report_figures.py
Outputs to: ./report_figures/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from scipy.stats import vonmises

# ── output folder ──────────────────────────────────────────────────────────────
OUT = "report_figures"
os.makedirs(OUT, exist_ok=True)

ACADEMIC_STYLE = {
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "figure.dpi": 150,
}
plt.rcParams.update(ACADEMIC_STYLE)

def save(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1.1  RIS-Assisted MU-MISO System Architecture (block diagram)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12); ax.set_ylim(0, 8)
ax.axis('off')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')

def draw_box(ax, x, y, w, h, label, color='#4472C4', fontsize=10, text_color='white'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='#2c3e50', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold', wrap=True,
            multialignment='center')

# Base station
draw_box(ax, 0.3, 2.5, 2.0, 3.0, 'Base Station\n(M Antennas)', '#1a5276')
# RIS panel
draw_box(ax, 4.5, 2.0, 2.5, 4.0, 'RIS\nPanel\n(L Elements)', '#117a65')
# Users
for i, (uy, label, blocked) in enumerate([(6.5,'User 1\n(LOS)',False),(5.0,'User 2\n(LOS)',False),
                                           (3.5,'User 3\n(Blocked)',True),(2.0,'User 4\n(Edge)',True)]):
    col = '#7d6608' if blocked else '#1a5276'
    draw_box(ax, 9.5, uy - 0.4, 2.0, 0.9, label, col, fontsize=8)

# H1 arrow: BS -> RIS
ax.annotate('', xy=(4.5, 4.5), xytext=(2.3, 4.0),
            arrowprops=dict(arrowstyle='->', color='#117a65', lw=2.5))
ax.text(3.2, 4.7, 'H₁\n(BS→RIS)', ha='center', fontsize=9, color='#117a65', fontstyle='italic')

# H2 arrows: RIS -> Users
for uy in [6.5, 5.1, 3.8, 2.2]:
    ax.annotate('', xy=(9.5, uy), xytext=(7.0, 4.0),
                arrowprops=dict(arrowstyle='->', color='#1a5276', lw=1.8, linestyle='dashed'))
ax.text(8.4, 5.5, 'H₂\n(RIS→Users)', ha='center', fontsize=9, color='#1a5276', fontstyle='italic')

# Direct path (weak)
ax.annotate('', xy=(9.5, 3.8), xytext=(2.3, 3.5),
            arrowprops=dict(arrowstyle='->', color='#922b21', lw=1.0,
                           linestyle=':', connectionstyle='arc3,rad=0.35'))
ax.text(5.8, 1.5, 'Direct Path (Weak / Blocked)', ha='center', fontsize=8,
        color='#922b21', fontstyle='italic')

# DRL agent box at bottom
draw_box(ax, 4.0, 0.2, 3.5, 1.0, 'DRL Agent\n(PPO / SAC)', '#6c3483', fontsize=9)
ax.annotate('', xy=(5.75, 2.0), xytext=(5.75, 1.2),
            arrowprops=dict(arrowstyle='->', color='#6c3483', lw=1.8))
ax.text(6.5, 1.6, 'Action:\nPhase Config Φ', fontsize=7.5, color='#6c3483')
ax.annotate('', xy=(7.5, 0.7), xytext=(9.5, 2.0),
            arrowprops=dict(arrowstyle='->', color='#6c3483', lw=1.5,
                           connectionstyle='arc3,rad=-0.4'))
ax.text(8.8, 0.4, 'Reward:\n-max MSE', fontsize=7.5, color='#6c3483')

ax.set_title('Fig 1.1 — RIS-Assisted MU-MISO System Architecture', fontsize=13, fontweight='bold', pad=10)
save("Fig1_1_System_Architecture.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2.1  Phase-Dependent Amplitude Response
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
phi = np.linspace(0, 2*np.pi, 500)
beta_min, mu_PDA, kappa_PDA = 0.95, 0.21, 3.4
A = (1 - beta_min) * (np.sin(phi - mu_PDA + np.pi/2) ** kappa_PDA) + beta_min

ax.plot(phi, A, color='#1a5276', lw=2.5, label='Amplitude A(φ)')
ax.axhline(beta_min, color='#922b21', lw=1.5, linestyle='--', label=f'β_min = {beta_min}')
ax.fill_between(phi, beta_min, A, alpha=0.15, color='#1a5276')
ax.axvspan(0, 1.0, alpha=0.08, color='red')
ax.text(0.5, 0.955, 'HWI Loss Zone', ha='center', fontsize=9, color='#922b21')
ax.set_xlabel('Phase Shift φ (radians)', fontsize=11)
ax.set_ylabel('Normalized Reflection Amplitude A(φ)', fontsize=11)
ax.set_title('Fig 2.1 — Phase-Dependent Amplitude Response of a Practical RIS Element', fontsize=11, fontweight='bold')
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
ax.set_ylim(0.93, 1.01)
ax.legend(fontsize=10)
save("Fig2_1_PDA_Response.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2.2  Von-Mises Phase Error Distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
theta = np.linspace(-np.pi, np.pi, 500)
for kappa, label, color in [(1.2, 'κ=1.2 (Moderate Noise)', '#1a5276'),
                              (5.0, 'κ=5.0 (Mild Noise)', '#117a65'),
                              (0.5, 'κ=0.5 (Heavy Noise)', '#922b21')]:
    pdf = vonmises.pdf(theta, kappa)
    ax.plot(theta, pdf, lw=2.3, label=label, color=color)

ax.set_xlabel('Phase Error θ (radians)', fontsize=11)
ax.set_ylabel('Probability Density', fontsize=11)
ax.set_title('Fig 2.2 — Von-Mises Distribution for Hardware Phase Error Modeling', fontsize=11, fontweight='bold')
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
ax.legend(fontsize=10)
ax.text(0, 0.08, 'Used in this project\n(κ = 1.2)', ha='center', fontsize=8.5,
        color='#1a5276', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
save("Fig2_2_VonMises.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2.4  PPO Clipped Surrogate Objective
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
r = np.linspace(0.5, 1.5, 400)
adv = 1.0  # positive advantage
eps = 0.2

unclipped = r * adv
clipped = np.clip(r, 1 - eps, 1 + eps) * adv
objective = np.minimum(unclipped, clipped)

ax.plot(r, unclipped, lw=1.5, linestyle='--', color='#7f8c8d', label='r·Â (Unclipped)')
ax.plot(r, objective, lw=2.5, color='#1a5276', label='CLIP Objective (min)')
ax.axvspan(0.5, 1-eps, alpha=0.12, color='#922b21', label='Clipped Region')
ax.axvspan(1+eps, 1.5, alpha=0.12, color='#922b21')
ax.axvline(1-eps, lw=1.2, linestyle=':', color='#922b21')
ax.axvline(1+eps, lw=1.2, linestyle=':', color='#922b21')
ax.text(1-eps, 0.55, f'1-ε={1-eps}', ha='right', fontsize=9, color='#922b21')
ax.text(1+eps, 0.55, f'1+ε={1+eps}', ha='left',  fontsize=9, color='#922b21')
ax.set_xlabel('Probability Ratio r_t = π_θ(a|s) / π_θ_old(a|s)', fontsize=10)
ax.set_ylabel('Surrogate Objective Value', fontsize=10)
ax.set_title('Fig 2.4 — PPO Clipped Surrogate Objective Function (Positive Advantage)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.text(1.0, 0.7, 'Prevents too-large\npolicy updates', ha='center', fontsize=9,
        color='#2c3e50', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
save("Fig2_4_PPO_Clip.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4.1  SAC Greedy Baseline K=4, L=4
# ══════════════════════════════════════════════════════════════════════════════
def sac_bar_chart(rates, total, L, fig_name, fig_label):
    users = [f'User {i+1}' for i in range(len(rates))]
    colors = ['#922b21' if r == max(rates) else ('#7d6608' if r < 1.0 else '#1a5276') for r in rates]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(users, rates, color=colors, edgecolor='white', linewidth=1.2, width=0.55)
    ax.axhline(1.0, color='#117a65', lw=1.8, linestyle='--', label='Outage Threshold (1.0 bps/Hz)')
    ax.set_ylabel('Individual User Rate (bps/Hz)', fontsize=11)
    ax.set_title(f'{fig_label} — SAC Greedy Baseline: Per-User Rate (L={L}, K=4)\nTotal Rate = {total} bps/Hz', fontsize=10, fontweight='bold')
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{rate:.2f}', ha='center', fontsize=10, fontweight='bold')
    patches = [mpatches.Patch(color='#922b21', label='Dominant User (Greedy Monopoly)'),
               mpatches.Patch(color='#7d6608', label='Starved User (<1.0 bps/Hz)'),
               mpatches.Patch(color='#1a5276', label='Adequate Service')]
    ax.legend(handles=patches + [mpatches.Patch(color='#117a65', label='Outage Threshold')], fontsize=8.5)
    save(fig_name)

sac_bar_chart([0.39, 0.91, 5.45, 0.79], '7.54', 4,  "Fig4_1_SAC_L4_K4.png",  "Fig 4.1")
sac_bar_chart([0.67, 2.97, 8.42, 2.26], '14.32', 16, "Fig4_2_SAC_L16_K4.png", "Fig 4.2")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4.3  PPO Convergence L=16, K=4  (from actual time_log data)
# ══════════════════════════════════════════════════════════════════════════════
import csv

def load_time_log(path):
    steps, rewards = [], []
    try:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row['Time step']))
                key = [k for k in row if 'Reward' in k or 'reward' in k][0]
                rewards.append(float(row[key]))
    except Exception:
        pass
    return steps, rewards

fig, ax = plt.subplots(figsize=(9, 5))

# Load real data for L=16 K=4
s16, r16 = load_time_log("analysis_logs_PPO/time_log_L16.csv")
if s16:
    ax.plot(s16, r16, color='#1a5276', lw=2.2, label='L=16, K=4 (Min-Max Reward)')
    ax.axhline(r16[-1], color='#1a5276', lw=1.0, linestyle=':', alpha=0.6)
    ax.text(s16[-1]*0.6, r16[-1]+0.15, f'Converged: {r16[-1]:.3f}', fontsize=9, color='#1a5276')

# Load real data for L=4 K=4
s4, r4 = load_time_log("analysis_logs_PPO/time_log_L4.csv")
if s4:
    ax.plot(s4, r4, color='#117a65', lw=2.2, linestyle='--', label='L=4, K=4 (Min-Max Reward)')

ax.set_xlabel('Training Steps', fontsize=11)
ax.set_ylabel('Best Min-Max Reward (−max MSE)', fontsize=11)
ax.set_title('Fig 4.3 — PPO Min-Max Training Convergence (K=4 Users)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
save("Fig4_3_PPO_Convergence_L16.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4.4  Convergence comparison L=16 vs L=64
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
s16, r16 = load_time_log("analysis_logs_PPO/time_log_L16.csv")
s64, r64 = load_time_log("analysis_logs_PPO_K4_L64/time_log_L64.csv")
if s16: ax.plot(s16, r16, color='#1a5276', lw=2.2, label='L=16 (Faster Convergence)')
if s64: ax.plot(s64, r64, color='#922b21', lw=2.2, linestyle='--', label='L=64 (Larger Action Space)')
ax.set_xlabel('Training Steps', fontsize=11)
ax.set_ylabel('Best Min-Max Reward', fontsize=11)
ax.set_title('Fig 4.4 — PPO Convergence: L=16 vs L=64 (K=4 Users)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.text(0.55, 0.15, 'L=64 requires more steps\ndue to larger action space (2⁶⁴)',
        transform=ax.transAxes, fontsize=9, color='#922b21',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
save("Fig4_4_PPO_Convergence_L16vsL64.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4.5  Side-by-side SAC vs PPO grouped bar chart, K=4, L=16
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
users = ['User 1', 'User 2', 'User 3', 'User 4']
sac_rates  = [0.67, 2.97, 8.42, 2.26]
ppo_rates  = [0.40, 0.39, 0.42, 0.44]
x = np.arange(len(users))
w = 0.35

bars1 = ax.bar(x - w/2, sac_rates, w, label='SAC Baseline (Greedy)', color='#922b21', edgecolor='white')
bars2 = ax.bar(x + w/2, ppo_rates, w, label='PPO Min-Max (Fair)',    color='#1a5276', edgecolor='white')
ax.axhline(1.0, color='#117a65', lw=1.8, linestyle='--', label='Outage Threshold (1.0 bps/Hz)')
for bar, v in zip(bars1, sac_rates):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'{v:.2f}', ha='center', fontsize=9)
for bar, v in zip(bars2, ppo_rates):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'{v:.2f}', ha='center', fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(users, fontsize=11)
ax.set_ylabel('Individual User Rate (bps/Hz)', fontsize=11)
ax.set_title('Fig 4.5 — Per-User Rate: SAC Greedy vs PPO Min-Max\n(K=4 Users, L=16 Elements, 100k Steps)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.text(0.68, 0.72, '99.3% Fairness Gap\nReduction', transform=ax.transAxes,
        fontsize=10, fontweight='bold', color='#117a65',
        bbox=dict(boxstyle='round', facecolor='#eafaf1', alpha=0.9))
save("Fig4_5_SAC_vs_PPO_K4_L16.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4.6  PPO K=8, L=4 — Perfect Equilibrium
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
users8 = [f'User {i+1}' for i in range(8)]
rates8 = [0.19] * 8
bars = ax.bar(users8, rates8, color='#117a65', edgecolor='white', linewidth=1.2, width=0.6)
ax.axhline(0.19, color='#6c3483', lw=2.0, linestyle='--', alpha=0.6, label='Rate = 0.19 bps/Hz (All Users)')
ax.axhline(1.0,  color='#922b21', lw=1.5, linestyle=':', label='Outage Threshold (1.0 bps/Hz)')
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, '0.19', ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(0, 0.35)
ax.set_ylabel('Individual User Rate (bps/Hz)', fontsize=11)
ax.set_title('Fig 4.6 — Perfect Fairness: K=8 Users, L=4 Elements\nFairness Spread = 0.00 bps/Hz (Zero Variance)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.text(0.5, 0.75, '★ All 8 Users: Exactly 0.19 bps/Hz ★\nFairness Spread = 0.00', transform=ax.transAxes,
        ha='center', fontsize=11, fontweight='bold', color='#117a65',
        bbox=dict(boxstyle='round', facecolor='#eafaf1', alpha=0.9))
save("Fig4_6_PPO_K8_L4_PerfectEquilibrium.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4.7  Fairness Gap vs L elements
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
L_vals = [4, 16, 64]
sac_gap = [5.06, 7.75, None]  # SAC only for K=4
ppo_gap_k4 = [1.07, 0.05, 1.58]
ppo_gap_k8 = [0.00, 0.07, 0.00]

ax.plot(L_vals, sac_gap, 'o--', color='#922b21', lw=2.2, markersize=9,
        label='SAC Baseline (K=4)', zorder=3)
ax.plot(L_vals, ppo_gap_k4, 's-', color='#1a5276', lw=2.2, markersize=9,
        label='PPO Min-Max (K=4)', zorder=3)
ax.plot(L_vals, ppo_gap_k8, '^-', color='#117a65', lw=2.2, markersize=9,
        label='PPO Min-Max (K=8)', zorder=3)

# Annotate key point
ax.annotate('99.3% Reduction\n(0.05 bps/Hz)', xy=(16, 0.05), xytext=(30, 1.8),
            arrowprops=dict(arrowstyle='->', color='#1a5276', lw=1.5),
            fontsize=9, color='#1a5276',
            bbox=dict(boxstyle='round', facecolor='#eaf2ff', alpha=0.9))
ax.annotate('Perfect Equilibrium\n(0.00 spread)', xy=(4, 0.00), xytext=(8, 0.9),
            arrowprops=dict(arrowstyle='->', color='#117a65', lw=1.5),
            fontsize=9, color='#117a65',
            bbox=dict(boxstyle='round', facecolor='#eafaf1', alpha=0.9))

ax.set_xlabel('Number of RIS Elements (L)', fontsize=11)
ax.set_ylabel('Fairness Gap (Max − Min Rate, bps/Hz)', fontsize=11)
ax.set_title('Fig 4.7 — Fairness Gap vs. Number of RIS Elements', fontsize=11, fontweight='bold')
ax.set_xticks(L_vals); ax.set_xticklabels(['L=4', 'L=16', 'L=64'])
ax.legend(fontsize=10)
save("Fig4_7_FairnessGap_vs_L.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4.8  Histogram: Rate Distribution Comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Fig 4.8 — Rate Distribution: SAC (Greedy) vs PPO (Fair)', fontsize=12, fontweight='bold')

# SAC K=4 L=16 distribution
sac_data = [0.67, 2.97, 8.42, 2.26]
ax1.bar(['0.0–1.0', '1.0–3.0', '3.0–6.0', '6.0–10.0'],
        [1, 2, 0, 1], color='#922b21', edgecolor='white', width=0.5)
# More precise: just plot individual rates
ax1.cla()
ax1.hist(sac_data, bins=np.arange(0, 10, 0.5), color='#922b21', edgecolor='white', rwidth=0.8)
ax1.axvline(1.0, color='black', lw=1.8, linestyle='--', label='Outage Threshold')
ax1.set_xlabel('User Rate (bps/Hz)', fontsize=10)
ax1.set_ylabel('Number of Users', fontsize=10)
ax1.set_title('SAC Greedy (K=4, L=16)\nHigh Variance, User Starvation', fontsize=10)
ax1.set_ylim(0, 3); ax1.legend(fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.4)

# PPO K=8 L=4 perfect equilibrium
ppo_data = [0.19] * 8
ax2.hist(ppo_data, bins=np.arange(0, 0.5, 0.02), color='#117a65', edgecolor='white', rwidth=0.8)
ax2.axvline(0.19, color='#6c3483', lw=2.0, linestyle='--', label='All Users: 0.19 bps/Hz')
ax2.set_xlabel('User Rate (bps/Hz)', fontsize=10)
ax2.set_ylabel('Number of Users', fontsize=10)
ax2.set_title('PPO Min-Max (K=8, L=4)\nPerfect Equilibrium (Spread = 0.00)', fontsize=10)
ax2.set_ylim(0, 10); ax2.legend(fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
save("Fig4_8_Distribution_Histogram.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1.2  Relay vs RIS Comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Fig 1.2 — Conventional Active Relay vs Passive RIS Signal Propagation', fontsize=12, fontweight='bold')

for ax, title, relay_label, color in [
    (ax1, 'Conventional Active Relay', 'Active Relay\n(Needs Power Amp)', '#922b21'),
    (ax2, 'RIS-Aided System',          'RIS Panel\n(No Power Amp)', '#117a65')]:
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    ax.set_title(title, fontsize=11, fontweight='bold')
    # BS
    draw_box(ax, 0.2, 2.0, 1.5, 2.0, 'Base\nStation', '#1a5276', fontsize=9)
    # Middle node
    draw_box(ax, 4.0, 2.5, 1.8, 1.0, relay_label, color, fontsize=7.5)
    # User
    draw_box(ax, 8.0, 2.0, 1.5, 2.0, 'User\nDevice', '#1a5276', fontsize=9)
    ax.annotate('', xy=(4.0, 3.0), xytext=(1.7, 3.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.0))
    ax.annotate('', xy=(8.0, 3.0), xytext=(5.8, 3.0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.0))
    if ax == ax2:
        ax.text(5.0, 4.8, '✓ No extra power\n✓ Passive reflection', ha='center',
                fontsize=8.5, color='#117a65', bbox=dict(boxstyle='round', facecolor='#eafaf1'))
    else:
        ax.text(5.0, 4.8, '✗ Requires power amplifier\n✗ Adds cost & latency', ha='center',
                fontsize=8.5, color='#922b21', bbox=dict(boxstyle='round', facecolor='#fdf2f8'))

save("Fig1_2_Relay_vs_RIS.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3.2  Geometric ULA Channel Model
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')
ax.set_facecolor('#f8f9fa')
ax.set_title('Fig 3.2 — Rician Channel Model with ULA Steering Vectors', fontsize=12, fontweight='bold')

draw_box(ax, 0.3, 3.0, 1.5, 2.5, 'BS\n(M Antennas\nULA)', '#1a5276')
draw_box(ax, 4.2, 2.5, 1.8, 3.0, 'RIS\n(L Elements\nULA)', '#117a65')

for uy, label in [(6.5,'User 1'),(5.0,'User 2'),(3.5,'User 3'),(2.0,'User 4')]:
    draw_box(ax, 8.2, uy-0.3, 1.5, 0.7, label, '#6c3483', fontsize=8)

# LoS path BS→RIS
ax.annotate('', xy=(4.2, 4.0), xytext=(1.8, 4.0),
            arrowprops=dict(arrowstyle='->', color='#117a65', lw=2.5))
ax.text(3.0, 4.3, 'H₁ (Rician LoS + Scatter)', fontsize=8.5, ha='center', color='#117a65')

# Angle annotations
ax.annotate('', xy=(2.8, 5.2), xytext=(1.8, 4.5),
            arrowprops=dict(arrowstyle='->', color='#922b21', lw=1.5))
ax.text(2.5, 5.4, 'AoD: θ_BS', fontsize=8, color='#922b21')

ax.annotate('', xy=(3.6, 5.5), xytext=(4.2, 4.5),
            arrowprops=dict(arrowstyle='->', color='#d4ac0d', lw=1.5))
ax.text(2.9, 5.8, 'AoA: θ_RIS', fontsize=8, color='#d4ac0d')

# RIS to users
for uy in [6.2, 4.85, 3.35, 1.85]:
    ax.annotate('', xy=(8.2, uy), xytext=(6.0, 4.0),
                arrowprops=dict(arrowstyle='->', color='#6c3483', lw=1.5, linestyle='dashed'))
ax.text(7.2, 2.5, 'H₂ (Rician Scatter\nMultipath)', fontsize=8, ha='center', color='#6c3483')

ax.text(5.0, 1.0, 'ULA Spacing: d = λ/2 for both BS and RIS arrays', ha='center',
        fontsize=9, color='#2c3e50',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

save("Fig3_2_Channel_Model_ULA.png")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n✅  All figures saved to ./{OUT}/")
print(f"   Total: {len(os.listdir(OUT))} files")
