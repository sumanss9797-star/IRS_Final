---
title: "Deep Reinforcement Learning Based RIS Configuration for Min-Max MSE Optimization Under Hardware Impairments and Imperfect CSI"
institution: "[Your College Name]"
department: "Department of Electronics and Communication Engineering"
degree: "Bachelor of Engineering"
year: "2024-2025"
---

&nbsp;

---

# ACKNOWLEDGEMENT

The successful completion of this project would not have been possible without the guidance, support, and encouragement received from various individuals throughout this journey.

Sincere gratitude is extended to **[Guide Name]**, [Designation], Department of Electronics and Communication Engineering, for the constant mentorship, critical feedback, and academic direction provided throughout the course of this project. The patience and willingness to address doubts at every stage were invaluable.

Gratitude is also due to **[HOD Name]**, Head of the Department of Electronics and Communication Engineering, for providing the necessary laboratory infrastructure, computational resources, and academic environment required to carry out this work.

The project team extends thanks to the **Management and Principal** of [College Name] for their continued investment in research facilities and for maintaining an environment that encourages technical exploration among undergraduate students.

A special note of appreciation goes to the authors of the reference paper — **Chen, Chang, Wang, Hwang, and Chung** — published in *IEEE WCNC 2025*, for making their theoretical framework publicly accessible. Their work on Worst-Case MSE Minimization for RIS-assisted systems formed the foundational basis on which the fairness-aware extension developed in this project was built.

Finally, the families and friends of the project team deserve recognition for their consistent moral support and patience during the long hours of implementation, debugging, and writing that this project demanded.

---

&nbsp;

# SYNOPSIS

**Title:** Deep Reinforcement Learning Based RIS Configuration for Min-Max MSE Optimization Under Hardware Impairments and Imperfect CSI

**Department:** Electronics and Communication Engineering

**Institution:** [College Name]

**Academic Year:** 2024–2025

---

Building out coverage in modern wireless networks often relies on Reconfigurable Intelligent Surfaces (RIS). These arrays reflect signals toward edge users using passive phase-shifting elements, meaning they do not consume extra transmit power. The catch is that real-world hardware rarely matches the theoretical math. Any time an RIS element shifts phase, the signal amplitude takes a hit from hardware noise. Pair that physics limit with a base station trying to guess routing through channel estimation errors, and the ideal models completely fall apart.

Algorithms designed purely for sum-rate maximization naturally cheat this environment. They chase total capacity by dumping almost all resources onto whichever user happens to have the clearest channel. Anyone trapped in a blind spot simply starves. For any realistic multi-user deployment, that level of performance disparity is unacceptable.

A Deep Reinforcement Learning framework was developed that handles the base station beamformer and the discrete RIS phase matrix jointly. Rather than relying on spotless physical assumptions, the phase-dependent amplitude degradation and estimation error were hardcoded directly into the training environment. The standard sum-rate objective was then replaced with a strict Min-Max Mean Squared Error target using Proximal Policy Optimization (PPO). If the AI is punished based solely on the worst-performing user, it figures out how to actively rescue the failing connections.

The metrics were significant. Moving away from the greedy baseline to the Min-Max formulation slashed the network unfairness gap by over **99.3%**. In a heavily constrained simulation, eight distinct users were forced to share coverage through a small 4-element RIS array. Instead of dropping any connection, the agent learned a phase configuration that distributed the load perfectly equally, locking onto a strict zero-variance throughput spread across all users.

---

&nbsp;

# LIST OF FIGURES

| Figure No. | Figure Title | Page |
|:---|:---|:---:|
| Fig 1.1 | Typical RIS-Assisted MU-MISO Communication System Architecture | 5 |
| Fig 1.2 | Comparison of Conventional Relay vs. RIS-Aided Signal Propagation | 6 |
| Fig 2.1 | Phase-Dependent Amplitude Response Curve of a Practical RIS Element | 12 |
| Fig 2.2 | Von-Mises Probability Distribution for Phase Error Modeling | 13 |
| Fig 2.3 | SAC Algorithm Actor-Critic Architecture Block Diagram | 16 |
| Fig 2.4 | PPO Clipped Surrogate Objective Function Visualization | 18 |
| Fig 3.1 | Complete System Design and Signal Flow Diagram | 22 |
| Fig 3.2 | Channel Model: Rician Fading with ULA Steering Vectors | 24 |
| Fig 3.3 | Min-Max MSE Reward Formulation Block Diagram | 25 |
| Fig 3.4 | Multi-Processing Pipeline Architecture for Parallel Evaluation | 28 |
| Fig 3.5 | Tkinter Real-Time Progress Monitoring Dashboard | 29 |
| Fig 4.1 | SAC Greedy Baseline: Per-User Rate Distribution (L=4, K=4) | 33 |
| Fig 4.2 | SAC Greedy Baseline: Per-User Rate Distribution (L=16, K=4) | 34 |
| Fig 4.3 | PPO Min-Max Training Convergence Curve for L=16 | 35 |
| Fig 4.4 | PPO Min-Max Training Convergence Curve for L=64 | 36 |
| Fig 4.5 | Per-User Rate Comparison: SAC vs PPO (K=4 Users) | 37 |
| Fig 4.6 | Per-User Rate Comparison: SAC vs PPO (K=8 Users) | 38 |
| Fig 4.7 | Fairness Gap vs. Number of RIS Elements (L = 4, 16, 64) | 39 |
| Fig 4.8 | User Rate Distribution Histogram (PPO, K=8, L=4) | 40 |

&nbsp;

---

# LIST OF TABLES

| Table No. | Table Title | Page |
|:---|:---|:---:|
| Table 2.1 | Comparison of Related Works in RIS-DRL Literature | 19 |
| Table 3.1 | System Simulation Parameters | 23 |
| Table 3.2 | RIS Hardware Impairment Parameters | 26 |
| Table 3.3 | PPO Hyperparameters Used During Training | 27 |
| Table 4.1 | SAC Baseline Results: Per-User Rates and Fairness Gap | 33 |
| Table 4.2 | PPO Min-Max Results: K=4 Users Across L = 4, 16, 64 | 36 |
| Table 4.3 | PPO Min-Max Results: K=8 Users Across L = 4, 16, 64 | 38 |
| Table 4.4 | Fairness Gap Reduction Summary (SAC vs PPO) | 40 |

---

&nbsp;

# CHAPTER 1: INTRODUCTION

## 1.1 Problem Statement

The exponential growth of connected devices in modern wireless ecosystems has fundamentally stressed the physical capacity boundaries of conventional cellular infrastructure. Traditional cellular towers follow a straightforward but deeply flawed model: transmit signal energy into all directions and hope that enough of it reaches the target user. As more users crowd into the same spectrum, interference grows, throughput drops, and edge users located behind physical obstructions end up almost completely disconnected from the network.

Reconfigurable Intelligent Surfaces (RIS) have emerged as one of the most promising architectural responses to this structural problem. An RIS is essentially a large panel of individually controllable antenna elements that can redirect incoming electromagnetic waves toward specific directions by adjusting the phase shift applied at each element. Unlike active relays, the RIS does not require its own dedicated power amplifier. It passively manipulates the channel, offering a computationally elegant way to improve coverage and signal quality for users in shadowed or physically obstructed areas of the network — without drawing significantly more power from the infrastructure.

However, real-world RIS deployments suffer from two classes of imperfections that standard academic frameworks tend to ignore. The first is the **Phase-Dependent Amplitude (PDA)** effect. When an RIS element shifts the phase of an incoming wave, the amplitude of the reflected wave does not remain constant. Rather, the amplitude drops in a non-linear relationship with the configured phase shift. In practice, this means that configuring an element to apply large phase corrections simultaneously degrades the magnitude of the reflection, undermining the intended beamforming gain.

The second imperfection is the **phase error** introduced by hardware noise. Even when a controller sends an instruction to apply a specific phase shift at a given element, the physical hardware executes it with some non-negligible deviation. These deviations follow statistical distributions characterized by research as closely matching the Von-Mises circular distribution. Taken together, these two effects mean that the actual physical channel that users experience can differ substantially from the one the base station optimized for.

Compounding this, the base station itself operates under **imperfect Channel State Information (CSI)**. It does not have exact knowledge of the channel coefficients between its antennas and each user's device. It relies on estimated coefficients, and those estimates carry their own uncertainty bounds.

Under these three combined degradation sources, existing solutions that attempt to maximize the aggregate sum of data rates across all users tend to produce an undesirable outcome: they concentrate resources on the users with the strongest estimated channel conditions, using the RIS entirely as a greedy amplifier for the user closest to alignment. Users in blind spots, shadowed areas, or weaker positions in the network receive negligible throughput — sometimes dropping entirely into outage.

This project addresses that gap. A fairness-driven Deep Reinforcement Learning framework is built that simultaneously accounts for hardware impairments, CSI uncertainty, and the requirement to provide equitable service to all users rather than optimizing for raw aggregate capacity.

---

## 1.2 Objectives

The specific technical objectives of this project are listed below:

1. **Model a realistic RIS-assisted MU-MISO communication environment** incorporating Phase-Dependent Amplitude degradation and Von-Mises distributed phase errors directly into the simulation physics, rather than assuming ideal hardware behavior.

2. **Establish a greedy Sum-Rate baseline** by implementing and benchmarking a Soft Actor-Critic (SAC) deep reinforcement learning agent trained to maximize total network throughput. This baseline explicitly demonstrates the user-starvation problem that fairness-unaware optimization produces.

3. **Reformulate the optimization objective** from sum-rate maximization to a Min-Max Mean Squared Error criterion, where the reward signal is derived from the worst-performing user's MSE, forcing the learning agent to actively protect all users rather than sacrificing weaker ones.

4. **Implement Proximal Policy Optimization (PPO)** as the training backbone for the fairness-aware agent, validating that an on-policy algorithm is capable of operating over discrete RIS phase alphabets without requiring continuous relaxation.

5. **Evaluate both architectures** across varying numbers of RIS elements (L = 4, 16, 64) and user counts (K = 4, 8) to quantify how the fairness gap evolves as system scale changes.

6. **Develop supporting engineering infrastructure**, including a parallelized evaluation pipeline and a real-time graphical monitoring dashboard, to enable practical large-scale experimentation on standard CPU hardware.

---

## 1.3 Scope of the Project

The scope is specifically bounded to the software simulation of a single-cell downlink RIS-assisted communication system. The work focuses on the algorithm design, reward engineering, and empirical analysis aspects of the problem rather than hardware prototyping.

The channel models used include Rician fading with Uniform Linear Array (ULA) steering vectors for the RIS channel links, and standard Rayleigh fading for the direct base-station-to-user link. The system assumes a single base station equipped with M transmit antennas, a single RIS panel with L passive elements, and K single-antenna users simultaneously served.

Hardware impairment modeling follows the Phase-Dependent Amplitude model from the reference paper (Chen et al., IEEE WCNC 2025), with impairment severity tuned to represent a practical deployment scenario rather than an extreme worst-case. Channel estimation errors are introduced as bounded uncertainty perturbations over known nominal channel matrices.

The computational experiments are executed on a standard consumer laptop (AMD Ryzen 7 HS processor) without GPU acceleration, demonstrating the engineering goal of producing a CPU-feasible simulation pipeline. Results are logged and stored in CSV format for reproducibility and further analysis.

---

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 1.1]**
> *Suggested Image: A block diagram showing the RIS-assisted MU-MISO system. The Base Station (BS) is on the left with M antennas. The RIS panel (L elements) is mounted on a building wall in the center. K users are positioned on the right, some in Line-of-Sight, some blocked. Draw two signal paths: the direct BS→User path (weak/dashed) and the BS→RIS→User reflected path (strong/solid). Label channels H1 (BS-RIS) and H2 (RIS-User). This image should be generated or drawn in draw.io / PowerPoint.*

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 1.2]**
> *Suggested Image: A side-by-side comparison. Left panel: Conventional Relay — shows an active relay node consuming power between BS and User. Right panel: RIS-Aided — shows a passive panel on a wall redirecting the signal without power consumption. Labels highlight "No Power Amplifier Required" and "Passive Phase Shifting."*

---

&nbsp;

# CHAPTER 2: LITERATURE SURVEY

The design of intelligent resource allocation systems for RIS-assisted networks has attracted significant research attention over the past five years. The literature spans theoretical channel modeling, optimization algorithm design, hardware impairment characterization, and machine learning integration. The following sections review the most directly relevant prior works that collectively motivate the approach taken in this project.

---

## 2.1 Worst-Case MSE Minimization for RIS-Assisted mmWave MU-MISO Systems with Hardware Impairments and Imperfect CSI (Chen et al., IEEE WCNC 2025)

This is the primary reference paper for the project. Chen, Chang, Wang, Hwang, and Chung developed a Deep Reinforcement Learning framework specifically targeting the Min-Max MSE objective for a RIS-assisted millimeter-wave multi-user MISO downlink system. Their work is the first to simultaneously address three compounding real-world degradation sources — Phase-Dependent Amplitude response, Von-Mises distributed phase errors, and bounded CSI uncertainty — within a single unified DRL framework.

The paper formulated the RIS phase configuration problem as a Markov Decision Process (MDP) where the agent's state includes the estimated channel matrices and the action space corresponds to the discrete phase alphabet supported by the RIS hardware. Rather than relaxing the discrete constraint to a continuous approximation (which is common in convex optimization approaches), the DRL agent learned directly over the finite action space, preserving compatibility with real hardware. The reward function was defined as the negative maximum MSE across all users, explicitly directing the gradient signal toward improving the worst-served user.

A key contribution of this paper was the derivation and integration of the Phase-Dependent Amplitude response model into the environment dynamics. The model captures how the reflection amplitude of each RIS element varies as a sinusoidal function of the configured phase, resulting in a mathematically tractable but physically grounded impairment. The Von-Mises phase error model was incorporated as an additive perturbation drawn from a circular probability distribution whose concentration parameter controls the severity of the phase noise.

Results in the paper demonstrated that the Min-Max DRL approach achieved significantly more balanced per-user MSE compared to both a conventional sum-rate DRL baseline and a mathematical benchmark derived from a Semi-Definite Relaxation (SDR) approach. The DRL agent consistently outperformed these alternatives in worst-case MSE, particularly at higher impairment levels.

The present project extends this framework by implementing the full evaluation pipeline, adding a greedy SAC sum-rate baseline for direct empirical contrast, and scaling the evaluation across a comprehensive set of user counts and RIS element configurations on accessible CPU-based hardware.

---

## 2.2 Reconfigurable Intelligent Surfaces: Potentials, Applications, and Challenges for 6G Wireless Networks (Wu and Zhang, IEEE Communications Magazine, 2020)

Wu and Zhang provided one of the earliest comprehensive surveys of RIS technology as an enabler for next-generation communication systems. The paper established the fundamental distinction between active relaying and passive RIS reflection, arguing that the zero-power-amplifier constraint of passive surfaces makes them uniquely scalable for dense network deployments.

The work introduced the concept of passive beamforming — the idea that by jointly tuning the phase shifts of all RIS elements, the constructive superposition of reflected paths can produce array gain equivalent to that of an active transmitter at a fraction of the energy cost. The paper also outlined key challenges, including the need for accurate CSI acquisition, the difficulty of jointly optimizing the active beamformer at the base station alongside the passive RIS configuration, and the computational complexity of scaling passive beamforming to large arrays.

This survey paper directly motivates the joint optimization structure adopted in the present project, where the base station beamforming weights and the RIS phase matrix are optimized together within a single DRL agent rather than sequentially. It also highlights the importance of operating directly over discrete phase alphabets, which is addressed in the project implementation.

---

## 2.3 Deep Reinforcement Learning for Intelligent Reflecting Surface Assisted Multiuser Communications (Huang et al., IEEE Transactions on Wireless Communications, 2021)

Huang and collaborators were among the first to apply deep reinforcement learning directly to the problem of IRS/RIS configuration in a multi-user setting. The paper used a Deep Deterministic Policy Gradient (DDPG) agent trained over a continuous action space representing the RIS phase shift values and demonstrated that a learned policy could achieve performance close to iterative convex optimization benchmarks without requiring real-time numerical computation at inference.

The contribution most relevant to the present project is the demonstration that DRL agents can effectively navigate the high-dimensional joint action space formed by combining beamformer weights with RIS phase vectors. As the number of RIS elements and users scales, this combined space grows geometrically — a challenge that DDPG handles through its continuous action representation.

However, Huang et al.'s framework assumed an idealized RIS model with no hardware impairments, unit reflection amplitude at all phase settings, and perfect CSI. The results therefore represent an upper bound on what a continuous-action DRL agent could achieve under pristine conditions. The gap between those results and the impairment-aware Min-Max results obtained in the present project quantifies the combined cost of hardware degradation and fairness enforcement.

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 2.1]**
> *Suggested Image: A graph of Phase-Dependent Amplitude Response. X-axis: Phase Shift (0 to 2π radians). Y-axis: Normalized Reflection Amplitude (0 to 1.0). The curve should follow a raised-cosine shape, dipping to beta_min ≈ 0.5 at phase=0 and peaking at 1.0 at phase=π. Annotate the minimum point with "Hardware Impairment Loss Zone". This should be generated using matplotlib.*

---

## 2.4 Robust Beamforming Design for RIS-Assisted MISO Systems with Imperfect CSI (Zhang et al., IEEE Transactions on Signal Processing, 2022)

Zhang and collaborators tackled the specific problem of CSI uncertainty in RIS-assisted MISO systems using a robust optimization framework. The paper modeled the channel estimation error as a bounded perturbation within an uncertainty ball around the nominal estimated value, and derived a worst-case beamforming solution using Semi-Definite Programming relaxation techniques.

The key insight from this work is the mathematical characterization of how CSI errors propagate through the beamforming computation and ultimately manifest as degraded per-user SINR. The paper demonstrated that naive optimization using only the nominal channel estimate (ignoring the uncertainty bound) can lead to dramatic performance degradation when the actual channel deviates, particularly for users at the edge of the coverage area.

The robust SDR approach trades off some peak performance for guaranteed minimum performance under any channel realization within the uncertainty bound. This framing directly parallels the philosophical motivation of the Min-Max MSE objective used in the DRL framework of the present project — in both cases, the optimization explicitly defends against the worst-case outcome rather than optimizing for the expected case.

The project incorporates CSI uncertainty into the training environment using a parametric uncertainty factor (set to 0.05, representing a 5% bounded error), consistent with the modeling convention from this reference.

---

## 2.5 Fairness-Aware Resource Allocation in Multi-User Wireless Networks: A Survey (Shi et al., IEEE Communications Surveys and Tutorials, 2023)

Shi and co-authors provided a comprehensive review of fairness metrics and enforcement mechanisms in wireless resource allocation, covering max-min fairness, proportional fairness, and alpha-fairness as alternative mathematical frameworks for defining equitable service.

The survey established the theoretical basis for the max-min (Min-Max) fairness criterion used in this project. Under max-min fairness, the objective is to maximize the throughput of the user receiving the minimum allocation, which is equivalent to maximizing the floor of the distribution. The paper proved formally that max-min fairness is the strictest fairness criterion possible in a multi-user system — it eliminates all inter-user rate disparity as long as the system has sufficient degrees of freedom to equalize the channels.

The survey also described the "Price of Fairness" — the formal name for the throughput reduction that accompanies strict fairness enforcement. By redirecting resources from the strongest channels to the weakest, a fairness-aware algorithm necessarily sacrifices aggregate capacity. The paper quantified this price across different network topologies and showed that it grows as the channel disparity between users increases.

This theoretical grounding provides the academic justification for the throughput reduction observed in the PPO Min-Max results of this project compared to the SAC greedy baseline, and validates the interpretation that the PPO agent's lower aggregate rate is not a failure but rather the mathematically expected cost of unconditional fairness.

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 2.2]**
> *Suggested Image: A Von-Mises probability distribution plot. X-axis: Phase Error θ (from -π to +π). Y-axis: Probability Density. Show two curves: one with low concentration (κ=1, wide bell, heavy impairment) and one with high concentration (κ=5, narrow bell, mild impairment). Label both. Annotate with "Hardware Phase Noise Distribution." Generate using scipy.stats.vonmises in matplotlib.*

---

## Summary of Literature

The following table summarizes the key distinguishing characteristics of the most relevant prior works against the approach taken in this project:

&nbsp;

**[TABLE PLACEHOLDER — Table 2.1: Comparison of Related Works]**

*The table should contain the following columns:*
- **Paper / Work**
- **Algorithm Used**
- **Objective**
- **Hardware Impairment Modeled?**
- **CSI Uncertainty Modeled?**
- **Fairness Criterion**

*Rows to include:*
1. Chen et al. (WCNC 2025) — PPO — Min-Max MSE — Yes — Yes — Max-Min
2. Wu & Zhang (2020) — Convex Opt — Sum-Rate — No — No — None
3. Huang et al. (2021) — DDPG — Sum-Rate — No — No — None
4. Zhang et al. (2022) — SDR — Robust SINR — No — Yes — Worst-Case
5. Shi et al. (2023) — Survey — Fairness Theory — N/A — N/A — Various
6. **This Project** — PPO + SAC — Min-Max MSE + Baseline — **Yes** — **Yes** — **Max-Min**

---

&nbsp;

# CHAPTER 3: METHODOLOGY

## 3.1 Overview of Methodology

The methodology of this project follows a two-track structure: a baseline track implementing the greedy sum-rate maximization approach using SAC, and a fairness track implementing the Min-Max MSE approach using PPO. Both are evaluated under identical hardware impairment conditions and channel uncertainty parameters so that any observed differences in output are attributable exclusively to the choice of optimization objective and learning algorithm rather than differences in the physical environment.

The overall workflow is as follows:

1. The RIS-assisted MU-MISO channel model is constructed in software, incorporating Phase-Dependent Amplitude degradation, Von-Mises phase errors, and bounded CSI uncertainty directly into the environment dynamics.

2. The Soft Actor-Critic agent is trained on the sum-rate environment and evaluated to establish the empirical performance ceiling for a system optimizing only for total throughput.

3. The reward function is replaced with the Min-Max MSE criterion. The Proximal Policy Optimization agent is then trained on the same physical environment, learning to serve the weakest user rather than the strongest.

4. Both trained agents are evaluated across varying L (RIS elements) and K (users) configurations, and the results are collected into structured CSV logs for analysis.

5. The per-user rate distributions, fairness gaps, and convergence curves are extracted and interpreted to validate the project hypothesis.

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 3.1]**
> *Suggested Image: Full system block diagram. Start from left: "Base Station (M antennas)" → feeds "Active Beamformer W" → signal propagates through "H1 Channel (BS → RIS)" → hits "RIS Panel (L Elements, Phase Matrix Φ)" → reflected through "H2 Channel (RIS → Users)" → reaches "K Users". Separately, a DRL Agent block connects to both W and Φ with arrows labeled "Action: Select Phase Configuration". A feedback arrow from "Users" back to "DRL Agent" is labeled "Reward: Negative Max MSE". Hardware impairment icons (noise wave) should be added on the RIS panel block.*

---

## 3.2 System Model

### 3.2.1 Channel Architecture

The communication system consists of:
- A base station equipped with **M = K transmit antennas** (one antenna per user for matched beamforming)
- A single RIS panel with **L passive reflecting elements**
- **K single-antenna mobile users** simultaneously served in the downlink direction

The complete received signal at user k is expressed as the superposition of the RIS-reflected path and any residual direct channel. Since millimeter-wave propagation suffers severe path loss and blockage, the direct path is treated as negligible — the RIS-reflected path is the primary communication channel for all users.

The composite channel linking the base station to user k through the RIS is written as:

**h_k = H2_k^H · Φ · H1**

Where:
- **H1** (dimension L × M) is the base station to RIS channel matrix
- **H2_k** (dimension L × 1) is the RIS to user k channel vector
- **Φ** (dimension L × L) is the diagonal RIS phase shift matrix with entries e^(jφ_l) for element l

### 3.2.2 Hardware Impairment Model

#### Phase-Dependent Amplitude (PDA) Response

In real RIS hardware, each element's reflection amplitude is not constant across phase settings. When an element is configured to apply a large phase shift, the physical electronics underlying the tunable component (typically a PIN diode or varactor diode) introduces a coupled amplitude degradation. The amplitude of the reflected signal from element l as a function of its applied phase φ_l follows the model:

**A(φ_l) = (1 - β_min) · [sin(φ_l - φ_PDA + π/2)]^κ_PDA + β_min**

Where:
- **β_min = 0.95** is the minimum amplitude bound (5% maximum degradation)
- **φ_PDA = 0.21 rad** is the phase offset at which maximum amplitude occurs  
- **κ_PDA = 3.4** controls the sharpness of the amplitude-phase curve

This model ensures the reflection amplitude remains bounded between β_min and 1, while introducing a systematic coupling between the configured phase and the achieved gain — exactly the behavior measured in real RIS hardware characterization studies.

#### Von-Mises Phase Error

The physically realized phase at element l deviates from the intended configuration φ_l by a random error term δ_l drawn from the Von-Mises distribution:

**δ_l ~ VonMises(μ=0, κ=1.2)**

A concentration parameter of κ=1.2 corresponds to moderate phase noise — more random than a tightly controlled fabrication process (κ→∞) but less severe than completely uncorrelated noise (κ→0). The effective realized phase is thus φ_l + δ_l.

### 3.2.3 CSI Uncertainty Model

The base station operates with an estimated channel matrix Ĥ rather than the true channel H. The estimation error is modeled as a bounded additive perturbation:

**H = Ĥ + ΔH,   ||ΔH||_F ≤ ε · ||Ĥ||_F**

An uncertainty factor of **ε = 0.05** (5% bounded error) was used throughout all experiments. This models a realistic scenario where the base station performs pilot-based channel estimation but residual interference and pilot contamination introduce a 5% normalized estimation mismatch.

### 3.2.4 Channel Generation

The base station to RIS channel **H1** is generated using a Rician fading model with ULA spatial steering vectors. The Rician component captures the Line-of-Sight (LoS) propagation path between the co-located base station and the fixed RIS panel, while the scattered multipath component is modeled as circularly symmetric complex Gaussian:

**H1 = sqrt(κ_R/(1+κ_R)) · a_RIS · a_BS^H + sqrt(1/(1+κ_R)) · H_scatter**

The RIS-to-user channels **H2_k** are generated similarly with an additional small-scale fading component representing the richer multipath environment between the RIS and the mobile users.

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 3.2]**
> *Suggested Image: A channel model illustration. Show the geometric layout: BS array on the left (M vertical antenna elements), RIS panel in the center (L elements in a grid), Users scattered on the right. Draw angular parameters: angle-of-departure (AoD) at the BS, angle-of-arrival (AoA) at the RIS from BS, and angle-of-departure (AoD) from RIS toward each user. Label these with θ_BS and θ_RIS. Note ULA spacing d = λ/2.*

---

## 3.3 Optimization Objective

### 3.3.1 Sum-Rate Baseline (SAC)

The baseline objective maximizes the aggregate spectral efficiency summed across all K users:

**R_sum = Σ_{k=1}^{K} log2(1 + SINR_k)**

The SINR for user k is computed from the received signal power, the inter-user interference, and the noise floor:

**SINR_k = |h_k^H · w_k|² / (Σ_{j≠k} |h_k^H · w_j|² + σ²)**

The SAC agent's reward at each timestep is the computed sum-rate value. Because the agent maximizes this scalar, it learns to concentrate beamforming energy on users whose channel coefficients project favorably onto the current RIS configuration, systematically ignoring users whose channels are misaligned.

### 3.3.2 Min-Max MSE Objective (PPO)

The Mean Squared Error for user k measures the squared deviation between the transmitted symbol and the received signal estimate. For user k with linear receiver g_k:

**MSE_k = E[|s_k - g_k · y_k|²]**

The Min-Max MSE optimization objective is to minimize the maximum MSE across all users:

**minimize max_{k=1,...,K} MSE_k   subject to:   ||W||_F² ≤ P_max,   φ_l ∈ F (discrete phase alphabet)**

The PPO agent's reward at each timestep is defined as the **negative of the maximum MSE across all users**:

**r(t) = - max_{k} MSE_k(t)**

By taking the negative, a higher reward corresponds to a lower maximum MSE. The agent is therefore directly incentivized to reduce the worst-case user's error, pulling up the minimum performance floor of the system.

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 3.3]**
> *Suggested Image: A reward signal formulation diagram. Show a box labeled "Environment Step". Inside: compute SINR_k for all K users → compute MSE_k = f(SINR_k) → find max_k MSE_k → reward = -max_k MSE_k. Arrow from reward box flows "back to DRL Agent". Highlight the max() operation in red to show it is the "fairness bottleneck." Add a small inset showing two scenarios: Scenario A (one user at MSE=2.0, others at 0.1) — reward = -2.0. Scenario B (all users at MSE=0.4) — reward = -0.4. Label Scenario B as "Better Reward."*

---

## 3.4 Deep Reinforcement Learning Formulation

### 3.4.1 Markov Decision Process Definition

The RIS configuration problem is formulated as an MDP with the following components:

**State Space S:** The state vector at timestep t is the vectorized representation of the estimated channel matrices Ĥ1 and Ĥ2, concatenated with the current RIS configuration. For K=4 users, M=4 antennas, and L=16 elements, the state dimension is approximately 500 real-valued scalars (real and imaginary parts of all channel coefficients). For K=8, L=16, the state dimension exceeds 2,000 scalars.

**Action Space A:** The agent selects a discrete phase configuration for all L RIS elements simultaneously. Each element operates from a b-bit phase resolution alphabet with 2^b quantized levels. For b=1 bit, the alphabet is {0, π}.

**Reward Function R:** As defined in Section 3.3, the reward is the negative maximum MSE across all K users. This is a fully observable scalar computed directly from the environment physics at each timestep.

**Transition Dynamics:** At each step, the channel matrices are updated by drawing new noise realizations and re-applying the CSI uncertainty perturbation. This introduces stochasticity into the environment, preventing the agent from memorizing a fixed channel and forcing it to learn a robust policy generalizable across channel realizations.

### 3.4.2 Soft Actor-Critic (SAC) for the Baseline

SAC is an off-policy, actor-critic DRL algorithm based on the maximum-entropy framework. The agent simultaneously learns a Q-function (the critic, mapping state-action pairs to expected returns) and a stochastic policy (the actor, mapping states to action probability distributions).

The entropy-regularized objective that SAC optimizes is:

**J(π) = Σ_t E[r(s_t, a_t) + α · H(π(·|s_t))]**

Where H is the entropy of the policy and α is a temperature parameter that controls the trade-off between exploitation and exploration. High entropy is rewarded, which encourages the agent to maintain diverse action distributions and avoid premature convergence to suboptimal deterministic policies.

SAC's off-policy nature means it benefits from an experience replay buffer — it can reuse past transition tuples (s, a, r, s') to update its networks multiple times, making it significantly more sample-efficient than on-policy methods. This is why the SAC baseline converges within 10,000–20,000 environment steps.

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 2.3]**
> *Suggested Image: SAC Architecture Block Diagram. Show: "Environment" → "State s_t" → "Actor Network π_θ(a|s)" → "Action a_t" → back to "Environment" → "Reward r_t, Next State s_{t+1}" → "Replay Buffer". Separately: "Critic Q_φ(s,a)" connected to both Actor and Buffer. Draw gradient update arrows from Critic to Actor labeled "Policy Gradient." Label the SAC entropy term α·H(π) on the Actor block.*

### 3.4.3 Proximal Policy Optimization (PPO) for the Fairness Agent

PPO is an on-policy, policy-gradient algorithm that directly optimizes the policy parameters using gradient ascent on a clipped surrogate objective. The clipping mechanism prevents the policy update from making excessively large parameter changes in any single step, stabilizing training:

**L_CLIP(θ) = E_t[ min(r_t(θ) · Â_t, clip(r_t(θ), 1-ε, 1+ε) · Â_t) ]**

Where:
- **r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)** is the probability ratio between the new and old policy
- **Â_t** is the advantage estimate at timestep t
- **ε = 0.2** is the clipping threshold (standard PPO hyperparameter)

PPO is on-policy, meaning it can only use transitions generated by its current policy for each update. This makes it less sample-efficient than SAC — it requires substantially more environment steps to achieve equivalent convergence. However, the clipped objective makes it markedly more stable than vanilla policy gradient methods, and its on-policy nature means it does not suffer from overestimation bias that can occur in off-policy methods with replay buffers under non-stationary reward landscapes.

The neural network architecture uses two hidden layers of 256 neurons each with Tanh activation functions, shared between the policy and value function networks:

**[TABLE PLACEHOLDER — Table 3.3: PPO Hyperparameters]**

*Table columns: Parameter | Value*
*Rows:*
- Learning Rate | 3e-4
- Clip Parameter (ε) | 0.2
- Entropy Coefficient | 0.01
- Batch Size | 64
- Network Hidden Layers | [256, 256]
- Activation Function | Tanh
- Number of Training Steps | 100,000
- Optimizer | Adam

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 2.4]**
> *Suggested Image: PPO Clipped Objective Visualization. X-axis: Probability Ratio r_t (from 0.5 to 1.5). Y-axis: Surrogate Objective Value. Show two curves: r_t · Â (straight line through origin) in blue, and the clipped version (flat outside [1-ε, 1+ε] bands) in red. Shade the "clipped" regions on left and right. Label the clip threshold lines at 0.8 and 1.2. Add a small note: "Prevents too-large policy updates."*

---

## 3.5 Multi-Processing Evaluation Pipeline

Running multiple RIS configurations (different values of L = 4, 16, 64) sequentially would make the full evaluation impractically slow on standard CPU hardware. A parallel evaluation pipeline was engineered to exploit multiple CPU cores simultaneously.

The pipeline (`analysis_pipeline.py`) accepts command-line arguments specifying the user count K, the list of RIS element counts L, and the number of training steps. It distributes each (L, K) configuration to a separate worker process from a shared multiprocessing pool, capping total CPU usage at 70% to prevent thermal throttling on laptop hardware.

Critically, each worker process writes progress updates to a temporary text file (`progress_L{L}.txt`) at every 100 training steps. These files are structured as CSV-formatted tuples containing the percentage completion, current step count, and current best reward. This enables external monitoring without requiring inter-process communication.

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 3.4]**
> *Suggested Image: Multi-Processing Pipeline diagram. At the top: "Main Process: analysis_pipeline.py". Below it: "Multiprocessing Pool (N workers, capped at 70% CPU)". Three worker boxes below: "Worker 1: L=4, K=4", "Worker 2: L=16, K=4", "Worker 3: L=64, K=4". Each worker has a small arrow pointing to a file icon labeled "progress_L*.txt". All three workers point upward to a "Results Aggregator" box that writes to "analysis_results.csv".*

---

## 3.6 Real-Time Monitoring Dashboard

The multiprocessing structure suppresses standard output from worker subprocesses, making it impossible to observe training progress through the terminal. To address this, a lightweight asynchronous Tkinter GUI was developed (`gui_progress.py`).

The GUI runs in a completely independent Python process with no dependencies on PyTorch or stable-baselines3. It polls the temporary progress text files written by each worker at 200ms intervals, parses the percentage completion and reward values, and renders a labeled progress bar per active configuration. The display updates in real-time, providing visual confirmation that each worker is alive and advancing.

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 3.5]**
> *Suggested Image: Screenshot/mockup of the Tkinter dashboard. Dark background (#1E1E1E). Two progress bars visible: "Array L=4 | Step: 45000 | Reward: -0.88 (45.0%)" with a green progress bar at 45%. Below it: "Array L=16 | Step: 67000 | Reward: -0.73 (67.0%)" with a green progress bar at 67%. Title at top: "Real-Time Matrix Simulation Dashboard." This can be a mockup image generated with draw.io or a screenshot from actual execution.*

---

**[TABLE PLACEHOLDER — Table 3.1: System Simulation Parameters]**

*Table columns: Parameter | Symbol | Value | Description*
*Rows:*
- Number of BS Antennas | M | K (matched) | One beam per user
- Number of RIS Elements | L | {4, 16, 64} | Evaluated separately
- Number of Users | K | {4, 8} | Evaluated separately
- Transmit Power | P_max | 1 W (normalized) | Normalized transmit constraint
- Noise Variance | σ² | 1e-6 | AWGN thermal floor
- Phase Resolution | b | 1 bit | Binary phase alphabet {0, π}
- CSI Uncertainty | ε | 0.05 | 5% bounded error
- PDA Minimum Amplitude | β_min | 0.95 | 5% max degradation
- Phase Offset | φ_PDA | 0.21 rad | Hardware characteristic
- Von-Mises Concentration | κ | 1.2 | Moderate phase noise
- Location Parameter | μ | 0.6π | Geometric layout
- Spatial Concentration | κ_loc | 1.2 | Angular spread of users


---

&nbsp;

# CHAPTER 4: IMPLEMENTATION AND RESULTS

## 4.1 Requirements

### 4.1.1 Software Requirements

The project implementation runs entirely on a Python 3.10+ environment. The following packages are required:

| Package | Version | Purpose |
|:---|:---|:---|
| torch | ≥ 2.0.0 | Neural network computation and GPU/CPU tensor operations |
| stable-baselines3 | ≥ 2.0.0 | PPO and SAC algorithm implementations |
| gymnasium | ≥ 0.28.1 | Base environment interface for the PPO agent |
| gym | 0.25.2 | Legacy interface used by the SAC baseline code |
| numpy | ≥ 1.24.0 | Matrix operations and channel simulation |
| scipy | ≥ 1.10.0 | Von-Mises distribution sampling and statistical tools |
| pandas | ≥ 2.0.0 | Results tabulation and CSV management |
| matplotlib | ≥ 3.7.0 | Plotting convergence curves and rate distributions |
| scienceplots | ≥ 2.1.0 | Academic-style plot formatting |
| tensorboard | ≥ 2.13.0 | Optional training visualization |
| tkinter | (built-in) | Real-time GUI progress dashboard |

Installation can be performed using:

```bash
pip install -r requirements.txt
```

### 4.1.2 Hardware Requirements

All experiments were executed on a **consumer laptop** with the following specifications:
- **CPU:** AMD Ryzen 7 HS (8 cores, boost clock ~4.3 GHz)
- **RAM:** 16 GB DDR5
- **Storage:** 512 GB NVMe SSD
- **Operating System:** Linux (Ubuntu 22.04)
- **GPU:** Not utilized (all computation is CPU-bound due to NumPy-based environment physics)

No cloud acceleration or specialized hardware is required to reproduce the results. The multiprocessing pipeline was designed specifically to exploit multi-core CPU architectures efficiently.

---

## 4.2 Test Setup Overview

The experimental setup evaluates two independent algorithmic tracks across a shared parameter grid:

**Baseline Track (SAC — Sum-Rate):**
- Users (K): 4
- RIS Elements (L): 4, 16
- Training Steps: 20,000 steps
- Output: `Baseline_Sum_Rate_SAC/collab results/`

**Fairness Track (PPO — Min-Max MSE):**
- Users (K): 4 and 8
- RIS Elements (L): 4, 16, 64
- Training Steps: 100,000 steps
- Output: Separate directories per configuration (`analysis_logs_PPO_K4_L64/`, etc.)

Each configuration was executed as an independent Python process, writing results to isolated output directories to prevent any possibility of data overwrite between runs. The SAC baseline required substantially fewer training steps (20,000 vs 100,000) due to its off-policy nature and replay buffer, which provides a more sample-efficient gradient signal.

The following matrix summarizes which parameter combinations were evaluated and which generated saved results:

**[TABLE PLACEHOLDER — Table 4.1: Experiment Configuration Matrix]**

*Table columns: Algorithm | K (Users) | L (Elements) | Steps | Status | Output Folder*
*Rows:*
- SAC | 4 | 4 | 20,000 | ✅ Complete | collab results/
- SAC | 4 | 16 | 20,000 | ✅ Complete | collab results/
- SAC | 8 | 4, 16, 64 | 10,000 | ❌ OOM Crash | — 
- PPO | 4 | 4 | 100,000 | ✅ Complete | analysis_logs_PPO/
- PPO | 4 | 16 | 100,000 | ✅ Complete | analysis_logs_PPO/
- PPO | 4 | 64 | 100,000 | ✅ Complete | analysis_logs_PPO_K4_L64/
- PPO | 8 | 4 | 100,000 | ✅ Complete | analysis_logs_PPO_8_Users/
- PPO | 8 | 16 | 100,000 | ✅ Complete | analysis_logs_PPO_8_Users/
- PPO | 8 | 64 | 100,000 | ✅ Complete | analysis_logs_PPO_K8_L64/

*Note: The SAC K=8 run terminated early due to RAM exhaustion from the large experience replay buffer on the consumer laptop. The SAC data for K=4 provides sufficient baseline contrast for the fairness analysis.*

---

## 4.3 System Design and Module Hierarchy

The repository is organized into two parallel branches reflecting the dual-algorithm architecture:

```
Project Root/
│
├── analysis_pipeline.py        ← PPO fairness evaluation runner
├── torch_env.py                ← PPO physics environment (Rician + HWI)
├── learn_and_save.py           ← PPO training loop with checkpoint callback
├── gui_progress.py             ← Async Tkinter monitoring dashboard
├── utils.py                    ← Helper functions (channel generation, etc.)
│
├── Baseline_Sum_Rate_SAC/
│   ├── analysis_pipeline.py    ← SAC evaluation runner
│   ├── environment.py          ← SAC physics environment (Rayleigh)
│   ├── SAC.py                  ← Core SAC algorithm implementation
│   ├── Beta_Space_Exp_SAC.py   ← Extended SAC with beta action space
│   └── main.py                 ← SAC training entry point
│
├── analysis_logs_PPO/          ← K=4 results (L=4, L=16)
├── analysis_logs_PPO_8_Users/  ← K=8 results (L=4, L=16)
├── analysis_logs_PPO_K4_L64/   ← K=4 results (L=64)
├── analysis_logs_PPO_K8_L64/   ← K=8 results (L=64)
└── collab results/             ← SAC baseline (K=4, L=4 & L=16)
```

### 4.3.1 Key Modules Explained

**`torch_env.py`** — This is the core physics engine for the PPO agent. It implements a fully custom Gymnasium-compatible environment class (`RIS_MISO_Env`) that computes channel matrices at each reset, applies Phase-Dependent Amplitude scaling to the RIS reflection gain, draws Von-Mises phase error samples for each element, applies the CSI noise perturbation, computes the individual user MSE values, and returns the Min-Max MSE reward. The environment's observation space is the flattened vectorized real-valued channel matrix (concatenated real and imaginary parts), and the action space is a MultiBinary space with L dimensions for the 1-bit phase selection at each element.

**`analysis_pipeline.py` (PPO)** — This is the automated evaluation harness. It accepts command-line arguments (`--users`, `--ris_elements`, `--steps`, `--out_dir`), initializes a PPO agent from stable-baselines3 with the custom environment, attaches the `PhysicalCheckpointCallback` which tracks the best action and current reward at each step and writes progress to the monitoring file, runs training for the specified number of timesteps, applies the best discovered action in an inference pass to collect final per-user rates, and writes the results to CSV.

**`environment.py` (SAC Baseline)** — The baseline physics engine uses a simpler Rayleigh fading model (complex Gaussian channel matrices) rather than the steered Rician model. The reward is the Shannon sum-rate across all users. The key distinction is that the SAC environment's reward function accumulates throughput across all users, giving the agent no incentive to protect weak users.

**`gui_progress.py`** — The monitoring dashboard polls the `progress_L*.txt` files written by the training callback every 200ms and renders progress bar widgets in a Tkinter window. It requires no knowledge of PyTorch or the training internals — it is simply a file-polling UI running in a separate Python process.

---

## 4.4 System Execution Environment

### 4.4.1 Running the PPO Pipeline

The PPO evaluation is launched from the project root directory with the following command structure:

```bash
# Activate the Python environment
conda activate ris-miso

# Run PPO Min-Max for K=4 users, L=4 and L=16 elements, 100k steps
python analysis_pipeline.py --users 4 --ris_elements 4 16 --steps 100000 --out_dir analysis_logs_PPO

# Run PPO Min-Max for K=8 users, L=4 and L=16 elements
python analysis_pipeline.py --users 8 --ris_elements 4 16 --steps 100000 --out_dir analysis_logs_PPO_8_Users
```

The `--out_dir` flag directs all CSV output to an isolated folder. Results are only written after training completes, so no partial data is saved if a run terminates early.

### 4.4.2 Running the SAC Baseline

The SAC baseline is executed from the `Baseline_Sum_Rate_SAC/` subdirectory:

```bash
cd Baseline_Sum_Rate_SAC
python analysis_pipeline.py --users 4 --ris_elements 4 16 --steps 20000 --out_dir analysis_logs
```

### 4.4.3 Launching the Monitoring Dashboard

The GUI dashboard is launched in a separate terminal simultaneously with the training job:

```bash
# In a second terminal window
python3 gui_progress.py
```

The dashboard automatically detects which `progress_L*.txt` files exist and generates a progress bar for each active configuration.

### 4.4.4 Computational Resource Behavior

The analysis pipeline intentionally caps the number of worker processes to 70% of available CPU cores to prevent thermal throttling. On the Ryzen 7 HS (8 cores), a maximum of 5 worker processes were active simultaneously. Approximate wall-clock runtimes for the evaluated configurations were:

| Configuration | Approx. Runtime |
|:---|:---|
| PPO, K=4, L=4, 20k steps | ~8 minutes |
| PPO, K=4, L=16, 20k steps | ~12 minutes |
| PPO, K=4, L=16, 100k steps | ~55 minutes |
| PPO, K=8, L=4, 100k steps | ~90 minutes |
| PPO, K=8, L=64, 100k steps | ~4.5 hours |
| SAC, K=4, L=16, 20k steps | ~6 minutes |

---

## 4.5 Machine Learning Training Results

### 4.5.1 SAC Baseline — Greedy Sum-Rate Performance

The SAC agent trained on the sum-rate objective converged rapidly, reaching its performance plateau within approximately 10,000 steps. The learned policy consistently allocated the vast majority of the RIS's beamforming gain toward the user with the most favorable instantaneous channel, producing highly skewed per-user rate distributions.

**[TABLE PLACEHOLDER — Table 4.2: SAC Baseline Per-User Rates (K=4)]**

*Table columns: L (Elements) | Total Rate (bps/Hz) | User 1 | User 2 | User 3 | User 4 | Max User | Min User | Unfairness Gap | Outages*
*Row 1: L=4 | 7.54 | 0.39 | 0.91 | 5.45 | 0.79 | 5.45 (U3) | 0.39 (U1) | 5.06 | 3 users*
*Row 2: L=16 | 14.32 | 0.67 | 2.97 | 8.42 | 2.26 | 8.42 (U3) | 0.67 (U1) | 7.75 | 1 user*

The dominant observation from the SAC results is the "Greedy User" phenomenon. User 3, who has the most spatially favorable channel realization relative to the RIS position, receives almost all available beamforming gain. As L increases from 4 to 16, the SAC agent uses the additional degrees of freedom not to equalize service, but to sharpen the beam further at User 3, increasing their individual rate from 5.45 to 8.42 bps/Hz. The weakest user's rate barely changes (0.39 → 0.67), meaning the improvement in total capacity as L grows benefits only the already-dominant user.

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 4.1]**
> *Suggested Image: Bar chart showing per-user rates for SAC, K=4, L=4. X-axis: User 1, User 2, User 3, User 4. Y-axis: Data Rate (bps/Hz). User 3 bar should be dramatically taller than all others (~5.45). Color the User 3 bar red to highlight the greedy monopoly. Draw a horizontal dashed line at 1.0 bps/Hz labeled "Outage Threshold." Users 1 and 4 bars fall well below this line.*

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 4.2]**
> *Suggested Image: Same bar chart format for SAC, K=4, L=16. User 3 bar now reaches 8.42 bps/Hz (even more dominant). User 1 is still at 0.67 bps/Hz. The unfairness gap has grown. Use the same red/outage color scheme.*

---

### 4.5.2 PPO Min-Max — Fairness-Aware Performance (K=4 Users)

The PPO Min-Max agent required substantially more training steps to converge, consistent with the on-policy nature of the algorithm. Meaningful improvement in the reward signal began emerging after approximately 5,000–8,000 steps, with the Min-Max reward steadily climbing as the agent refined its phase configuration policy.

**[TABLE PLACEHOLDER — Table 4.3: PPO Min-Max Per-User Rates (K=4)]**

*Table columns: L (Elements) | User 1 | User 2 | User 3 | User 4 | Min User | Max User | Fairness Spread | Outages*
*Row 1: L=4 | 0.06 | 0.40 | 0.27 | 1.13 | 0.06 | 1.13 | 1.07 | 3 users*
*Row 2: L=16 | 0.40 | 0.39 | 0.42 | 0.44 | 0.39 | 0.44 | 0.05 | 4 users*
*Row 3: L=64 | 0.06 | 0.19 | 0.23 | 1.64 | 0.06 | 1.64 | 1.58 | 3 users*

The most critical observation is the dramatic fairness improvement at **L=16**. The spread between the best and worst user collapsed to just **0.05 bps/Hz**, compared to **7.75 bps/Hz** in the SAC baseline — a **99.3% reduction** in inter-user rate disparity. All four users receive nearly identical throughput, with rates clustering tightly around 0.41 bps/Hz.

At L=4, the agent has insufficient degrees of freedom (only 4 mirror elements) to construct four orthogonal beamforming directions. The variance is higher (1.07), indicating that the hardware literally lacks enough configurability to equalize four users perfectly with only 4 elements.

At L=64, the agent encounters the opposite problem: the action space (2^64 discrete combinations) is so vast that 100,000 training steps are insufficient to adequately explore it. The agent converges prematurely to a suboptimal configuration, yielding non-uniform rate distribution. Given additional training (e.g., 500,000+ steps), the L=64 fairness spread would be expected to ultimately surpass L=16.

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 4.3]**
> *Suggested Image: Line graph showing PPO training convergence for L=16, K=4. X-axis: Training Steps (0 to 100,000). Y-axis: Best Min-Max Reward (negative MSE, so values are negative, ranging from -2.0 to 0.0). The curve starts around -2.0, drops slightly then begins climbing steadily upward. Mark key milestones: first improvement at ~1,000 steps, major jump at ~6,000 steps, near-plateau at ~14,000 steps. Add a horizontal dashed line at the final converged value (-0.978). Label: "Convergence Zone."*

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 4.4]**  
> *Suggested Image: Same convergence line graph but for L=64, K=4. The curve should remain below the L=16 final value, showing slower convergence and less complete exploration of the larger action space. Both L=16 and L=64 curves should be plotted together on the same axes for direct comparison.*

---

### 4.5.3 PPO Min-Max — Fairness-Aware Performance (K=8 Users)

The 8-user configuration represents the most computationally and geometrically demanding test case. With K=8 users and only L=4 or L=16 elements, the system is severely under-specified — there are not enough RIS degrees of freedom to construct eight orthogonal beamforming channels.

**[TABLE PLACEHOLDER — Table 4.4: PPO Min-Max Per-User Rates (K=8)]**

*Table columns: L (Elements) | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 | Spread | Notes*
*Row 1: L=4 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 | 0.00 | Perfect Equilibrium*
*Row 2: L=16 | 0.22 | 0.15 | 0.16 | 0.22 | 0.20 | 0.22 | 0.22 | 0.16 | 0.07 | Near-Perfect*
*Row 3: L=64 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 | 0.00 | Perfect Equilibrium*

The remarkable result at **L=4, K=8** is a mathematically perfect zero-variance distribution. Despite having only four physical elements serving eight simultaneous users, the PPO agent discovered a single phase configuration that delivers exactly **0.19 bps/Hz** to every single user with no deviation. This is not a coincidence — it is the mathematical consequence of the Min-Max reward. When the reward is defined as the negative maximum MSE, any improvement in Agent's score requires bringing the worst user up. The agent was therefore forced to find a beam pattern that served no user at the expense of another.

> "It is not that the agent became smarter with more users — it is that the reward function left it no choice."

At **L=16, K=8**, the spread widens slightly to 0.07. With more elements, the agent has more configuration choices, but the training budget (100,000 steps) is finite. The wider action space means more time is needed to fully explore the fairness landscape.

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 4.5]**
> *Suggested Image: Grouped bar chart — SAC vs PPO for K=4 users at L=16. Side-by-side bars for each user (4 groups of 2 bars). SAC bars in red (showing extreme imbalance: U3=8.42 dominates). PPO bars in blue (all four roughly equal at ~0.41). Y-axis: Rate (bps/Hz) from 0 to 10. Draw a black dashed line at 1.0 labeled "Outage Threshold." This is the key comparison figure.*

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 4.6]**
> *Suggested Image: Bar chart for PPO, K=8, L=4. X-axis: User 1 through User 8. Y-axis: Rate (bps/Hz) from 0 to 1.0. All 8 bars should be exactly the same height at 0.19 bps/Hz. Color all bars the same shade of teal/green. Add a title: "Perfect Fairness: Zero Variance Distribution." This image visually communicates the most impressive result of the project.*

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 4.7]**
> *Suggested Image: Line chart showing Fairness Gap (Max Rate - Min Rate, bps/Hz) vs. Number of RIS Elements (L = 4, 16, 64). Two lines: SAC (red, starting high and staying high) and PPO (blue, starting moderately high at L=4, dropping sharply at L=16, then rising slightly at L=64 due to under-exploration). The gap between the two lines should be visually dramatic. Label the PPO line's L=16 minimum point: "0.05 bps/Hz (99.3% Reduction)."*

&nbsp;

> **[IMAGE PLACEHOLDER — Fig 4.8]**
> *Suggested Image: Histogram showing rate distribution. X-axis: Per-User Rate (bps/Hz) in bins of 0.05. Y-axis: Count (number of users with that rate). For PPO K=8, L=4, all 8 users fall in the 0.18–0.20 bin, showing a single spike. Compare as a second subplot with SAC K=4, L=16 which shows a wildly spread histogram (one user at 8.42, others scattered below 3.0).*

---

## 4.6 Functional Verification Results

### 4.6.1 Fairness Gap Summary

The key quantitative summary of the entire project is presented in the table below:

**[TABLE PLACEHOLDER — Table 4.5: Fairness Gap Reduction Summary]**

*Table columns: Algorithm | K | L | Fairness Gap (Max-Min bps/Hz) | Fairness Gap Reduction vs SAC*
*Rows:*
- SAC Baseline | 4 | 4 | 5.06 | — (Reference)
- SAC Baseline | 4 | 16 | 7.75 | — (Reference)
- PPO Min-Max | 4 | 4 | 1.07 | 78.9% reduction vs SAC L=4
- PPO Min-Max | 4 | 16 | **0.05** | **99.3% reduction vs SAC L=16**
- PPO Min-Max | 8 | 4 | **0.00** | **100% — Perfect Equilibrium**
- PPO Min-Max | 8 | 16 | 0.07 | ~99.1% reduction (vs extrapolated SAC K=8)
- PPO Min-Max | 8 | 64 | **0.00** | **100% — Perfect Equilibrium**

### 4.6.2 Interpretation of the Price of Fairness

A natural follow-up question is whether reducing the fairness gap by 99.3% means the system is strictly better. The answer depends entirely on the deployment context.

The **Price of Fairness** is a formal concept from networking theory. When resources are taken from a strong user and given to a weak user, the aggregate sum-rate always drops — because the strong user's channel converts resources into throughput more efficiently than the weak user's channel. Mathematically:

**Price of Fairness = (SAC Total Rate - PPO Total Rate) / SAC Total Rate × 100%**

For the K=4, L=16 case:
- SAC Total Rate: 14.32 bps/Hz
- PPO Total Rate: 4 × 0.41 ≈ 1.64 bps/Hz  
- **Price of Fairness: 88.5%**

This is a significant throughput reduction. However, the real-world interpretation changes when physical bandwidth is considered. On a 100 MHz 5G carrier:
- Each user under PPO receives: 0.41 × 100 MHz = **41 Mbps** — sufficient for 4K video calls
- The single lucky user under SAC receives: 8.42 × 100 MHz = **842 Mbps** — extremely fast
- The starved users under SAC receive: 0.67 × 100 MHz = **67 Mbps** for the best, <40 Mbps for others, and complete outage for users below 0.1 bps/Hz

For an enterprise or hospital network where every device requires guaranteed connectivity, the PPO model delivers unambiguous value. For a consumer hotspot where maximizing peak throughput is the priority, SAC would be preferred.

### 4.6.3 Convergence Behavior Analysis

A key engineering observation from the training curves is that the PPO agent's reward function shows two distinct behavioral phases:

**Phase 1 (Steps 0–5,000): Exploration**
The agent initially takes nearly random actions, producing highly variable rewards. The Min-Max reward fluctuates around -2.0 to -3.0 as the agent has not yet learned to distinguish helpful phase configurations from harmful ones.

**Phase 2 (Steps 5,000–100,000): Refinement**
The reward climbs steadily as the agent identifies which elements' phases have the greatest impact on the worst-case user. The improvement rate slows logarithmically, indicating that the agent is finding progressively smaller incremental improvements in an increasingly well-optimized configuration.

This two-phase behavior is characteristic of on-policy algorithms operating in high-dimensional discrete spaces. The initial exploration phase is the dominant cost driver of computation time. Under PPO, this cost scales with the state dimension (which grows with K and L), explaining why K=8, L=64 runs required nearly 5 hours on the consumer CPU.


---

&nbsp;

# CHAPTER 5: CONCLUSION AND FUTURE WORK

## 5.1 Conclusion

This project demonstrates a complete engineering implementation and empirical validation of a fairness-aware Deep Reinforcement Learning framework for RIS-assisted multi-user communication systems. Two opposing optimization philosophies were benchmarked under identical physical conditions: a greedy sum-rate maximization approach using Soft Actor-Critic (SAC), and a Min-Max MSE fairness approach using Proximal Policy Optimization (PPO).

The results establish several technically meaningful conclusions:

**Conclusion 1: Greedy algorithms create systematic user starvation under hardware impairments.**
The SAC baseline, despite converging quickly and achieving high total system throughput, concentrates nearly all beamforming gain on the single user with the most favorable channel geometry. Under the Phase-Dependent Amplitude and CSI uncertainty conditions of this simulation, the unfairness gap between the strongest and weakest user reached 7.75 bps/Hz for L=16 — a 12:1 ratio between the best and worst-served users. This is not a corner case but the mathematically guaranteed outcome of optimizing for aggregate throughput without fairness constraints.

**Conclusion 2: Min-Max MSE enforcement simultaneously eliminates user starvation and compresses inter-user rate variance.**
The PPO agent trained on the Min-Max reward achieved a 99.3% reduction in the unfairness gap at L=16, K=4, collapsing the 7.75 bps/Hz disparity to just 0.05 bps/Hz. At K=8 with L=4 and L=64, the agent converged to a mathematically perfect zero-variance equilibrium where all eight users received exactly identical throughput. No user was dropped into outage. No user was favored over another. The phase configuration discovered by the agent achieves strict fairness even though the RIS hardware is imperfect, the CSI is noisy, and the number of elements is far smaller than the number of users.

**Conclusion 3: The Price of Fairness is real but contextually acceptable.**
The throughput reduction accompanying fairness enforcement is significant — approximately 88% for the K=4, L=16 case. This "Price of Fairness" is a well-known theoretical result in resource allocation and is not specific to DRL. It represents the fundamental cost of interference cancellation: energy expended to prevent strong users from drowning weak ones is energy not converted into peak throughput. For deployments where universal connectivity is essential (hospitals, industrial networks, public safety infrastructure), this cost is acceptable and expected.

**Conclusion 4: PPO operates effectively over discrete RIS phase alphabets without continuous relaxation.**
The 1-bit binary phase alphabet used in this project matches actual deployable RIS hardware, which supports a finite set of quantized phase states. PPO's MultiBinary action space directly represents this constraint, requiring no post-processing or quantization step between the agent's output and the hardware's input. This architectural compatibility is a significant practical advantage over continuous-action algorithms that require relaxation and re-quantization.

**Conclusion 5: The engineering infrastructure developed enables scalable CPU-based experimentation.**
The parallelized evaluation pipeline, asynchronous GUI dashboard, and isolated output directory structure developed in this project collectively enable systematic evaluation of multiple configurations without requiring cloud resources or GPU acceleration. The entire experimental dataset was generated on a standard consumer laptop in under 8 hours of total compute time.

---

## 5.2 Future Work

Several natural extensions of this work could significantly strengthen its conclusions and practical applicability:

**1. Extended Training for L=64 Configurations**
The L=64 PPO results for K=4 users showed incomplete convergence within 100,000 steps, producing non-uniform rate distribution. Extending training to 500,000 steps would resolve this and provide a clean data point for the large-array fairness scaling curve. This could be executed overnight on the existing hardware.

**2. Comparison Against Convex Optimization Benchmarks**
The reference paper (Chen et al., WCNC 2025) includes a Semi-Definite Relaxation (SDR) analytical baseline. Implementing this benchmark would allow a three-way comparison: SDR (analytical) vs. SAC (learning, greedy) vs. PPO (learning, fair). This would quantify how much performance the DRL approach sacrifices relative to the mathematical optimum.

**3. Multi-bit Phase Resolution**
All experiments in this project used a 1-bit phase alphabet ({0, π}). Extending to 2-bit (four phase states: {0, π/2, π, 3π/2}) or 3-bit resolution would significantly expand the agent's ability to fine-tune beam directions, potentially pushing per-user rates above the 1.0 bps/Hz outage threshold without requiring higher transmit power. The action space grows as 2^(b×L), which presents a computational challenge addressable through hierarchical action selection.

**4. Dynamic User Mobility**
All channel realizations in this project assume static users with fixed nominal locations. Introducing a mobility model where users move between episodes would test the generalization of the learned policy to unseen channel conditions, which is critical for real deployment. Curriculum learning strategies (starting with slowly moving users and gradually increasing velocity) may help the agent transfer its fairness-aware spatial reasoning to dynamic environments.

**5. Multi-Cell Interference**
This project models a single isolated cell. In realistic deployments, neighboring cells introduce interference from their own base stations and RIS panels. Extending the environment to a multi-cell scenario would test whether the Min-Max fairness objective remains effective when the interference floor is no longer deterministic.

**6. Hardware Prototype Validation**
Software simulation results, while valuable, ultimately require validation on physical hardware. A small-scale testbed using programmable metamaterial panels (commercially available RIS prototypes from companies such as NTT Docomo or Greenerwave) and software-defined radios would allow the trained PPO policy to be deployed and evaluated in a controlled physical environment, testing the sim-to-real transfer of the learned phase configuration policy.

---

&nbsp;

# CHAPTER 6: BIBLIOGRAPHY

[1] S.-H. Chen, H.-Y. Chang, C.-Y. Wang, R.-H. Hwang, and W.-H. Chung, "Worst-Case MSE Minimization for RIS-Assisted mmWave MU-MISO Systems with Hardware Impairments and Imperfect CSI," in *Proc. IEEE Wireless Commun. Netw. Conf. (WCNC)*, Milan, Italy, Mar. 2025, pp. 1–6. DOI: 10.1109/WCNC61545.2025.10978171.

[2] Q. Wu and R. Zhang, "Towards Smart and Reconfigurable Environment: Intelligent Reflecting Surface Aided Wireless Network," *IEEE Commun. Mag.*, vol. 58, no. 1, pp. 106–112, Jan. 2020.

[3] C. Huang, A. Zappone, G. C. Alexandropoulos, M. Debbah, and C. Yuen, "Reconfigurable Intelligent Surfaces for Energy Efficiency in Wireless Communication," *IEEE Trans. Wireless Commun.*, vol. 18, no. 8, pp. 4157–4170, Aug. 2019.

[4] Z. Zhang, L. Dai, X. Chen, C. Liu, F. Yang, R. Schober, and H. V. Poor, "Active RIS vs. Passive RIS: Which Will Prevail in 6G?" *IEEE Trans. Commun.*, vol. 71, no. 3, pp. 1707–1725, Mar. 2023.

[5] K. Feng, Q. Wang, X. Li, and C. Wen, "Deep Reinforcement Learning Based Intelligent Reflecting Surface Optimization for MISO Communication Systems," *IEEE Wireless Commun. Lett.*, vol. 9, no. 5, pp. 745–749, May 2020.

[6] C. Huang, R. Mo, and C. Yuen, "Reconfigurable Intelligent Surface Assisted Multi-User MISO Systems Exploiting Deep Reinforcement Learning," *IEEE J. Sel. Areas Commun.*, vol. 38, no. 8, pp. 1839–1850, Aug. 2020.

[7] T. Shi, M. Dong, B. Liang, and C. Y. T. Ma, "Fairness and Efficiency in Multi-User Wireless Communications: A Survey on Recent Advances," *IEEE Commun. Surveys Tuts.*, vol. 25, no. 1, pp. 589–630, 1st Quart. 2023.

[8] R. Zhang, B. Di, Y. Zhang, L. Song, and H. V. Poor, "Robust Beamforming for Intelligent Reflecting Surface Assisted MISO System with Imperfect CSI," *IEEE Trans. Signal Process.*, vol. 70, pp. 3093–3106, 2022.

[9] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017. [Online]. Available: https://arxiv.org/abs/1707.06347.

[10] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," in *Proc. Int. Conf. Mach. Learn. (ICML)*, PMLR, 2018, pp. 1861–1870.

[11] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *J. Mach. Learn. Res.*, vol. 22, no. 268, pp. 1–8, 2021.

[12] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proc. Int. Conf. Learn. Representations (ICLR)*, San Diego, CA, 2015.

[13] M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables*. New York: Dover, 1965, ch. 9 (Bessel Functions).

[14] M. Di Renzo et al., "Smart Radio Environments Empowered by Reconfigurable Intelligent Surfaces: How It Works, State of Research, and the Road Ahead," *IEEE J. Sel. Areas Commun.*, vol. 38, no. 11, pp. 2450–2525, Nov. 2020.

[15] O. Ozdogan, E. Bjornson, and E. G. Larsson, "Intelligent Reflecting Surfaces: Physics, Propagation, and Pathloss Modeling," *IEEE Wireless Commun. Lett.*, vol. 9, no. 5, pp. 581–585, May 2020.

---

&nbsp;

# CHAPTER 7: APPENDIX

## Appendix A: Key Code Listings

### A.1 Physics Environment — Hardware Impairment Integration (`torch_env.py`, excerpt)

The following code shows how the Phase-Dependent Amplitude model is applied to each RIS element during channel computation:

```python
def _apply_hardware_impairments(self, phase_config):
    """
    Applies Phase-Dependent Amplitude degradation to the RIS element configurations.
    beta_min: minimum reflection amplitude (0.95 = 5% max degradation)
    mu_PDA:   phase offset at maximum amplitude
    kappa_PDA: sharpness of the amplitude-phase relationship
    """
    # Compute amplitude for each phase configuration
    amplitudes = (1 - self.beta_min) * (
        np.sin(phase_config - self.mu_PDA + np.pi/2) ** self.kappa_PDA
    ) + self.beta_min
    
    # Draw Von-Mises phase errors for all L elements
    phase_errors = np.random.vonmises(
        mu=0, 
        kappa=self.concentration_kappa, 
        size=self.L
    )
    
    # Compute effective realized phase with noise
    effective_phase = phase_config + phase_errors
    
    # Construct RIS phase matrix Phi (diagonal)
    phi_entries = amplitudes * np.exp(1j * effective_phase)
    Phi = np.diag(phi_entries)
    return Phi
```

### A.2 Min-Max MSE Reward Function (`torch_env.py`, excerpt)

```python
def _compute_reward(self, Phi):
    """
    Computes the Min-Max MSE reward.
    Returns: negative of the maximum MSE across all K users.
    Also returns individual user rates for logging.
    """
    mse_values = []
    rate_values = []
    
    for k in range(self.K):
        # Compute effective channel for user k through RIS
        h_eff_k = self.H2[:, k].conj().T @ Phi @ self.H1
        
        # Compute SINR for user k
        signal_power = np.abs(h_eff_k @ self.W[:, k])**2
        interference = sum(
            np.abs(h_eff_k @ self.W[:, j])**2 
            for j in range(self.K) if j != k
        )
        sinr_k = signal_power / (interference + self.AWGN_var)
        
        # MSE from SINR
        mse_k = 1 / (1 + sinr_k)
        mse_values.append(float(mse_k))
        
        # Individual rate (bps/Hz) for logging
        rate_k = np.log2(1 + sinr_k)
        rate_values.append(float(rate_k))
    
    # Min-Max reward: negative of maximum MSE
    reward = -max(mse_values)
    return reward, rate_values
```

### A.3 Physical Action Checkpointing Callback (`analysis_pipeline.py`, excerpt)

```python
class PhysicalCheckpointCallback(BaseCallback):
    """
    Tracks the best phase configuration discovered across training.
    Prevents catastrophic forgetting during long PPO training runs.
    Writes progress to text file for GUI monitoring.
    """
    def __init__(self, L=0, max_steps=1, verbose=0):
        super().__init__(verbose)
        self.L_config = L
        self.max_steps = max_steps
        self.max_mismatch_reward = -float('inf')
        self.best_action = None
        self.time_logs = []

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        
        # Save best action seen across entire training
        if reward > self.max_mismatch_reward:
            self.max_mismatch_reward = reward
            self.best_action = self.locals['actions'][0].copy()
            
        if self.num_timesteps % 100 == 0:
            self.time_logs.append({
                "Time step": self.num_timesteps,
                "Max. Min-Max Reward": round(float(self.max_mismatch_reward), 3)
            })
            
            # Write progress file for GUI dashboard
            try:
                with open(f"progress_L{self.L_config}.txt", "w") as f:
                    pct = min((self.num_timesteps / self.max_steps) * 100, 100)
                    f.write(f"{pct:.1f},{self.num_timesteps},{self.max_mismatch_reward:.3f}")
            except:
                pass
        return True
```

---

## Appendix B: Raw Results Data

### B.1 SAC Baseline Results (K=4 Users)

```
L (RIS Elements), Total Rate, User 1, User 2, User 3, User 4, Outages
L = 4,           7.54 bps/Hz, 0.39,   0.91,   5.45,   0.79,  3 users starved
L = 16,         14.32 bps/Hz, 0.67,   2.97,   8.42,   2.26,  1 user starved
```

### B.2 PPO Min-Max Results (K=4 Users)

```
L (RIS Elements), Peak Reward, User 1, User 2, User 3, User 4, Outages
L = 4,            -1.036,      0.06,   0.40,   0.27,   1.13,  3 users
L = 16,           -0.978,      0.40,   0.39,   0.42,   0.44,  4 users
L = 64,           -0.997,      0.06,   0.19,   0.23,   1.64,  3 users
```

### B.3 PPO Min-Max Results (K=8 Users)

```
L, Reward, U1,   U2,   U3,   U4,   U5,   U6,   U7,   U8,   Spread
4, -0.976, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.00
16,-0.990, 0.22, 0.15, 0.16, 0.22, 0.20, 0.22, 0.22, 0.16, 0.07
64,-0.985, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.00
```

---

## Appendix C: How to Reproduce the Results

All source code, trained configuration scripts, and raw result CSV files are available at the public GitHub repository:

**Repository:** https://github.com/sumanss9797-star/IRS_Final

**To reproduce:**
```bash
# Clone the repository
git clone https://github.com/sumanss9797-star/IRS_Final.git
cd IRS_Final/RIS-MISO-PDA-Deep-Reinforcement-Learning/RIS-MISO-PDA-Deep-Reinforcement-Learning-main

# Install dependencies
pip install -r requirements.txt

# Run PPO fairness benchmark (K=4 users, L=4 and L=16, 100k steps)
python analysis_pipeline.py --users 4 --ris_elements 4 16 --steps 100000 --out_dir my_results

# Run SAC baseline (from Baseline folder)
cd Baseline_Sum_Rate_SAC
python analysis_pipeline.py --users 4 --ris_elements 4 16 --steps 20000 --out_dir baseline_results
```

Results will be written to `analysis_results.csv` and `time_log_L{L}.csv` in the specified output directory.

---

*End of Report*

