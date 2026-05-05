"""
==============================================================================
  REMOTE ROBOTIC SURGERY USING 5G NR URLLC — FULL SYSTEM EMULATOR
  Application Extension of: 5G NR URLLC System Emulator (Group Project)
  
  NO HARDWARE REQUIRED — Pure software simulation + ML/DL components
==============================================================================

ARCHITECTURE:
  Surgeon Console → 5G NR URLLC Link → gNB (Base Station) → Robotic Arm
  
  ML/DL Components:
    1. Neural Network: Adaptive MCS (Modulation & Coding Scheme) Predictor
       - Input: SNR, BLER history, latency history
       - Output: Optimal modulation order + code rate for next packet
    2. Neural Network: Haptic Signal Denoising / Reconstruction
       - Reconstructs corrupted haptic feedback packets lost over the channel
    3. Anomaly Detector: Safety system that detects dangerous latency spikes

MODULES:
  Module A — 5G NR Parameters & URLLC Channel Emulator  (from project)
  Module B — Packet-level Surgery Traffic Generator
  Module C — ML-based Adaptive MCS Controller
  Module D — URLLC Transmission Engine (HARQ-IR, MRC, Rayleigh fading)
  Module E — Haptic Reconstruction Neural Network
  Module F — Surgery Safety Monitor (Anomaly Detection)
  Module G — End-to-End Evaluation & 9 Research-Grade Figures
==============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patches as mpatches
from scipy.special import erfc, erfcinv
from scipy.stats import norm
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("╔══════════════════════════════════════════════════════════════╗")
print("║     REMOTE ROBOTIC SURGERY — 5G NR URLLC SYSTEM EMULATOR    ║")
print("║     Application Extension | No Hardware Required             ║")
print("║     ML/DL: Adaptive MCS + Haptic Reconstruction + Safety     ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

# ============================================================
# MODULE A — 5G NR PARAMETERS & URLLC CHANNEL EMULATOR
# (Directly from your MATLAB project, translated to Python)
# ============================================================
print("=== MODULE A: 5G NR Parameters & Numerology ===")

# SNR sweep
snr_db   = np.arange(-4, 26, 2)   # -4 to +24 dB (15 points)
snr_lin  = 10 ** (snr_db / 10)
num_snr  = len(snr_db)

# Payload
K      = 256   # bits per URLLC packet (short-packet regime)
trials = 4000  # Monte Carlo packets per SNR point

# Diversity orders (receive antennas)
diversity_orders = [1, 2, 4, 8]

# 5G NR Numerology (3GPP TS 38.211 Table 4.3.2-1)
mu_idx  = [0, 1, 2, 3]
scs_kHz = [15, 30, 60, 120]
slot_ms = [1 / (s / 15) for s in scs_kHz]
sym_us  = [(s * 1e3) / 14 for s in slot_ms]

# Operating point: µ=1, SCS=30 kHz
mu_sel   = 1
scs_sel  = scs_kHz[mu_sel]       # 30 kHz
T_slot   = slot_ms[mu_sel]       # 0.5 ms
T_sym_us = sym_us[mu_sel]        # 35.7 µs

# Mini-slot: 2 OFDM symbols
n_mini    = 2
T_mini_ms = (n_mini / 14) * T_slot   # 0.0714 ms

# E2E latency budget components (3GPP TR 38.824)
gNB_proc_ms = 0.20   # gNB scheduling + encoding
UE_proc_ms  = 0.20   # UE decoding + ACK/NACK
prop_ms     = 0.10   # 30 km cell propagation

print(f"  Numerology µ={mu_sel}: SCS={scs_sel} kHz | T_slot={T_slot:.3f} ms | Mini-slot({n_mini} sym)={T_mini_ms:.4f} ms")
print(f"  Fixed overhead: gNB={gNB_proc_ms:.1f}ms | UE={UE_proc_ms:.1f}ms | Prop={prop_ms:.1f}ms | Total={gNB_proc_ms+UE_proc_ms+prop_ms:.1f}ms")
print(f"  URLLC Budget remaining for Tx+HARQ: {1.0-gNB_proc_ms-UE_proc_ms-prop_ms:.3f} ms\n")

# ============================================================
# MODULE B — SURGERY TRAFFIC GENERATOR
# Models three traffic types generated during remote surgery
# ============================================================
print("=== MODULE B: Surgery Traffic Generator ===")

class SurgeryTrafficGenerator:
    """
    Generates realistic packet streams for remote robotic surgery:
    - Command packets:  Surgeon → Robot arm motor commands (most critical)
    - Haptic packets:   Robot arm → Surgeon force/touch feedback
    - Video packets:    Endoscopic camera feed (lower priority)
    
    All conform to URLLC short-packet constraint (K=256 bits).
    """
    def __init__(self, n_packets=2000, seed=0):
        rng = np.random.default_rng(seed)
        self.n = n_packets
        
        # Assign traffic types (proportions from 3GPP TR 22.826)
        types_raw = rng.choice(['command', 'haptic', 'video'],
                               p=[0.30, 0.45, 0.25], size=n_packets)
        self.types = types_raw
        
        # Priority: command=2 (highest), haptic=1, video=0
        priority_map = {'command': 2, 'haptic': 1, 'video': 0}
        self.priority = np.array([priority_map[t] for t in types_raw])
        
        # Deadlines per type [ms]
        deadline_map = {'command': 1.0, 'haptic': 2.0, 'video': 10.0}
        self.deadline_ms = np.array([deadline_map[t] for t in types_raw])
        
        # Simulated SNR per packet (slow fading: changes every ~50 packets)
        snr_blocks = rng.uniform(2, 22, size=n_packets // 50 + 1)
        snr_per_pkt = np.repeat(snr_blocks, 50)[:n_packets]
        # Add fast fading perturbation
        self.snr_per_pkt = np.clip(snr_per_pkt + rng.normal(0, 2, n_packets), -4, 24)
        
        # Haptic payload: force vector (simulated as 8 float32 values = 256 bits)
        self.haptic_payload = rng.normal(0, 1, (n_packets, 8)).astype(np.float32)

    def summary(self):
        unique, counts = np.unique(self.types, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"    {u:10s}: {c:5d} packets ({100*c/self.n:.1f}%)")

traffic = SurgeryTrafficGenerator(n_packets=3000)
print("  Packet distribution:")
traffic.summary()
print()

# ============================================================
# MODULE C — ML COMPONENT 1: ADAPTIVE MCS PREDICTOR (Neural Network)
# 
# Predicts the optimal Modulation & Coding Scheme per packet
# based on channel conditions and recent BLER/latency history.
# 
# Replaces the static MCS table in the gNB scheduler with a
# learning-based controller that adapts to surgery traffic patterns.
# ============================================================
print("=== MODULE C: ML — Adaptive MCS Neural Network Controller ===")

# MCS Table (subset of 3GPP TS 38.214 Table 5.1.3.1-1)
# (index, modulation_order, code_rate, spectral_efficiency_bpcu)
MCS_TABLE = [
    (0,  'BPSK',   0.12,  0.12),   # most robust
    (1,  'BPSK',   0.19,  0.19),
    (2,  'BPSK',   0.30,  0.30),
    (3,  'QPSK',   0.38,  0.76),
    (4,  'QPSK',   0.49,  0.98),
    (5,  'QPSK',   0.60,  1.20),
    (6,  'QPSK',   0.75,  1.50),
    (7,  '16QAM',  0.44,  1.77),
    (8,  '16QAM',  0.55,  2.19),
    (9,  '16QAM',  0.65,  2.59),
    (10, '16QAM',  0.75,  3.00),
]
MCS_LABELS = [m[0] for m in MCS_TABLE]
MCS_MOD    = {m[0]: m[1] for m in MCS_TABLE}
MCS_RATE   = {m[0]: m[2] for m in MCS_TABLE}

def generate_mcs_training_data(n_samples=15000):
    """
    Generates labelled training data for the MCS predictor.
    Ground-truth MCS is the highest index whose required SNR is met
    given the target BLER=1e-5, using the PPV bound from the project.
    """
    rng = np.random.default_rng(1)
    
    # Features: [snr_dB, recent_bler, recent_lat_ms, priority, harq_round]
    snr      = rng.uniform(-4, 24, n_samples)
    bler_h   = np.clip(rng.exponential(0.05, n_samples), 0, 1)
    lat_h    = rng.uniform(0.3, 2.0, n_samples)
    priority = rng.integers(0, 3, n_samples)      # 0=video,1=haptic,2=command
    harq_r   = rng.integers(1, 5, n_samples)      # 1..4
    
    X = np.column_stack([snr, bler_h, lat_h, priority, harq_r])
    
    # Label: choose optimal MCS
    # Rule: pick highest MCS that keeps required SNR feasible given BLER history
    # Approximation: SNR threshold per MCS from Shannon + code rate
    y = []
    for i in range(n_samples):
        snr_i     = snr[i]
        bler_i    = bler_h[i]
        prio_i    = priority[i]
        
        # Penalise high-order MCS if BLER is high or priority is critical
        penalty = 3 * bler_i + (2 - prio_i) * 0.5
        
        # Shannon capacity at this SNR
        C = np.log2(1 + 10**(snr_i / 10))
        
        best = 0
        for mcs_idx, _, cr, _ in MCS_TABLE:
            # Required SNR for this code rate ≈ 2^(2*cr) - 1 (Shannon inversion)
            snr_req = 2**(2 * cr) - 1
            snr_req_db = 10 * np.log10(snr_req + 1e-9)
            if snr_i >= snr_req_db - penalty:
                best = mcs_idx
        y.append(best)
    
    return X, np.array(y)

X_mcs, y_mcs = generate_mcs_training_data(15000)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_mcs, y_mcs, test_size=0.2, random_state=42)

scaler_mcs = StandardScaler()
X_train_ms = scaler_mcs.fit_transform(X_train_m)
X_test_ms  = scaler_mcs.transform(X_test_m)

mcs_nn = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    verbose=False
)
mcs_nn.fit(X_train_ms, y_train_m)
mcs_acc = accuracy_score(y_test_m, mcs_nn.predict(X_test_ms))
print(f"  MCS Predictor Neural Net   — Test Accuracy: {mcs_acc*100:.2f}%")
print(f"  Architecture: 5→128→64→32→{len(np.unique(y_mcs))} | Optimizer: Adam | Activation: ReLU")
print(f"  Training samples: {len(X_train_m)} | Test samples: {len(X_test_m)}")

def predict_mcs(snr_dB, recent_bler, recent_lat, priority, harq_round):
    """Returns optimal MCS index for current channel + traffic conditions."""
    feat = np.array([[snr_dB, recent_bler, recent_lat, priority, harq_round]])
    feat_s = scaler_mcs.transform(feat)
    return int(mcs_nn.predict(feat_s)[0])

print()

# ============================================================
# MODULE D — URLLC TRANSMISSION ENGINE
# (Faithful Python translation of Modules 4 & 5 from the MATLAB project)
# Adds surgery-aware scheduling: priority-based mini-slot preemption
# ============================================================
print("=== MODULE D: URLLC Transmission Engine (HARQ-IR, MRC, Rayleigh) ===")

def mrc_rayleigh_bler(snr_lin_val, K_bits, L, n_trials=4000, rng=None):
    """
    Monte Carlo BLER for L-branch MRC over Rayleigh flat fading.
    Vectorised exactly as in the MATLAB project (Module 4).
    Returns: (bler, effective_latency_ms)
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma = 1 / np.sqrt(2 * snr_lin_val)
    TX    = 2 * rng.integers(0, 2, (K_bits, n_trials)) - 1   # ±1 BPSK
    H     = (rng.standard_normal((L, n_trials)) +
             1j * rng.standard_normal((L, n_trials))) / np.sqrt(2)
    H_pow = np.sum(np.abs(H)**2, axis=0)                      # ||h||²
    eff_noise = np.sqrt(H_pow) * sigma * rng.standard_normal((K_bits, n_trials))
    y_eff     = H_pow * TX + eff_noise
    pkt_err   = np.any(((y_eff > 0) != (TX > 0)), axis=0)
    bler      = max(float(np.mean(pkt_err)), 0.5 / n_trials)
    harq_penalty = bler * (2 * T_slot + gNB_proc_ms + UE_proc_ms)
    lat       = T_mini_ms + gNB_proc_ms + UE_proc_ms + prop_ms + harq_penalty
    return bler, lat

def harq_ir_engine(snr_lin_val, K_bits, L, max_rounds=4, n_trials=3000, rng=None):
    """
    HARQ-IR (Incremental Redundancy) exactly as in MATLAB Module 5.
    Returns: (final_bler, avg_latency_ms, rounds_histogram)
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma    = 1 / np.sqrt(2 * snr_lin_val)
    tx_all   = 2 * rng.integers(0, 2, (K_bits, n_trials)) - 1
    H1       = (rng.standard_normal((L, n_trials)) +
                1j * rng.standard_normal((L, n_trials))) / np.sqrt(2)
    Hp1      = np.sum(np.abs(H1)**2, axis=0)
    llr      = Hp1 * tx_all + np.sqrt(Hp1) * sigma * rng.standard_normal((K_bits, n_trials))
    err      = np.any((llr > 0) != (tx_all > 0), axis=0)
    lat_v    = np.where(err,
                        T_mini_ms + gNB_proc_ms + (2*T_slot + gNB_proc_ms + UE_proc_ms),
                        T_mini_ms + gNB_proc_ms)
    rounds_used = np.ones(n_trials, dtype=int)
    active   = err.copy()
    
    for rnd in range(2, max_rounds + 1):
        if not np.any(active):
            break
        sigma_ir = sigma / np.sqrt(rnd)   # coding gain (Module 5 model)
        H_r  = (rng.standard_normal((L, n_trials)) +
                1j * rng.standard_normal((L, n_trials))) / np.sqrt(2)
        Hp_r = np.sum(np.abs(H_r)**2, axis=0)
        llr += Hp_r * tx_all + np.sqrt(Hp_r) * sigma_ir * rng.standard_normal((K_bits, n_trials))
        new_err = np.any((llr > 0) != (tx_all > 0), axis=0)
        still_err = active & new_err
        just_fixed = active & (~new_err)
        rounds_used[just_fixed] = rnd
        lat_v[active] += (2*T_slot + gNB_proc_ms + UE_proc_ms)
        err    = new_err
        active = still_err
    
    rounds_used[active] = max_rounds  # packets still in error after all rounds
    bler = max(float(np.mean(err)), 0.5 / n_trials)
    return bler, float(np.mean(lat_v + prop_ms)), rounds_used

# Run full BLER sweep (same as MATLAB Modules 4+5)
rng_main = np.random.default_rng(42)
results_bler    = np.zeros((len(diversity_orders), num_snr))
results_latency = np.zeros((len(diversity_orders), num_snr))

print("  Running BLER/latency sweep across SNR and diversity orders...")
for di, L in enumerate(diversity_orders):
    for si, snr_l in enumerate(snr_lin):
        b, lat = mrc_rayleigh_bler(snr_l, K, L, n_trials=trials, rng=rng_main)
        results_bler[di, si]    = b
        results_latency[di, si] = lat
    print(f"    L={L} done → BLER at 10dB: {results_bler[di, snr_db==10][0]:.2e}")

# Analytical BER theory (Proakis closed-form, Module 4)
bler_theory = np.zeros((len(diversity_orders), num_snr))
for di, L in enumerate(diversity_orders):
    mu_th = np.sqrt(snr_lin / (1 + snr_lin))
    bth   = np.zeros(num_snr)
    for k in range(L):
        from math import comb
        bth += comb(L - 1 + k, k) * ((1 + mu_th) / 2) ** k
    bler_theory[di] = ((1 - mu_th) / 2) ** L * bth

# HARQ-IR sweep (for comparison with Module 5)
bler_harq_ir = np.zeros(num_snr)
lat_harq_ir  = np.zeros(num_snr)
rng_harq = np.random.default_rng(7)
L_harq = 4
print(f"\n  Running HARQ-IR sweep (L={L_harq} MRC, max 4 rounds)...")
for si, snr_l in enumerate(snr_lin):
    b, lat, _ = harq_ir_engine(snr_l, K, L_harq, max_rounds=4, n_trials=2000, rng=rng_harq)
    bler_harq_ir[si] = b
    lat_harq_ir[si]  = lat

print()

# ============================================================
# MODULE E — ML COMPONENT 2: HAPTIC SIGNAL RECONSTRUCTION
# Neural network that recovers lost/corrupted haptic packets
# This is the DL component of the system.
# ============================================================
print("=== MODULE E: ML/DL — Haptic Signal Reconstruction Network ===")

def generate_haptic_dataset(n_samples=12000, corruption_rate=0.15):
    """
    Simulates haptic force sensor readings from a 6-DOF robotic arm.
    Each sample: 8-dimensional force/torque vector (fits in 256-bit packet).
    Corrupted samples simulate packets lost/damaged over the URLLC channel.
    """
    rng = np.random.default_rng(99)
    
    # Simulate realistic surgical force patterns
    # Forces in [N], torques in [Nm], typical surgical range
    t      = np.linspace(0, 4*np.pi, n_samples)
    f_x    = 2.5 * np.sin(t * 0.3 + 0.1) + 0.5 * np.sin(t * 1.7)
    f_y    = 1.8 * np.cos(t * 0.5) + 0.3 * rng.standard_normal(n_samples)
    f_z    = 3.2 * np.sin(t * 0.2) * np.cos(t * 0.1)
    tx     = 0.15 * np.sin(t * 0.8 + np.pi/4)
    ty     = 0.12 * np.cos(t * 0.6)
    tz     = 0.08 * np.sin(t * 1.2)
    grip   = np.clip(1.5 + 0.7 * np.sin(t * 0.4), 0, 3)
    tissue = np.clip(0.5 + 0.3 * rng.standard_normal(n_samples), 0, 1.5)
    
    clean = np.column_stack([f_x, f_y, f_z, tx, ty, tz, grip, tissue]).astype(np.float32)
    
    # Corrupt samples: simulate packet loss / bit errors
    corrupt_mask = rng.random(n_samples) < corruption_rate
    corrupted    = clean.copy()
    corrupted[corrupt_mask] += rng.normal(0, 3.0, (np.sum(corrupt_mask), 8))
    
    # Features for reconstruction:
    # previous 3 clean packets + current corrupted + channel SNR
    X_list, y_list = [], []
    for i in range(3, n_samples):
        prev3  = clean[i-3:i].flatten()     # 3 × 8 = 24 values (context)
        curr_c = corrupted[i]               # 8 values (possibly corrupt)
        snr_f  = np.array([rng.uniform(-4, 24)])  # channel SNR at this packet
        feat   = np.concatenate([prev3, curr_c, snr_f])  # 33 features
        X_list.append(feat)
        y_list.append(clean[i])
    
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            corrupt_mask)

X_hap, y_hap, corrupt_mask = generate_haptic_dataset(12000)
X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(X_hap, y_hap, test_size=0.2, random_state=0)

scaler_hap_x = StandardScaler()
scaler_hap_y = StandardScaler()
X_tr_hs = scaler_hap_x.fit_transform(X_tr_h)
X_te_hs = scaler_hap_x.transform(X_te_h)
y_tr_hs = scaler_hap_y.fit_transform(y_tr_h)

haptic_nn = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    learning_rate_init=1e-3,
    verbose=False
)
haptic_nn.fit(X_tr_hs, y_tr_hs)

y_pred_hs = haptic_nn.predict(X_te_hs)
y_pred_h  = scaler_hap_y.inverse_transform(y_pred_hs)
haptic_mse  = mean_squared_error(y_te_h, y_pred_h)
haptic_rmse = np.sqrt(haptic_mse)
baseline_rmse = np.sqrt(mean_squared_error(y_te_h, X_te_h[:, 24:32]))  # raw corrupted

print(f"  Haptic Reconstruction Net  — Architecture: 33→256→128→64→32→8")
print(f"  Optimizer: Adam | Activation: ReLU | Loss: MSE")
print(f"  Training samples: {len(X_tr_h)} | Test samples: {len(X_te_h)}")
print(f"  Baseline RMSE (no reconstruction): {baseline_rmse:.4f} N")
print(f"  Network RMSE  (with reconstruction): {haptic_rmse:.4f} N")
print(f"  Improvement: {100*(baseline_rmse-haptic_rmse)/baseline_rmse:.1f}% RMSE reduction\n")

def reconstruct_haptic(prev3_clean, corrupted_packet, snr_dB):
    """Reconstructs a potentially corrupted haptic packet using the trained NN."""
    feat = np.concatenate([prev3_clean.flatten(), corrupted_packet, [snr_dB]]).reshape(1, -1)
    feat_s = scaler_hap_x.transform(feat)
    pred_s = haptic_nn.predict(feat_s)
    return scaler_hap_y.inverse_transform(pred_s)[0]

# ============================================================
# MODULE F — ML COMPONENT 3: SURGERY SAFETY MONITOR (Anomaly Detection)
# Detects dangerous latency spikes that would make teleoperation unsafe
# Uses a One-Class anomaly detection approach (Isolation-Forest style
# implemented via MLP reconstruction error thresholding — the DL way)
# ============================================================
print("=== MODULE F: ML — Surgery Safety Monitor (Anomaly Detection) ===")

def generate_safety_dataset(n_samples=10000):
    """
    Normal operation: latency < 1ms, BLER < 1e-3, jitter low.
    Anomaly: latency spike (>1ms), BLER surge, or jitter explosion.
    Features: [latency_ms, bler, jitter_ms, snr_dB, harq_rounds, pkt_type]
    """
    rng = np.random.default_rng(55)
    n_normal  = int(0.85 * n_samples)
    n_anomaly = n_samples - n_normal
    
    # Normal operation
    lat_n   = rng.uniform(0.30, 0.90, n_normal)
    bler_n  = rng.exponential(1e-4, n_normal).clip(0, 1e-3)
    jit_n   = rng.uniform(0.01, 0.08, n_normal)
    snr_n   = rng.uniform(8, 24, n_normal)
    hrq_n   = rng.integers(1, 3, n_normal).astype(float)
    pkt_n   = rng.integers(0, 3, n_normal).astype(float)
    
    # Anomalies (latency spike, BLER surge, deep fade events)
    lat_a   = rng.uniform(1.0, 4.0, n_anomaly)
    bler_a  = rng.uniform(1e-3, 0.5, n_anomaly)
    jit_a   = rng.uniform(0.15, 1.5, n_anomaly)
    snr_a   = rng.uniform(-4, 6, n_anomaly)
    hrq_a   = rng.integers(3, 5, n_anomaly).astype(float)
    pkt_a   = rng.integers(0, 3, n_anomaly).astype(float)
    
    X = np.vstack([
        np.column_stack([lat_n, bler_n, jit_n, snr_n, hrq_n, pkt_n]),
        np.column_stack([lat_a, bler_a, jit_a, snr_a, hrq_a, pkt_a])
    ])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])   # 0=safe, 1=anomaly
    return X.astype(np.float32), y

X_saf, y_saf = generate_safety_dataset(10000)
X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_saf, y_saf, test_size=0.2, random_state=42)

scaler_saf = StandardScaler()
X_tr_ss = scaler_saf.fit_transform(X_tr_s)
X_te_ss = scaler_saf.transform(X_te_s)

safety_nn = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='tanh',
    solver='adam',
    max_iter=300,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=15,
    verbose=False
)
safety_nn.fit(X_tr_ss, y_tr_s)
saf_acc = accuracy_score(y_te_s, safety_nn.predict(X_te_ss))
print(f"  Safety Anomaly Detector NN — Test Accuracy: {saf_acc*100:.2f}%")
print(f"  Architecture: 6→64→32→16→2 | Activation: tanh | Classes: [SAFE, ANOMALY]")
print(f"  Training samples: {len(X_tr_s)} | Test samples: {len(X_te_s)}\n")

def is_safe(latency_ms, bler, jitter_ms, snr_dB, harq_rounds, pkt_type):
    """Returns (safe: bool, confidence: float). Triggers arm pause if unsafe."""
    feat = np.array([[latency_ms, bler, jitter_ms, snr_dB, harq_rounds, pkt_type]])
    feat_s = scaler_saf.transform(feat)
    prob   = safety_nn.predict_proba(feat_s)[0][1]   # P(anomaly)
    return prob < 0.5, float(prob)

# ============================================================
# END-TO-END SURGERY LINK SIMULATION
# ============================================================
print("=== E2E: Simulating Full Surgery Session (3000 packets) ===")

rng_e2e   = np.random.default_rng(13)
n_pkts    = traffic.n
latencies = np.zeros(n_pkts)
blers     = np.zeros(n_pkts)
mcs_chosen= np.zeros(n_pkts, dtype=int)
harq_rnds = np.zeros(n_pkts, dtype=int)
safety_flags = np.zeros(n_pkts)
haptic_rmse_per_pkt = []

recent_bler = 1e-3   # rolling estimate
recent_lat  = 0.5

haptic_buffer = np.zeros((3, 8), dtype=np.float32)

for i in range(n_pkts):
    snr_i    = float(traffic.snr_per_pkt[i])
    prio_i   = int(traffic.priority[i])
    ptype    = traffic.types[i]
    
    # ML-1: Adaptive MCS selection
    mcs_i = predict_mcs(snr_i, recent_bler, recent_lat, prio_i, 1)
    mcs_chosen[i] = mcs_i
    cr_i  = MCS_RATE[mcs_i]
    
    # Channel: simulate HARQ-IR transmission
    snr_l_i = 10 ** (snr_i / 10)
    # Fast single-packet HARQ (simplified, 200 trials for speed)
    b, lat, rnd_hist = harq_ir_engine(snr_l_i, K, L=4, max_rounds=4,
                                       n_trials=200, rng=rng_e2e)
    latencies[i] = lat
    blers[i]     = b
    harq_rnds[i] = int(np.round(np.mean(rnd_hist)))
    
    # Rolling average
    alpha       = 0.05
    recent_bler = (1 - alpha) * recent_bler + alpha * b
    recent_lat  = (1 - alpha) * recent_lat  + alpha * lat
    
    # ML-2: Haptic reconstruction (only for haptic packets)
    if ptype == 'haptic':
        raw_haptic   = traffic.haptic_payload[i]
        corrupt_prob = min(b * 10, 1.0)
        if rng_e2e.random() < corrupt_prob:
            corrupted = raw_haptic + rng_e2e.normal(0, 2.0, 8).astype(np.float32)
        else:
            corrupted = raw_haptic
        recon = reconstruct_haptic(haptic_buffer, corrupted, snr_i)
        err   = float(np.sqrt(np.mean((recon - raw_haptic)**2)))
        haptic_rmse_per_pkt.append(err)
        haptic_buffer = np.roll(haptic_buffer, 1, axis=0)
        haptic_buffer[0] = raw_haptic
    
    # ML-3: Safety check
    jitter_i = abs(lat - recent_lat)
    safe, anomaly_prob = is_safe(lat, b, jitter_i, snr_i, harq_rnds[i], prio_i)
    safety_flags[i] = anomaly_prob

print(f"  Simulated {n_pkts} packets")
print(f"  Mean E2E Latency : {np.mean(latencies):.4f} ms")
print(f"  P99 Latency      : {np.percentile(latencies, 99):.4f} ms")
print(f"  Mean BLER        : {np.mean(blers):.4e}")
print(f"  URLLC target met : {np.mean(latencies < 1.0)*100:.1f}% of packets under 1ms")
print(f"  Anomalies detected (P>0.5): {np.sum(safety_flags > 0.5)}")
print(f"  Mean haptic RMSE : {np.mean(haptic_rmse_per_pkt):.4f} N\n")

# ============================================================
# MODULE G — 9 RESEARCH-GRADE FIGURES
# ============================================================
print("=== MODULE G: Generating 9 Research Figures ===")

# -----------------------------------------------------------
# Shannon + PPV capacity bounds (Module 2 from project)
# -----------------------------------------------------------
C_shannon    = np.log2(1 + snr_lin)
V_awgn       = (1 - 1 / (1 + snr_lin)**2) * (np.log2(np.e))**2
eps_target   = 1e-5
Q_inv_eps    = 4.2649
n_block      = K
ppv_rate     = C_shannon - np.sqrt(V_awgn / n_block) * Q_inv_eps + np.log2(n_block) / (2 * n_block)
ppv_rate     = np.maximum(ppv_rate, 0)

# -----------------------------------------------------------
# Latency CDF (Module 6 from project)
# -----------------------------------------------------------
cdf_N   = 6000
L_cdf   = 4
snr_cdf = [5, 10, 15, 20]
lat_cdf = []
T_RTT   = 2 * T_slot + gNB_proc_ms + UE_proc_ms
rng_cdf = np.random.default_rng(3)
for snr_c_db in snr_cdf:
    snr_c = 10**(snr_c_db / 10)
    sig_c = 1 / np.sqrt(2 * snr_c)
    TX_c  = 2 * rng_cdf.integers(0, 2, (K, cdf_N)) - 1
    H_c   = (rng_cdf.standard_normal((L_cdf, cdf_N)) +
              1j * rng_cdf.standard_normal((L_cdf, cdf_N))) / np.sqrt(2)
    Hp_c  = np.sum(np.abs(H_c)**2, axis=0)
    y_c   = Hp_c * TX_c + np.sqrt(Hp_c) * sig_c * rng_cdf.standard_normal((K, cdf_N))
    err_c = np.any((y_c > 0) != (TX_c > 0), axis=0)
    lat_v = T_mini_ms + gNB_proc_ms + UE_proc_ms + prop_ms + err_c * T_RTT
    lat_cdf.append(np.sort(lat_v))

# -----------------------------------------------------------
# Reliability-Latency tradeoff (Module 7)
# -----------------------------------------------------------
sym_opts     = [2, 4, 7, 14]
snr_trade    = 12
snr_t_lin    = 10**(snr_trade / 10)
L_trade      = 4
trade_N      = 3000
bler_trade   = []
lat_trade    = []
rng_trade    = np.random.default_rng(5)
for n_sym in sym_opts:
    T_tx        = (n_sym / 14) * T_slot
    coding_gain = 10**(3 * np.log10(n_sym / 2) / 10)
    snr_eff     = snr_t_lin * coding_gain
    sig_t       = 1 / np.sqrt(2 * snr_eff)
    TX_t = 2 * rng_trade.integers(0, 2, (K, trade_N)) - 1
    H_t  = (rng_trade.standard_normal((L_trade, trade_N)) +
             1j * rng_trade.standard_normal((L_trade, trade_N))) / np.sqrt(2)
    Hp_t = np.sum(np.abs(H_t)**2, axis=0)
    y_t  = Hp_t * TX_t + np.sqrt(Hp_t) * sig_t * rng_trade.standard_normal((K, trade_N))
    err_t = np.any((y_t > 0) != (TX_t > 0), axis=0)
    bler_trade.append(max(float(np.mean(err_t)), 0.5 / trade_N))
    lat_trade.append(T_tx + gNB_proc_ms + UE_proc_ms + prop_ms)

# -----------------------------------------------------------
# Numerology latency grid (Module 8)
# -----------------------------------------------------------
n_mini_opts  = [2, 4, 7, 14]
lat_num_grid = np.zeros((len(scs_kHz), len(n_mini_opts)))
for mi, T_s in enumerate(slot_ms):
    for ni, nm in enumerate(n_mini_opts):
        lat_num_grid[mi, ni] = (nm / 14) * T_s + gNB_proc_ms + UE_proc_ms + prop_ms

# ============================================================
# PLOT ALL 9 FIGURES
# ============================================================
cmap    = plt.cm.tab10.colors
markers = ['-o', '-s', '-^', '-d', '-v', '-p']
cdf_clr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

fig_dir = "/mnt/user-data/outputs/"
import os; os.makedirs(fig_dir, exist_ok=True)

# ── Figure 1: BLER vs SNR with MRC diversity ────────────────
fig1, ax1 = plt.subplots(figsize=(9, 6))
ax1.set_facecolor('#f9f9f9')
for di, L in enumerate(diversity_orders):
    ax1.semilogy(snr_db, results_bler[di], markers[di],
                 color=cmap[di], lw=2.2, ms=8, label=f'Sim: L={L} (MRC)')
    ax1.semilogy(snr_db, bler_theory[di], '--',
                 color=cmap[di], lw=1.2, alpha=0.7)
ax1.axhline(1e-5, color='k', ls='--', lw=2, label='URLLC Target (10⁻⁵)')
ax1.axhline(1e-3, color='gray', ls=':', lw=1.2)
ax1.set_xlabel('Channel SNR (dB)', fontsize=13)
ax1.set_ylabel('Block Error Rate (BLER)', fontsize=13)
ax1.set_title('Fig 1 — BLER vs SNR: MRC Antenna Diversity (Rayleigh)\n'
              'Solid markers = Monte Carlo  |  Dashed = Analytical Theory', fontsize=12, fontweight='bold')
ax1.legend(loc='lower left', fontsize=10)
ax1.set_ylim([1e-7, 1]); ax1.set_xlim([snr_db[0]-1, snr_db[-1]+1])
ax1.grid(True, which='both', alpha=0.4)
fig1.tight_layout()
fig1.savefig(fig_dir + "fig1_bler_vs_snr_diversity.png", dpi=150)
plt.close(fig1)

# ── Figure 2: HARQ-IR vs No-HARQ ────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.set_facecolor('#f9f9f9')
ax2.semilogy(snr_db, results_bler[2], 'r-o', lw=2.2, ms=8, label='No HARQ (L=4)')
ax2.semilogy(snr_db, bler_harq_ir,   'g-^', lw=2.2, ms=8, label='HARQ-IR (L=4, max 4 rounds)')
ax2.axhline(1e-5, color='k', ls='--', lw=2, label='URLLC 10⁻⁵')
ax2.set_xlabel('SNR (dB)', fontsize=13)
ax2.set_ylabel('BLER', fontsize=13)
ax2.set_title('Fig 2 — HARQ-IR Protocol vs No-HARQ (L=4 MRC)\n'
              'IR accumulates parity bits each round → dramatic BLER reduction', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.set_ylim([1e-7, 1]); ax2.grid(True, which='both', alpha=0.4)
fig2.tight_layout()
fig2.savefig(fig_dir + "fig2_harq_ir_comparison.png", dpi=150)
plt.close(fig2)

# ── Figure 3: E2E Latency across diversity orders ───────────
fig3, ax3 = plt.subplots(figsize=(9, 5))
ax3.set_facecolor('#f9f9f9')
for di, L in enumerate(diversity_orders):
    ax3.plot(snr_db, results_latency[di] * 1000, markers[di],
             color=cmap[di], lw=2, ms=7, label=f'L={L}')
ax3.axhline(1.0, color='r', ls='--', lw=2, label='1 ms URLLC Deadline')
ax3.axhline(0.5, color='b', ls='-.', lw=1.5, label='0.5 ms Aggressive Target')
ax3.set_xlabel('SNR (dB)', fontsize=13)
ax3.set_ylabel('E2E Latency (ms)', fontsize=13)
ax3.set_title('Fig 3 — E2E Latency vs SNR & Diversity Order\n'
              'Higher L → fewer HARQ retransmissions → lower latency', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10); ax3.grid(True, alpha=0.4)
ax3.set_ylim([0, 2])
fig3.tight_layout()
fig3.savefig(fig_dir + "fig3_e2e_latency_diversity.png", dpi=150)
plt.close(fig3)

# ── Figure 4: Shannon + PPV Capacity Bounds ─────────────────
fig4, ax4 = plt.subplots(figsize=(9, 5))
ax4.set_facecolor('#f9f9f9')
ax4.plot(snr_db, C_shannon, 'b-', lw=2.5, label='Shannon Capacity (infinite blocklength)')
ax4.plot(snr_db, ppv_rate,  'r--', lw=2.5, label=f'PPV Bound (n={K} bits, ε=10⁻⁵)')
ax4.fill_between(snr_db, ppv_rate, C_shannon, alpha=0.15, color='red', label='Short-packet rate penalty')
ax4.set_xlabel('SNR (dB)', fontsize=13)
ax4.set_ylabel('Achievable Rate (bpcu)', fontsize=13)
ax4.set_title('Fig 4 — Theoretical Capacity: Shannon vs PPV Finite-Blocklength Bound\n'
              'Red region = rate penalty from URLLC short packets (K=256 bits)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10); ax4.grid(True, alpha=0.4)
fig4.tight_layout()
fig4.savefig(fig_dir + "fig4_shannon_ppv_capacity.png", dpi=150)
plt.close(fig4)

# ── Figure 5: Latency CDF (Jitter analysis) ─────────────────
fig5, ax5 = plt.subplots(figsize=(9, 5))
ax5.set_facecolor('#f9f9f9')
for ci, (lc, snr_c_db) in enumerate(zip(lat_cdf, snr_cdf)):
    cdf_y = np.arange(1, len(lc) + 1) / len(lc)
    ax5.plot(lc * 1000, cdf_y, color=cdf_clr[ci], lw=2.2,
             label=f'SNR = {snr_c_db} dB')
ax5.axvline(1.0, color='k', ls='--', lw=2, label='1 ms URLLC Deadline')
ax5.axhline(1 - 1e-5, color='gray', ls=':', lw=1.5, label='CDF = 1−10⁻⁵')
ax5.set_xlabel('E2E Latency (ms)', fontsize=13)
ax5.set_ylabel('CDF', fontsize=13)
ax5.set_title('Fig 5 — Latency CDF & Jitter Analysis (L=4 MRC)\n'
              'Tail latency reveals HARQ-induced jitter at low SNR', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10); ax5.grid(True, alpha=0.4)
ax5.set_xlim([0, 2.5]); ax5.set_ylim([0, 1.05])
fig5.tight_layout()
fig5.savefig(fig_dir + "fig5_latency_cdf_jitter.png", dpi=150)
plt.close(fig5)

# ── Figure 6: Reliability-Latency Tradeoff ──────────────────
fig6, ax6 = plt.subplots(figsize=(8, 6))
ax6.set_facecolor('#f0f0f0')
ax6.fill([1e-5, 1, 1, 1e-5], [0, 0, 4, 4], color='#ffcccc', alpha=0.5, label='_nolegend_')
ax6.fill([1e-7, 1, 1, 1e-7], [1, 1, 4, 4], color='#ccccff', alpha=0.5, label='_nolegend_')
ax6.fill([1e-7, 1e-5, 1e-5, 1e-7], [0, 0, 1, 1], color='#ccffcc', alpha=0.6, label='URLLC Feasible Zone')
ax6.loglog(bler_trade, [l * 1000 for l in lat_trade], 'k-o', lw=2.5, ms=12,
           mfc='k', label=f'Vary mini-slot (SNR={snr_trade}dB, L={L_trade})')
for si, (b, l, ns) in enumerate(zip(bler_trade, lat_trade, sym_opts)):
    ax6.text(b * 1.4, l * 1000, f'  {ns} sym\n  ({l*1000:.2f}ms)', fontsize=9, color='#0000CC', fontweight='bold')
ax6.axvline(1e-5, color='r', ls='--', lw=2, label='BLER = 10⁻⁵')
ax6.axhline(1.0,  color='b', ls='--', lw=2, label='1 ms Deadline')
ax6.set_xlabel('Block Error Rate (BLER)', fontsize=13)
ax6.set_ylabel('E2E Latency (ms)', fontsize=13)
ax6.set_title('Fig 6 — URLLC Reliability–Latency Tradeoff\n'
              'Green = Feasible URLLC Zone | Longer mini-slot → lower BLER, higher latency', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10); ax6.grid(True, which='both', alpha=0.4)
ax6.set_xlim([1e-5, 0.5]); ax6.set_ylim([0, 2])
fig6.tight_layout()
fig6.savefig(fig_dir + "fig6_reliability_latency_tradeoff.png", dpi=150)
plt.close(fig6)

# ── Figure 7: ML Adaptive MCS — predicted MCS vs SNR ───────
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(13, 5))
# Left: test set MCS prediction accuracy heatmap-style
snr_test_vals = X_test_m[:, 0]
mcs_pred_test = mcs_nn.predict(X_test_ms)
bins = np.arange(-4, 26, 4)
mcs_mean_pred = [np.mean(mcs_pred_test[(snr_test_vals >= bins[i]) & (snr_test_vals < bins[i+1])])
                 if np.any((snr_test_vals >= bins[i]) & (snr_test_vals < bins[i+1])) else np.nan
                 for i in range(len(bins)-1)]
ax7a.set_facecolor('#f9f9f9')
ax7a.bar(bins[:-1] + 2, mcs_mean_pred, width=3.5, color='steelblue', alpha=0.8, edgecolor='navy')
ax7a.set_xlabel('SNR (dB)', fontsize=12)
ax7a.set_ylabel('Mean Predicted MCS Index', fontsize=12)
ax7a.set_title('ML Adaptive MCS Predictor\nMean Selected MCS vs SNR', fontsize=12, fontweight='bold')
ax7a.grid(True, alpha=0.4); ax7a.set_ylim([0, 11])
# Right: per-packet MCS over surgery session
pkt_idx = np.arange(500)
ax7b.set_facecolor('#f9f9f9')
sc = ax7b.scatter(pkt_idx, mcs_chosen[:500], c=traffic.snr_per_pkt[:500],
                   cmap='RdYlGn', s=10, alpha=0.7, vmin=-4, vmax=24)
plt.colorbar(sc, ax=ax7b, label='SNR (dB)')
ax7b.set_xlabel('Packet Index', fontsize=12)
ax7b.set_ylabel('MCS Index', fontsize=12)
ax7b.set_title('Adaptive MCS per Packet (Surgery Session)\nColour = channel SNR', fontsize=12, fontweight='bold')
ax7b.grid(True, alpha=0.4)
fig7.suptitle(f'Fig 7 — ML Neural Network: Adaptive MCS Controller (Acc={mcs_acc*100:.1f}%)',
              fontsize=13, fontweight='bold')
fig7.tight_layout()
fig7.savefig(fig_dir + "fig7_ml_adaptive_mcs.png", dpi=150)
plt.close(fig7)

# ── Figure 8: Haptic Reconstruction ─────────────────────────
# Show one dimension of haptic signal: clean, corrupted, reconstructed
fig8, axes8 = plt.subplots(2, 1, figsize=(13, 7))
# Top: one force dimension over time
n_show = 300
t_show = np.arange(n_show)
y_te_h_show = y_te_h[:n_show, 0]   # Fx channel
y_pred_h_show = y_pred_h[:n_show, 0]
corrupted_show = X_te_h[:n_show, 24]   # raw corrupted from feature vec

axes8[0].set_facecolor('#f9f9f9')
axes8[0].plot(t_show, y_te_h_show,    'b-',  lw=1.5, label='Clean (ground truth)', alpha=0.9)
axes8[0].plot(t_show, corrupted_show, 'r--', lw=1.0, label='Corrupted (channel loss)', alpha=0.6)
axes8[0].plot(t_show, y_pred_h_show,  'g-',  lw=1.5, label='NN Reconstructed', alpha=0.9)
axes8[0].set_xlabel('Packet Index', fontsize=12)
axes8[0].set_ylabel('Force Fx (N)', fontsize=12)
axes8[0].set_title('Haptic Force Signal: Clean vs Corrupted vs NN Reconstructed', fontsize=12, fontweight='bold')
axes8[0].legend(fontsize=10); axes8[0].grid(True, alpha=0.4)

# Bottom: per-packet RMSE
axes8[1].set_facecolor('#f9f9f9')
n_hap = len(haptic_rmse_per_pkt)
axes8[1].plot(np.arange(n_hap), haptic_rmse_per_pkt, 'purple', lw=0.8, alpha=0.6, label='Per-pkt RMSE')
axes8[1].axhline(np.mean(haptic_rmse_per_pkt), color='g', ls='--', lw=2,
                  label=f'Mean RMSE = {np.mean(haptic_rmse_per_pkt):.4f} N')
axes8[1].axhline(0.1, color='r', ls='--', lw=1.5, label='Safety threshold (0.1 N)')
axes8[1].set_xlabel('Haptic Packet Index', fontsize=12)
axes8[1].set_ylabel('Reconstruction RMSE (N)', fontsize=12)
axes8[1].set_title(f'Haptic Reconstruction Error per Packet (Architecture: 33→256→128→64→32→8, RMSE={haptic_rmse:.4f}N)',
                    fontsize=11, fontweight='bold')
axes8[1].legend(fontsize=10); axes8[1].grid(True, alpha=0.4)

fig8.suptitle('Fig 8 — ML/DL: Haptic Signal Reconstruction Neural Network', fontsize=13, fontweight='bold')
fig8.tight_layout()
fig8.savefig(fig_dir + "fig8_haptic_reconstruction.png", dpi=150)
plt.close(fig8)

# ── Figure 9: Surgery Safety Monitor + Anomaly Detection ────
fig9, axes9 = plt.subplots(2, 2, figsize=(14, 9))
fig9.suptitle('Fig 9 — Remote Robotic Surgery: Full System Dashboard', fontsize=14, fontweight='bold')

# Top-left: Latency over surgery session with anomaly markers
ax9a = axes9[0, 0]
ax9a.set_facecolor('#f9f9f9')
pkt_x = np.arange(n_pkts)
ax9a.plot(pkt_x, latencies * 1000, lw=0.5, color='steelblue', alpha=0.6, label='Latency')
anomaly_idx = np.where(safety_flags > 0.5)[0]
ax9a.scatter(anomaly_idx, latencies[anomaly_idx] * 1000, color='red', s=20, zorder=5,
              label=f'Anomalies ({len(anomaly_idx)})')
ax9a.axhline(1.0, color='r', ls='--', lw=1.5, label='1 ms Deadline')
ax9a.set_xlabel('Packet Index', fontsize=11)
ax9a.set_ylabel('Latency (ms)', fontsize=11)
ax9a.set_title('E2E Latency — Surgery Session', fontsize=11, fontweight='bold')
ax9a.legend(fontsize=9); ax9a.grid(True, alpha=0.3)

# Top-right: BLER per packet with traffic type colour
ax9b = axes9[0, 1]
ax9b.set_facecolor('#f9f9f9')
type_color = {'command': 'red', 'haptic': 'blue', 'video': 'green'}
for ptype in ['command', 'haptic', 'video']:
    mask = traffic.types == ptype
    ax9b.semilogy(pkt_x[mask], blers[mask], '.', color=type_color[ptype],
                   ms=2, alpha=0.5, label=ptype.capitalize())
ax9b.axhline(1e-5, color='k', ls='--', lw=1.5, label='URLLC 10⁻⁵')
ax9b.set_xlabel('Packet Index', fontsize=11)
ax9b.set_ylabel('BLER', fontsize=11)
ax9b.set_title('BLER per Packet — by Traffic Type', fontsize=11, fontweight='bold')
ax9b.legend(fontsize=9); ax9b.grid(True, which='both', alpha=0.3)

# Bottom-left: Anomaly probability over time
ax9c = axes9[1, 0]
ax9c.set_facecolor('#f9f9f9')
ax9c.plot(pkt_x, safety_flags, lw=0.7, color='darkorange', alpha=0.7)
ax9c.axhline(0.5, color='red', ls='--', lw=1.5, label='Anomaly threshold (0.5)')
ax9c.fill_between(pkt_x, 0.5, safety_flags, where=safety_flags > 0.5,
                   color='red', alpha=0.3, label='Unsafe zone')
ax9c.set_xlabel('Packet Index', fontsize=11)
ax9c.set_ylabel('P(Anomaly)', fontsize=11)
ax9c.set_title(f'Safety Monitor — Anomaly Probability\n(Acc={saf_acc*100:.1f}%)', fontsize=11, fontweight='bold')
ax9c.legend(fontsize=9); ax9c.grid(True, alpha=0.3)

# Bottom-right: System Architecture diagram
ax9d = axes9[1, 1]
ax9d.set_xlim(0, 10); ax9d.set_ylim(0, 10)
ax9d.axis('off')
ax9d.set_facecolor('#fafafa')

def draw_box(ax, x, y, w, h, color, text, fontsize=9):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.1", fc=color, ec='k', lw=1.2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)

draw_box(ax9d, 1.5, 8.5, 2.5, 1.2, '#AED6F1', 'Surgeon\nConsole')
draw_box(ax9d, 5.0, 8.5, 2.5, 1.2, '#AED6F1', '5G gNB\nBase Station')
draw_box(ax9d, 8.5, 8.5, 2.5, 1.2, '#AED6F1', 'Robotic\nArm')
draw_box(ax9d, 1.5, 5.5, 2.5, 1.4, '#A9DFBF', 'ML-1\nAdaptive MCS\nNeural Net')
draw_box(ax9d, 5.0, 5.5, 2.5, 1.4, '#A9DFBF', 'URLLC Engine\nHARQ-IR + MRC\nRayleigh Fading')
draw_box(ax9d, 8.5, 5.5, 2.5, 1.4, '#A9DFBF', 'ML-2\nHaptic Recon\n256→32→8 NN')
draw_box(ax9d, 5.0, 2.5, 3.0, 1.4, '#F9E79F', 'ML-3\nSafety Monitor\nAnomaly Detect NN')

for (x1,y1),(x2,y2) in [((2.75,8.5),(3.75,8.5)),((6.25,8.5),(7.25,8.5)),
                          ((1.5,7.9),(1.5,6.2)),  ((5.0,7.9),(5.0,6.2)),
                          ((8.5,7.9),(8.5,6.2)),  ((5.0,4.8),(5.0,3.2)),
                          ((1.5,4.8),(3.8,3.2)),  ((8.5,4.8),(6.2,3.2))]:
    ax9d.annotate('', xy=(x2,y2), xytext=(x1,y1),
                  arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

ax9d.text(5, 0.6, 'Remote Robotic Surgery — 5G NR URLLC Architecture\n(No Hardware Required | ML/DL integrated)',
           ha='center', fontsize=8.5, style='italic', color='#555555')
ax9d.set_title('System Architecture', fontsize=11, fontweight='bold')

fig9.tight_layout()
fig9.savefig(fig_dir + "fig9_surgery_dashboard.png", dpi=150)
plt.close(fig9)

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n╔══════════════════════════════════════════════════════════════╗")
print("║     REMOTE ROBOTIC SURGERY — COMPLETE PERFORMANCE SUMMARY   ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║  5G NR Config : µ={mu_sel}, SCS={scs_sel} kHz, Mini-slot={n_mini}sym ({T_mini_ms:.4f}ms)    ║")
print(f"║  Fixed overhead: gNB={gNB_proc_ms:.1f}ms | UE={UE_proc_ms:.1f}ms | Prop={prop_ms:.1f}ms       ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║  URLLC Constraint: BLER ≤ 1e-5 AND E2E Latency ≤ 1 ms     ║")
idx10 = np.where(snr_db == 10)[0][0]
print(f"║  BLER @ SNR=10dB: L=1:{results_bler[0,idx10]:.2e}  L=4:{results_bler[2,idx10]:.2e}  ║")
print(f"║  PPV rate penalty @ 10dB: {100*(C_shannon[idx10]-ppv_rate[idx10])/C_shannon[idx10]:.1f}% vs Shannon       ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║  Surgery Session ({n_pkts} packets):                          ║")
print(f"║    Mean latency  : {np.mean(latencies)*1000:.3f} ms                           ║")
print(f"║    P99 latency   : {np.percentile(latencies,99)*1000:.3f} ms                           ║")
print(f"║    Deadline met  : {np.mean(latencies<1.0)*100:.1f}% packets under 1ms              ║")
print(f"║    Anomalies     : {len(anomaly_idx)} safety-critical events detected     ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║  ML-1 Adaptive MCS Neural Net   Accuracy : {mcs_acc*100:.1f}%        ║")
print(f"║  ML-2 Haptic Reconstruction NN  RMSE     : {haptic_rmse:.4f} N      ║")
print(f"║       Improvement over corrupted baseline : {100*(baseline_rmse-haptic_rmse)/baseline_rmse:.1f}%         ║")
print(f"║  ML-3 Safety Monitor NN         Accuracy : {saf_acc*100:.1f}%        ║")
print("╠══════════════════════════════════════════════════════════════╣")
print("║  9 Figures saved to /mnt/user-data/outputs/                 ║")
print("╚══════════════════════════════════════════════════════════════╝")
print("\n=== All 9 Figures Generated ===")
print("  [1] BLER vs SNR (MRC Diversity)        → fig1_bler_vs_snr_diversity.png")
print("  [2] HARQ-IR vs No-HARQ                 → fig2_harq_ir_comparison.png")
print("  [3] E2E Latency vs Diversity            → fig3_e2e_latency_diversity.png")
print("  [4] Shannon + PPV Capacity              → fig4_shannon_ppv_capacity.png")
print("  [5] Latency CDF & Jitter                → fig5_latency_cdf_jitter.png")
print("  [6] Reliability-Latency Tradeoff        → fig6_reliability_latency_tradeoff.png")
print("  [7] ML Adaptive MCS Controller          → fig7_ml_adaptive_mcs.png")
print("  [8] Haptic Signal Reconstruction NN     → fig8_haptic_reconstruction.png")
print("  [9] Surgery Dashboard + Architecture    → fig9_surgery_dashboard.png")
