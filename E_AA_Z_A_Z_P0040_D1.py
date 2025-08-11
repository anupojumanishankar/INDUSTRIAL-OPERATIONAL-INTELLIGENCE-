"""
Synthetic 3‑phase MFM data generator for:
  DID = "E_AA_Z_A_Z_P0040_D1"  (user-requested)
  Asset (context): "E_AA_Z_A_X_P0040_D2 ... PCC7_OUT6 MCC12_OUT1 COMPRESSOR 17"

What it does
------------
• 10 days of 5‑minute data ending now (exclusive).
• Columns exactly as requested (kWh/kVArh/kVAh cumulative, per‑phase powers, V/I/PF, THD, etc.).
• Realistic relationships for a 60 kW, 415 V, 50 Hz, 3‑phase induction motor:
  FLA≈107 A, PF/η vs load curves, service factor 1.15, voltage/current imbalance,
  idle, starts, overloads, trips, power‑off windows (1–3 h), losses, spikes.
• Adds measurement noise, a few outliers (“huge noise”), and random NULLs.
• Saves CSV: ./synthetic_mfm_E_AA_Z_A_Z_P0040_D1_5min_10days.csv
"""

import numpy as np
import pandas as pd

# --------------------------- config ---------------------------------
np.random.seed(42)

RATED_KW = 60.0                      # output power rating
V_LL_NOM = 415.0                     # nominal line-line voltage
V_LN_NOM = V_LL_NOM / np.sqrt(3)     # nominal line-neutral (~240 V)
FLA = 107.0                          # full-load current (A)
FREQ_NOM = 50.0                      # Hz
SERVICE_FACTOR = 1.15                # occasional overload

DID = "E_AA_Z_A_Z_P0040_D1"
FID = "V1.0"
SLAVE_ID = 114

# PF and Efficiency vs load (piecewise-linear from user’s table)
pf_curve = np.array([
    [0.25, 0.60],
    [0.50, 0.75],
    [0.75, 0.82],
    [1.00, 0.85]
])
eta_curve = np.array([
    [0.25, 0.85],
    [0.50, 0.90],
    [0.75, 0.915],
    [1.00, 0.92]
])
# Current vs load mapping (approx.; includes 25/50/75/100% points)
i_curve = np.array([
    [0.25, 45.0],
    [0.50, 64.0],
    [0.75, 82.0],
    [1.00, 107.0]
])

# Sampling window: last 10 days, 5-minute data, ending at now (exclusive)
end = pd.Timestamp.now().floor("5min")
idx = pd.date_range(end=end, periods=10*24*12, freq="5min", inclusive="left")

# ----------------------- helper functions ---------------------------
def lerp_curve(x, curve):
    """Piecewise-linear interpolate y for x in [0,1] with given 2D curve [[x,y]...]."""
    xs, ys = curve[:,0], curve[:,1]
    x = np.clip(x, xs.min(), xs.max())
    return np.interp(x, xs, ys)

def daily_profile(n):
    """Gentle daily load shape: higher mid-day/evening, lower late night."""
    t = np.arange(n)
    # 24h sinusoid with mild second harmonic; scaled into [0,1]
    day = (np.sin(2*np.pi*t/(24*12) - np.pi/2) + 1)/2
    eve = (np.sin(4*np.pi*t/(24*12)) + 1)/8
    base = 0.45 + 0.4*day + eve
    return np.clip(base, 0.35, 1.05)

def random_windows(n, count_range, dur_5min_range):
    """Pick random 'count' windows; each lasts random duration in 5-min steps."""
    count = np.random.randint(count_range[0], count_range[1]+1)
    starts = np.random.choice(np.arange(n), size=count, replace=False)
    mask = np.zeros(n, dtype=bool)
    for s in starts:
        d = np.random.randint(dur_5min_range[0], dur_5min_range[1]+1)
        e = min(s+d, n)
        mask[s:e] = True
    return mask

def rolling_rssi(n, base=-62, jitter=4, drift=2):
    """RSSI around base (dBm) with slow drift and fast jitter."""
    slow = np.cumsum(np.random.randn(n)*0.02) * drift
    fast = np.random.randn(n) * jitter
    rssi = base + slow + fast
    return np.clip(rssi, -90, -40)

# -------------------------- scenarios --------------------------------
n = len(idx)

# Baseline fractional load (mechanical), then clamp to [0.4, 1.0] during RUN
base_load = daily_profile(n)

# Events:
power_off = random_windows(n, count_range=(3, 6), dur_5min_range=(12, 36))  # 1–3 hours
idle_run  = random_windows(n, count_range=(6, 10), dur_5min_range=(6, 18))   # 30–90 min low load
overload  = random_windows(n, count_range=(2, 4), dur_5min_range=(2, 6))     # 10–30 min
starts    = random_windows(n, count_range=(8, 14), dur_5min_range=(1, 1))    # single 5‑min spikes
trips     = random_windows(n, count_range=(2, 5), dur_5min_range=(3, 8))     # short forced stop

# Over-current spikes (non-start): single-interval surges
spikes    = random_windows(n, count_range=(8, 14), dur_5min_range=(1, 1))

# Effective operating state masks
off_mask   = power_off | trips
start_mask = starts & (~off_mask)
idle_mask  = idle_run & (~off_mask) & (~start_mask)
over_mask  = overload & (~off_mask) & (~start_mask)

# Choose mechanical load fraction by state
load_frac = base_load.copy()
# Typical RUN bounds
load_frac = np.clip(load_frac, 0.40, 1.00)
# Idle: very light load
load_frac[idle_mask] = np.random.uniform(0.10, 0.20, idle_mask.sum())
# Overload: up to service factor
load_frac[over_mask] = np.random.uniform(1.02, SERVICE_FACTOR, over_mask.sum())
# Start: electrical transient; mechanical load not yet applied (very low)
load_frac[start_mask] = np.random.uniform(0.05, 0.12, start_mask.sum())
# Off: no load
load_frac[off_mask] = 0.0

# Interpolate PF and efficiency
pf_avg = lerp_curve(load_frac, pf_curve)
eta    = lerp_curve(load_frac, eta_curve)

# Small PF shaping during start/spike/over
pf_avg[start_mask] *= np.random.uniform(0.20, 0.35, start_mask.sum()) / pf_avg[start_mask]
pf_avg[over_mask]  *= np.random.uniform(0.95, 1.00, over_mask.sum()) / pf_avg[over_mask]
pf_avg = np.clip(pf_avg, 0.15, 0.97)

# Voltage imbalance (% of VLN spread)
imb_pct = 0.005 + 0.01*np.random.rand(n)         # 0.5%–1.5%
imb_pct[over_mask]  += 0.003
imb_pct[start_mask] += 0.004
imb_pct = np.clip(imb_pct, 0.004, 0.025)

# Phase voltages (VLN) with imbalance and small noise
v_mean = V_LN_NOM * (1.0 + 0.01*np.random.randn(n))  # ~±1%
dR =  ( np.random.rand(n) - 0.5) * 2 * imb_pct
dY =  ( np.random.rand(n) - 0.5) * 2 * imb_pct
dB = -(dR + dY)  # keep sum roughly ~0
VR = v_mean * (1 + dR) + np.random.randn(n)*0.3
VY = v_mean * (1 + dY) + np.random.randn(n)*0.3
VB = v_mean * (1 + dB) + np.random.randn(n)*0.3
AVG_VLN = (VR + VY + VB)/3.0

# Line-line voltages (approx from AVG_VLN and imbalance)
VRY = (VR + VY) * 0.866 + np.random.randn(n)*0.6   # ≈ √3/2 * sum, small noise
VYB = (VY + VB) * 0.866 + np.random.randn(n)*0.6
VBR = (VB + VR) * 0.866 + np.random.randn(n)*0.6
AVG_VLL = (VRY + VYB + VBR)/3.0

# Base current from load fraction (interpolate i_curve), then modify by events
i_avg = lerp_curve(load_frac, i_curve)
# Starts: high inrush (averaged over 5‑min, not full 6–8x FLA)
i_avg[start_mask] = np.random.uniform(2.3, 3.3, start_mask.sum()) * FLA * 0.6  # ~1.4–2.0x FLA avg
# Over-current spikes (keep PF similar)
i_avg[spikes] *= np.random.uniform(1.3, 1.7, spikes.sum())
# Off: zero
i_avg[off_mask] = 0.0

# Split currents per phase with imbalance
ci = 0.5*imb_pct + 0.01*np.random.rand(n)  # current imbalance level
IR = i_avg * (1 + (np.random.rand(n)-0.5)*2*ci)
IY = i_avg * (1 + (np.random.rand(n)-0.5)*2*ci)
IB = np.maximum(0.0, 3*i_avg - (IR + IY))  # keep sum near 3*i_avg, non-negative
AVG_I = (IR + IY + IB)/3.0

# Apparent power per phase (kVA) using VLN * I (single‑phase basis)
R_KVA = (VR*IR)/1000.0
Y_KVA = (VY*IY)/1000.0
B_KVA = (VB*IB)/1000.0
TOTAL_KVA = R_KVA + Y_KVA + B_KVA

# Per‑phase PFs around average PF, then enforce consistency P = S*PF
phase_pf_jitter = 0.03 + 0.04*np.random.rand(n)
R_PF = np.clip(pf_avg * (1 + (np.random.rand(n)-0.5)*2*phase_pf_jitter), 0.10, 0.99)
Y_PF = np.clip(pf_avg * (1 + (np.random.rand(n)-0.5)*2*phase_pf_jitter), 0.10, 0.99)
B_PF = np.clip(pf_avg * (1 + (np.random.rand(n)-0.5)*2*phase_pf_jitter), 0.10, 0.99)
AVG_PF = (R_PF + Y_PF + B_PF)/3.0

# Real power per phase (kW)
R_KW = R_KVA * R_PF
Y_KW = Y_KVA * Y_PF
B_KW = B_KVA * B_PF
TOTAL_KW = R_KW + Y_KW + B_KW

# Reactive power per phase (kVAr) (lagging assumed)
R_KVAR = np.sqrt(np.maximum(R_KVA**2 - R_KW**2, 0.0))
Y_KVAR = np.sqrt(np.maximum(Y_KVA**2 - Y_KW**2, 0.0))
B_KVAR = np.sqrt(np.maximum(B_KVA**2 - B_KW**2, 0.0))
TOTAL_KVAR = R_KVAR + Y_KVAR + B_KVAR

# Frequency
FREQUENCY = FREQ_NOM + np.random.randn(n)*0.03
FREQUENCY[off_mask] = 0.0

# RSSI (wifi)
RSSI = rolling_rssi(n)
RSSI[off_mask] = np.where(np.random.rand(off_mask.sum())<0.4, np.nan, RSSI[off_mask]-np.random.uniform(6,12,off_mask.sum()))

# THD models (percent)
# Voltage THD rises a bit with voltage imbalance
R_THD_V = (1.5 + 180*abs(dR)) + np.random.rand(n)*0.5
Y_THD_V = (1.5 + 180*abs(dY)) + np.random.rand(n)*0.5
B_THD_V = (1.5 + 180*abs(dB)) + np.random.rand(n)*0.5
# Current THD: higher at light load and during odd events
light_load_factor = np.clip(0.6 - load_frac, 0.0, 0.6)  # bigger when load small
base_thdi = 6 + 25*light_load_factor + 60*imb_pct
R_THD_I = base_thdi + 2*np.random.rand(n)
Y_THD_I = base_thdi + 2*np.random.rand(n)
B_THD_I = base_thdi + 2*np.random.rand(n)
# Starts/spikes bump THD-I
for arr in (R_THD_I, Y_THD_I, B_THD_I):
    arr[start_mask] += np.random.uniform(8,14, start_mask.sum())
    arr[spikes]     += np.random.uniform(5,10, spikes.sum())
# Clamp reasonable ranges
for arr in (R_THD_V, Y_THD_V, B_THD_V, R_THD_I, Y_THD_I, B_THD_I):
    np.clip(arr, 0.5, 35, out=arr)

# Energy integrators (cumulative)
dt_h = 5/60.0
kWh_inc   = TOTAL_KW   * dt_h
kVArh_inc = TOTAL_KVAR * dt_h
kVAh_inc  = TOTAL_KVA  * dt_h

# Add a small “meter constant” bias to represent losses/cal errors
kWh_inc  *= (1.002 + 0.001*np.random.randn(n))
kVAh_inc *= (1.001 + 0.001*np.random.randn(n))
kVArh_inc*= (1.003 + 0.001*np.random.randn(n))

# Off intervals: integrators ~0
kWh_inc[off_mask] = 0.0
kVAh_inc[off_mask] = 0.0
kVArh_inc[off_mask] = 0.0

# Start from a non-zero reading to look like a real meter
kWh0, kVAh0, kVArh0 = 1000.0, 1200.0, 700.0
KWH   = kWh0   + np.cumsum(kWh_inc)
KVARH = kVArh0 + np.cumsum(kVArh_inc)
KVAH  = kVAh0  + np.cumsum(kVAh_inc)

# Per‑phase PF zero/NaN behavior on OFF: set PF to NaN, currents 0 already
for arr in (R_PF, Y_PF, B_PF, AVG_PF):
    arr[off_mask] = np.nan

# ------------------------ assemble dataframe ------------------------
df = pd.DataFrame({
    "TS": idx,
    "DID": DID,
    "FID": FID,
    "RSSI": RSSI,
    "SLAVE ID": SLAVE_ID,

    "KWH": KWH,
    "KVAH": KVAH,
    "KVARH": KVARH,

    "VR": VR, "VY": VY, "VB": VB,
    "AVG_VLN": AVG_VLN,
    "VRY": VRY, "VYB": VYB, "VBR": VBR,
    "AVG_VLL": AVG_VLL,

    "IR": IR, "IY": IY, "IB": IB,
    "AVG_I": AVG_I,

    "R_PF": R_PF, "Y_PF": Y_PF, "B_PF": B_PF,
    "AVG_PF": AVG_PF,

    "FREQUENCY": FREQUENCY,

    "R_KW": R_KW, "Y_KW": Y_KW, "B_KW": B_KW,
    "TOTAL_KW": TOTAL_KW,

    "R_KVA": R_KVA, "Y_KVA": Y_KVA, "B_KVA": B_KVA,
    "TOTAL_KVA": TOTAL_KVA,

    "R_KVAR": R_KVAR, "Y_KVAR": Y_KVAR, "B_KVAR": B_KVAR,
    "TOTAL_KVAR": TOTAL_KVAR,

    "R_THD-V": R_THD_V, "Y_THD-V": Y_THD_V, "B_THD-V": B_THD_V,
    "R_THD-I": R_THD_I, "Y_THD-I": Y_THD_I, "B_THD-I": B_THD_I,
})

# -------------------- inject noise & nulls --------------------------
# 1) small measurement noise already present; now insert sparse "huge noise" outliers
outlier_rows = np.random.choice(df.index, size=max(3, n//400), replace=False)  # ~0.25%
num_cols_for_outlier = 4
cols_numeric = [c for c in df.columns if c not in ("TS","DID","FID","SLAVE ID")]
for r in outlier_rows:
    cols = list(np.random.choice(cols_numeric, size=num_cols_for_outlier, replace=False))
    for c in cols:
        val = df.at[r, c]
        if pd.api.types.is_numeric_dtype(df[c]):
            factor = np.random.choice([3, 5, 10, 0.1])
            df.at[r, c] = val * factor

# 2) random NULLs across dataset (skip key ID/time columns)
null_mask = np.random.rand(n, len(cols_numeric)) < 0.006  # ~0.6% cells
for j, c in enumerate(cols_numeric):
    col = df[c].to_numpy(dtype="float64", copy=True)
    col[null_mask[:, j]] = np.nan
    df[c] = col

# 3) sanity: when TOTAL_KVA==0 set derived powers/currents to 0 and PF NaN
zero_kva = (df["TOTAL_KVA"]<=1e-6) | off_mask
for c in ["TOTAL_KW","TOTAL_KVAR","IR","IY","IB","AVG_I",
          "R_KW","Y_KW","B_KW","R_KVAR","Y_KVAR","B_KVAR",
          "R_KVA","Y_KVA","B_KVA"]:
    arr = df[c].to_numpy()
    arr[zero_kva] = 0.0
    df[c] = arr
for c in ["R_PF","Y_PF","B_PF","AVG_PF"]:
    df.loc[zero_kva, c] = np.nan

# 4) Frequency zero on off already; set voltages near zero when off (keep some noise)
for c in ["VR","VY","VB","VRY","VYB","VBR","AVG_VLN","AVG_VLL"]:
    arr = df[c].to_numpy()
    arr[off_mask] = np.random.uniform(0.0, 2.0, off_mask.sum())
    df[c] = arr

# ------------------------- save & preview ---------------------------
csv_path = f"./synthetic_mfm_{DID}_5min_10days.csv"
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")
print(df.head(8))
