# DRL-MPCC Integration

Safety-aware integration of Deep Reinforcement Learning with Model Predictive Contouring Control.

## Architecture

```
DRL Policy → Reference (a_ref, δ_ref) → MPCC with Safety Constraints → Safe Control Execution
```

The cost function implements time-decaying feedforward:
```
J = β·ω(t)·J_ref + (1-ω(t))·J_t
```
where `ω(t) = exp(-α·t)` gradually transitions from DRL guidance to pure MPCC optimization.

## File Structure

```
integration/
├── config_integration.yaml           # Configuration (switch modes, tune parameters)
├── forces_pro_setting_with_drl.py    # DRL-enabled ForcesProforces solver (23 params)
├── mpcc_drl_wrapper.py               # MPCC wrapper supporting both modes
├── drl_mpcc_integration.py           # Main integration controller class
├── test_step_by_step.py              # Step-by-step testing script
├── test_drl_mpcc_integration.py      # Full integration test
└── README.md                         # This file
```

## Quick Start

### Step 1: Run Tests

```bash
cd /home/yinjianyuan/MA/HighwayEnv

# Test all components
python integration/test_step_by_step.py
```

This will check:
- ✓ Module imports
- ✓ Configuration files
- ✓ Original MPCC solver
- ✓ Track extraction from highway_env
- ✓ Parameter vector construction
- ✓ Time-decay function

### Step 2: Generate Solvers (if needed)

#### Original MPCC Solver
If you don't have `./dynamics_solver_forcespro/`:
```bash
# Edit config/mpccparams.yaml
generate_solver: true

# Run once
python main.py

# Set back to false
generate_solver: false
```

#### DRL-MPCC Solver
To generate `./dynamics_solver_forcespro_drl/`:
```bash
# Edit integration/config_integration.yaml
solver:
  generate_solver: true

# Generate solver
python integration/test_drl_mpcc_integration.py

# Set back to false
solver:
  generate_solver: false
```

### Step 3: Choose Control Mode

Edit `integration/config_integration.yaml`:

#### Pure MPCC (Baseline)
```yaml
control_mode: "pure_mpcc"
```

#### DRL-MPCC Integrated
```yaml
control_mode: "drl_mpcc"

integration:
  alpha: 0.1      # Time decay rate: ω(t) = exp(-α·t)
  beta: 1.0       # DRL reference weight (0=pure MPCC, 1=full DRL tracking)
  K_a: 100.0      # Acceleration tracking weight
  K_delta: 100.0  # Steering tracking weight
```

### Step 4: Run Integration Test

```bash
python integration/test_drl_mpcc_integration.py
```

This will:
1. Extract track from `racetrack-single-v0`
2. Run pure MPCC (baseline)
3. Run DRL-MPCC integrated mode
4. Generate comparison plots in `integration/results/`

## Configuration Parameters

### Time Decay (`alpha`)
Controls how fast DRL influence decays along prediction horizon:

| α    | ω(0) | ω(T/2) | ω(T) | Behavior |
|------|------|--------|------|----------|
| 0.05 | 1.0  | 0.951  | 0.905| Slow decay, DRL dominates |
| 0.1  | 1.0  | 0.905  | 0.819| Balanced (default) |
| 0.2  | 1.0  | 0.819  | 0.670| Fast decay, MPCC dominates |
| 0.5  | 1.0  | 0.607  | 0.368| Very fast, mostly MPCC |

### Overall Weight (`beta`)
Controls global DRL influence:
- `beta = 0.0`: Pure MPCC (ignores DRL reference)
- `beta = 0.5`: 50/50 blend
- `beta = 1.0`: Full DRL tracking (default)

### Tracking Weights (`K_a`, `K_delta`)
Higher values = tighter tracking of DRL reference:
- `K_a`: Acceleration tracking stiffness
- `K_delta`: Steering tracking stiffness

## Usage in Code

### Pure MPCC Mode
```python
from integration.mpcc_drl_wrapper import MPCCDRLWrapper

mpcc = MPCCDRLWrapper(
    track, Tsim, vehicleparams, mpccparams,
    control_mode="pure_mpcc"
)

# Initialize
mpcc.initialize_trajectory(xinit, None, startidx)

# Control loop
for step in range(Nsim):
    z_current = mpcc.update(None)
```

### DRL-MPCC Integrated Mode
```python
from integration.mpcc_drl_wrapper import MPCCDRLWrapper
from integration.drl_mpcc_integration import SafetyAwareDRLMPCC

# 1. Create DRL-enabled MPCC
mpcc = MPCCDRLWrapper(
    track, Tsim, vehicleparams, mpccparams,
    control_mode="drl_mpcc"
)

# 2. Create integration controller
integrated = SafetyAwareDRLMPCC(
    mpcc_controller=mpcc,
    drl_policy=trained_ppo_model,  # Your trained PPO policy
    alpha=0.1,
    beta=1.0,
    K_a=100.0,
    K_delta=100.0
)

# 3. Initialize
mpcc.initialize_trajectory(xinit, None, startidx)

# 4. Control loop with DRL reference
for step in range(Nsim):
    result = integrated.step(
        obs=current_observation,
        enemyinfo=None,
        use_drl=True
    )

    z_current = result['z_current']
    u_ref = result['u_ref']        # DRL reference
    u_actual = result['u_actual']  # MPCC executed control
```

## Solver Parameter Dimensions

| Mode      | Parameters | Description |
|-----------|------------|-------------|
| pure_mpcc | 17         | Standard MPCC (xt, yt, ..., Qc, Ql, ...) |
| drl_mpcc  | 23         | +6 DRL params (a_ref, δ_ref, ω, K_a, K_delta, β) |

All parameters are **runtime-modifiable** (no recompilation needed).

## Tracking Error Analysis

After running integrated mode:
```python
stats = integrated.get_tracking_error()
print(f"RMSE acceleration: {stats['rmse_a']:.4f} m/s²")
print(f"RMSE steering: {stats['rmse_delta']:.4f} rad")
print(f"Max error accel: {stats['max_error_a']:.4f} m/s²")
print(f"Max error steering: {stats['max_error_delta']:.4f} rad")
```

## Troubleshooting

### "Solver not found" error
→ Generate solver first (see Step 2)

### "Parameter dimension mismatch"
→ Check `control_mode` matches solver:
  - `pure_mpcc` → uses `./dynamics_solver_forcespro/`
  - `drl_mpcc` → uses `./dynamics_solver_forcespro_drl/`

### DRL policy returns wrong action dimension
→ Ensure PPO policy outputs 2D continuous actions: `[a, δ]`

### Tracking error too large
→ Increase tracking weights `K_a` and `K_delta`

## Paper Implementation Verification

✅ **Equation (2)**: Combined cost function
```python
cost = beta * omega * J_ref + (1.0 - omega) * J_t
```

✅ **Equation (3)**: Reference tracking term
```python
J_ref = K_a * (a - a_ref)² + K_delta * (δ - δ_ref)²
```

✅ **Time decay**: `ω(t) = exp(-α·t)`
```python
omega = np.exp(-alpha * t_k)
```

## Next Steps

1. **Train DRL policy** that outputs `(a_ref, δ_ref)`
2. **Load trained model** into `SafetyAwareDRLMPCC`
3. **Tune parameters** (`α`, `β`, `K_a`, `K_delta`) for your scenario
4. **Compare performance** vs pure MPCC baseline
5. **Validate safety** constraints are always satisfied

## References

See paper: "Safety-Aware Reinforcement Learning with MPCC Feedforward Integration"
