"""
Action smoothing wrapper for reducing control oscillation.

Implements:
1. Exponential Moving Average (EMA): ã_t = α·ã_{t-1} + (1-α)·a_t
2. Rate limiting: |ã_t - ã_{t-1}| ≤ Δ_max
"""

import numpy as np
from typing import Optional


class SmoothAction:
    """
    Action smoother using EMA and rate limiting.

    Args:
        action_dim: Dimension of action space (e.g., 2 for [acceleration, steering])
        alpha: EMA smoothing factor (0.85-0.95 recommended)
               Higher α → smoother but slower response
        rate_limit_pct: Rate limit as percentage of action range (3-8% recommended)
                        e.g., 0.05 means max 5% change per step
        action_low: Lower bound of action space (e.g., [-1, -1])
        action_high: Upper bound of action space (e.g., [1, 1])
    """

    def __init__(
        self,
        action_dim: int = 2,
        alpha: float = 0.9,
        rate_limit_pct: float = 0.05,
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
    ):
        self.action_dim = action_dim
        self.alpha = alpha
        self.rate_limit_pct = rate_limit_pct

        # Default action bounds (normalized [-1, 1])
        if action_low is None:
            action_low = -np.ones(action_dim)
        if action_high is None:
            action_high = np.ones(action_dim)

        self.action_low = np.array(action_low)
        self.action_high = np.array(action_high)
        self.action_range = self.action_high - self.action_low

        # Compute rate limit (Δ_max)
        self.delta_max = self.rate_limit_pct * self.action_range

        # Previous smoothed action
        self.prev_action_smooth = None

        print(f"[ActionSmoothing] Initialized:")
        print(f"  Alpha (EMA): {self.alpha}")
        print(f"  Rate limit: {self.rate_limit_pct*100:.1f}% of range")
        print(f"  Delta_max: {self.delta_max}")

    def smooth(self, action: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing and rate limiting to action.

        Args:
            action: Raw action from policy, shape (action_dim,)

        Returns:
            smoothed_action: Smoothed and rate-limited action
        """
        action = np.array(action)

        # First step: initialize with raw action
        if self.prev_action_smooth is None:
            self.prev_action_smooth = action.copy()
            return action

        # Step 1: EMA smoothing
        # ã_t = α·ã_{t-1} + (1-α)·a_t
        action_ema = self.alpha * self.prev_action_smooth + (1 - self.alpha) * action

        # Step 2: Rate limiting
        # Δ = ã_t - ã_{t-1}
        delta = action_ema - self.prev_action_smooth

        # Clip to |Δ| ≤ Δ_max
        delta_clipped = np.clip(delta, -self.delta_max, self.delta_max)

        # Final smoothed action
        action_smooth = self.prev_action_smooth + delta_clipped

        # Ensure within action bounds
        action_smooth = np.clip(action_smooth, self.action_low, self.action_high)

        # Update previous action
        self.prev_action_smooth = action_smooth.copy()

        return action_smooth

    def reset(self):
        """Reset the smoother (call at episode start)."""
        self.prev_action_smooth = None

    def get_stats(self) -> dict:
        """Get smoothing statistics."""
        return {
            "alpha": self.alpha,
            "rate_limit_pct": self.rate_limit_pct,
            "delta_max": self.delta_max.tolist(),
            "prev_action": self.prev_action_smooth.tolist() if self.prev_action_smooth is not None else None,
        }


class ActionSmoothingWrapper:
    """
    Wrapper for AbstractEnv to add action smoothing.

    Usage:
        env = gym.make("racetrack-single-v0")
        env = ActionSmoothingWrapper(env, alpha=0.9, rate_limit_pct=0.05)
    """

    def __init__(
        self,
        env,
        alpha: float = 0.9,
        rate_limit_pct: float = 0.05,
    ):
        self.env = env

        # Determine action dimension
        if hasattr(env.action_space, 'shape'):
            action_dim = env.action_space.shape[0]
        else:
            action_dim = 1

        # Get action bounds
        if hasattr(env.action_space, 'low'):
            action_low = env.action_space.low
            action_high = env.action_space.high
        else:
            action_low = None
            action_high = None

        # Create smoother
        self.smoother = SmoothAction(
            action_dim=action_dim,
            alpha=alpha,
            rate_limit_pct=rate_limit_pct,
            action_low=action_low,
            action_high=action_high,
        )

    def step(self, action):
        """Step with smoothed action."""
        action_smooth = self.smoother.smooth(action)
        return self.env.step(action_smooth)

    def reset(self, **kwargs):
        """Reset environment and smoother."""
        self.smoother.reset()
        return self.env.reset(**kwargs)

    def __getattr__(self, name):
        """Delegate attribute access to wrapped env."""
        return getattr(self.env, name)


# Example usage
if __name__ == "__main__":
    # Test the smoother
    smoother = SmoothAction(
        action_dim=2,
        alpha=0.9,
        rate_limit_pct=0.05,
    )

    print("\n=== Testing Action Smoothing ===\n")

    # Simulate noisy actions
    np.random.seed(42)
    raw_actions = []
    smooth_actions = []

    for i in range(20):
        # Simulate noisy policy output
        if i < 10:
            action_raw = np.array([0.5 + 0.3 * np.random.randn(),
                                   0.2 + 0.2 * np.random.randn()])
        else:
            action_raw = np.array([0.8 + 0.3 * np.random.randn(),
                                   -0.3 + 0.2 * np.random.randn()])

        action_smooth = smoother.smooth(action_raw)

        raw_actions.append(action_raw)
        smooth_actions.append(action_smooth)

        if i % 5 == 0:
            print(f"Step {i:2d}: Raw={action_raw}, Smooth={action_smooth}, "
                  f"Delta={action_smooth - smoother.prev_action_smooth if i > 0 else 0}")

    raw_actions = np.array(raw_actions)
    smooth_actions = np.array(smooth_actions)

    # Compute statistics
    raw_jerk = np.std(np.diff(raw_actions, axis=0), axis=0)
    smooth_jerk = np.std(np.diff(smooth_actions, axis=0), axis=0)

    print(f"\n=== Statistics ===")
    print(f"Raw action jerk std:    {raw_jerk}")
    print(f"Smooth action jerk std: {smooth_jerk}")
    reduction = (1 - smooth_jerk/raw_jerk) * 100
    print(f"Jerk reduction:         Accel: {reduction[0]:.1f}%, Steering: {reduction[1]:.1f}%")
