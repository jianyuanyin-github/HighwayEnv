#!/usr/bin/env python3
"""
Generate clean track visualizations for paper figures.
No grids, no axes, just track layouts.
"""
import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches


def visualize_track(env_id, ax, title):
    """
    Visualize a single track/road network.

    Args:
        env_id: Environment ID (e.g., 'racetrack-complex-v0')
        ax: Matplotlib axis to draw on
        title: Title for the subplot
    """
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset()

    road_network = env.unwrapped.road.network

    # Special handling for intersection: draw simple cross
    if "intersection" in env_id.lower():
        # Draw a simple cross: one horizontal road + one vertical road
        road_width = 8
        road_length = 120

        # Horizontal road (full length with grey background)
        h_poly = Polygon(
            [
                [-road_length / 2, road_width / 2],
                [road_length / 2, road_width / 2],
                [road_length / 2, -road_width / 2],
                [-road_length / 2, -road_width / 2],
            ],
            facecolor="#E0E0E0",
            edgecolor="none",
            zorder=1,
        )
        ax.add_patch(h_poly)

        # Vertical road (full length with grey background)
        v_poly = Polygon(
            [
                [road_width / 2, -road_length / 2],
                [road_width / 2, road_length / 2],
                [-road_width / 2, road_length / 2],
                [-road_width / 2, -road_length / 2],
            ],
            facecolor="#E0E0E0",
            edgecolor="none",
            zorder=1,
        )
        ax.add_patch(v_poly)

        # Horizontal boundaries (break at intersection)
        gap = road_width  # Gap at intersection
        ax.plot(
            [-road_length / 2, -gap / 2],
            [road_width / 2, road_width / 2],
            "k-",
            linewidth=2,
            zorder=2,
        )
        ax.plot(
            [gap / 2, road_length / 2],
            [road_width / 2, road_width / 2],
            "k-",
            linewidth=2,
            zorder=2,
        )
        ax.plot(
            [-road_length / 2, -gap / 2],
            [-road_width / 2, -road_width / 2],
            "k-",
            linewidth=2,
            zorder=2,
        )
        ax.plot(
            [gap / 2, road_length / 2],
            [-road_width / 2, -road_width / 2],
            "k-",
            linewidth=2,
            zorder=2,
        )

        # Vertical boundaries (break at intersection)
        ax.plot(
            [road_width / 2, road_width / 2],
            [-road_length / 2, -gap / 2],
            "k-",
            linewidth=2,
            zorder=2,
        )
        ax.plot(
            [road_width / 2, road_width / 2],
            [gap / 2, road_length / 2],
            "k-",
            linewidth=2,
            zorder=2,
        )
        ax.plot(
            [-road_width / 2, -road_width / 2],
            [-road_length / 2, -gap / 2],
            "k-",
            linewidth=2,
            zorder=2,
        )
        ax.plot(
            [-road_width / 2, -road_width / 2],
            [gap / 2, road_length / 2],
            "k-",
            linewidth=2,
            zorder=2,
        )

        # Center dashed lines (full length, continuous through intersection)
        ax.plot(
            [-road_length / 2, road_length / 2],
            [0, 0],
            "k--",
            linewidth=1,
            alpha=0.5,
            zorder=3,
        )
        ax.plot(
            [0, 0],
            [-road_length / 2, road_length / 2],
            "k--",
            linewidth=1,
            alpha=0.5,
            zorder=3,
        )

    else:
        # Normal handling for other environments: draw all lanes
        for start_node in road_network.graph.keys():
            for end_node in road_network.graph[start_node].keys():
                lanes = road_network.graph[start_node][end_node]
                for lane in lanes:
                    # High-density sampling for smooth curves
                    if hasattr(lane, "length"):
                        num_points = max(50, int(abs(lane.length) / 0.5))
                        s_vals = np.linspace(0, lane.length, num_points)
                    else:
                        num_points = max(200, int(abs(lane.length) / 0.2))
                        s_vals = np.linspace(0, abs(lane.length), num_points)

                    # Get center line
                    center_positions = np.array([lane.position(s, 0) for s in s_vals])

                    # Get left and right boundaries
                    left_positions = np.array(
                        [lane.position(s, lane.width / 2) for s in s_vals]
                    )
                    right_positions = np.array(
                        [lane.position(s, -lane.width / 2) for s in s_vals]
                    )

                    # Fill the lane area with light grey
                    lane_polygon = np.vstack([left_positions, right_positions[::-1]])
                    poly = Polygon(
                        lane_polygon, facecolor="#E0E0E0", edgecolor="none", zorder=1
                    )
                    ax.add_patch(poly)

                    # Draw boundaries
                    ax.plot(
                        left_positions[:, 0],
                        left_positions[:, 1],
                        "k-",
                        linewidth=2,
                        solid_capstyle="round",
                        zorder=2,
                    )
                    ax.plot(
                        right_positions[:, 0],
                        right_positions[:, 1],
                        "k-",
                        linewidth=2,
                        solid_capstyle="round",
                        zorder=2,
                    )

                    # Draw center line (dashed)
                    ax.plot(
                        center_positions[:, 0],
                        center_positions[:, 1],
                        "k--",
                        linewidth=1,
                        alpha=0.5,
                        zorder=3,
                    )

    # Clean up the plot
    ax.set_aspect("equal")
    ax.axis("off")  # Remove axes
    # ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    # Set tight limits
    ax.margins(0.05)

    env.close()
    return ax


def main():
    """Generate paper figure with all environments."""

    # Define environments to visualize
    environments = [
        ("racetrack-complex-v0", "Complex Racetrack"),
        ("u-turn-v0", "U-Turn"),
        ("intersection-v0", "Intersection"),
        ("merge-v0", "Merge"),
        ("roundabout-v0", "Roundabout"),
    ]

    # Create figure with custom layout using GridSpec
    fig = plt.figure(figsize=(18, 12))
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # First row: 3 subplots
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Second row: 2 subplots (left and center positions for better visual balance)
    ax3 = fig.add_subplot(gs[1, 0:2])  # Merge - spans columns 0-1
    ax4 = fig.add_subplot(
        gs[1, 1:3]
    )  # Roundabout - spans columns 1-2 (overlaps, will adjust)

    axes = [ax0, ax1, ax2, ax3, ax4]

    # Visualize each environment
    for idx, (env_id, title) in enumerate(environments):
        print(f"Rendering {title}...")
        try:
            visualize_track(env_id, axes[idx], title)
            print(f"  ✓ {title} completed")
        except Exception as e:
            print(f"  ✗ Error rendering {title}: {e}")
            axes[idx].text(
                0.5,
                0.5,
                f"Error: {title}",
                ha="center",
                va="center",
                transform=axes[idx].transAxes,
            )
            axes[idx].axis("off")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_file = "paper_tracks_combined.png"
    plt.savefig(
        output_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"\n✓ Combined figure saved to: {output_file}")

    # Also save individual high-res versions
    print("\nGenerating individual high-resolution figures...")
    for env_id, title in environments:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(10, 10))
        try:
            visualize_track(env_id, ax_single, title)
            filename = f"paper_{title.lower().replace(' ', '_').replace('-', '_')}.png"
            plt.savefig(
                filename,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            print(f"  ✓ Saved {filename}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        plt.close(fig_single)

    plt.close(fig)
    print("\n✓ All figures generated successfully!")


if __name__ == "__main__":
    main()
