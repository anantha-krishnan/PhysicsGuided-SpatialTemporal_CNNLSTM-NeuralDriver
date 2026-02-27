import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob, os

# --- CONFIG ---
current_file_path = Path(os.path.abspath(__file__))
current_dir = current_file_path.parent
MODEL_SAVE_PATH = current_dir.parent / "Map_Layouts" 
RESULTS_DIR = MODEL_SAVE_PATH / "benchmark_results"
PLOTS_DIR = RESULTS_DIR
PLOTS_DIR.mkdir(exist_ok=True)

# PASS THRESHOLDS
THRESH_MAX_CTE = 0.9  # Meters
THRESH_RMSE = 0.7     # Meters

def analyze_and_plot():
    files = sorted(glob.glob(str(RESULTS_DIR / "*.csv")))
    
    print(f"{'SCENARIO':<25} | {'STATUS':<6} | {'MAX CTE':<8} | {'RMSE':<8}")
    print("-" * 60)

    for filepath in files:
        df = pd.read_csv(filepath)
        name = Path(filepath).stem
        
        # 1. Metrics Calculation
        max_cte = df['cte'].abs().max()
        rmse = np.sqrt((df['cte'] ** 2).mean())
        
        # 2. Pass/Fail Logic
        passed = max_cte < THRESH_MAX_CTE and rmse < THRESH_RMSE
        status = "PASS" if passed else "FAIL"
        
        # Console Report
        print(f"{name:<25} | {status:<6} | {max_cte:.3f}m  | {rmse:.3f}m")
        
        # 3. Generate Plots
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Analysis: {name} [{status}]", fontsize=16)
        
        # Plot A: Trajectory Overlay (Top Left)
        axs[0, 0].set_title("Trajectory: Reference vs Actual")
        axs[0, 0].plot(df['ref_x'], df['ref_y'], 'k--', label='Reference', alpha=0.7)
        axs[0, 0].plot(df['act_x'], df['act_y'], 'b-', label='Neural Driver', linewidth=2)
        axs[0, 0].axis('equal')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # Plot B: Cross Track Error (Bottom Left)
        axs[1, 0].set_title("Cross Track Error (CTE)")
        axs[1, 0].plot(df['step'], df['cte'], 'r-')
        axs[1, 0].axhline(y=THRESH_MAX_CTE, color='orange', linestyle='--', label='Limit')
        axs[1, 0].axhline(y=-THRESH_MAX_CTE, color='orange', linestyle='--')
        axs[1, 0].set_ylabel("Error (m)")
        axs[1, 0].set_ylim(-1.5, 1.5)
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # Plot C: Steering vs Curvature (Top Right)
        # This checks "Phase Lag" - does steer happen at the same time as curve?
        axs[0, 1].set_title("Reaction: Steer Cmd vs Path Curvature")
        ax2 = axs[0, 1].twinx()
        l1, = axs[0, 1].plot(df['step'], df['steer_cmd'], 'b-', label='Steer Cmd')
        l2, = ax2.plot(df['step'], df['curvature_input'], 'g--', label='Path Curvature', alpha=0.6)
        axs[0, 1].set_ylim(-1.1, 1.1)
        axs[0, 1].legend([l1, l2], ['Steer', 'Curvature'])
        axs[0, 1].grid(True)
        
        # Plot D: Heading Error (Bottom Right)
        axs[1, 1].set_title("Heading Error")
        axs[1, 1].plot(df['step'], np.degrees(df['heading_error']), 'm-')
        axs[1, 1].set_ylabel("Degrees")
        axs[1, 1].grid(True)
        
        # Save Plot
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{name}_analysis.png")
        plt.close()

    print("-" * 60)
    print(f"Analysis complete. Check '{PLOTS_DIR}' for images.")

if __name__ == "__main__":
    analyze_and_plot()