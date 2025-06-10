"""
Runs CMA-ES sequentially for multiple instances.
Creates an output folder with a timestamp for the experiment,
and runs the specified number of instances one after another.
"""

import argparse
from run_cmaes import run
from datetime import datetime
from snn.model_struct import SPIKE_DECAY_DEFAULT


def run_cmaes_instance(mode, gens, sigma, hidden_sizes, output_folder, 
                       run_number, spike_decay=SPIKE_DECAY_DEFAULT, robot_config_path=None):   
    """
    Wrapper to call the run() function from run_cmaes.py with provided parameters.
    """
    print(f"\n[Run {run_number}] Starting CMA-ES run...")
    run(
        mode=mode,
        gens=gens,
        sigma_val=sigma,
        hidden_sizes=hidden_sizes,
        output_folder=output_folder,
        run_number=run_number,
        spike_decay=spike_decay,
        robot_config_path=robot_config_path
    )
    print(f"[Run {run_number}] Finished CMA-ES run.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequential CMA-ES runner')

    parser.add_argument('--mode', default="h",
                        help='Output mode: h-headless, s-screen, v-video, b-both')
    parser.add_argument('--gens', type=int, default=100,
                        help='Number of generations to run')
    parser.add_argument('--sigma', type=float, default=3,
                        help='Sigma value for CMA-ES')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[2,2])
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Output folder for results')
    parser.add_argument('--runs', type=int, default=5,
                        help='Total number of CMA-ES runs to execute')
    parser.add_argument('--spike_decay', type=float,
                        default=SPIKE_DECAY_DEFAULT, help='Spike decay rate for neurons')
    parser.add_argument('--robot_config', type=str,
                        help='Robot config path', default="bestbot.json")
    args = parser.parse_args()

    # Create a timestamped output folder for all runs
    if args.exp_name == None:
        file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        file_name = args.exp_name

    for run_number in range(1, args.runs + 1):
                run_cmaes_instance(
            mode=args.mode,
            gens=args.gens,
            sigma=args.sigma,
            hidden_sizes=args.hidden_sizes,
            output_folder=file_name,
            run_number=run_number,
            spike_decay=args.spike_decay,
            robot_config_path=args.robot_config
        )

    print("\nAll CMA-ES runs completed.")