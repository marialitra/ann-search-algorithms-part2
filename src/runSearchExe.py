import libraries

def build_executable():
    """
    Runs 'make search' to compile the executable.
    """
    if libraries.os.path.exists("./search"):
        print("--- Executable './search' found. Skipping build. ---")
        return True

    print("--- 1. Building executable (running 'make search')... ---")

    try:
        build_process = libraries.subprocess.run(["make", "search"], capture_output=True, text=True, check=True)
        print("Build complete: './search' is ready.")
        return True
    except libraries.subprocess.CalledProcessError as e:
        print("--- ERROR: Build failed. ---")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
         print("--- ERROR: 'make' command not found. Is it installed? ---")
         return False

def run_ivfflat_sift(command_list):
    """
    Runs the ./search executable with optimized parameters for speed.
    """
    print("\n--- 2. Running SIFT with OPTIMIZED parameters... ---")

    # --- 1. SETUP PATHS ---
    dataset_file = "./Data/SIFT/sift_base_100k.fvecs"

    if not libraries.os.path.exists(dataset_file):
        print(f"--- ERROR: Dataset file '{dataset_file}' not found! ---")
        return

    # --- 2. OPTIMIZATION: Auto-detect CPU cores ---
    # Use all available cores for maximum parallelism
    num_cores = str(libraries.multiprocessing.cpu_count())
    print(f"Detected {num_cores} CPU cores. Setting OMP_NUM_THREADS.")

    run_env = libraries.os.environ.copy()
    # Setting threads to match core count usually gives best performance
    run_env["OMP_NUM_THREADS"] = num_cores
    run_env["OMP_NESTED"] = "TRUE"
    run_env["OMP_MAX_ACTIVE_LEVELS"] = "2"

    # --- 3. OPTIMIZATION: Tuning Parameters ---
    # Original: -kclusters 40, -nprobe 3
    # Optimized for 100k points:
    #   - kclusters: Increased to 100 (closer to sqrt(N)) to make buckets smaller/faster
    #   - nprobe: Reduced to 1 for maximum speed (trade-off: lower accuracy)

    k_clusters = "40"
    n_probe = "3"

    print(f"Running command: {' '.join(command_list)}")
    print(f"Optimizations: kclusters={k_clusters}, nprobe={n_probe}, threads={num_cores}")

    try:
        libraries.subprocess.run(
            command_list,
            env=run_env,
            text=True,
            check=True
        )
        print("\n--- SIFT run complete. ---")

    except libraries.subprocess.CalledProcessError as e:
        print(f"--- ERROR: SIFT run failed with return code {e.returncode} ---")
    except FileNotFoundError:
        print("--- ERROR: './search' executable not found. ---")

# --- Main execution ---
if __name__ == "__main__":
    if build_executable():
        run_ivfflat_sift()