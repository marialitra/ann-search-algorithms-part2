import libraries

def build_executable():
    """
    Runs 'make search' to compile the executable.
    """
    if libraries.os.path.exists("./Ivfflat/search"):
        print("--- Executable './Ivfflat/search' found. Skipping build. ---")
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

def run_ivfflat(command_list):
    """
    Runs the ./search executable with optimized parameters for speed.
    """

    # --- 2. OPTIMIZATION: Auto-detect CPU cores ---
    # Use all available cores for maximum parallelism
    num_cores = str(libraries.multiprocessing.cpu_count())
    print(f"Detected {num_cores} CPU cores. Setting OMP_NUM_THREADS.")

    run_env = libraries.os.environ.copy()
    # Setting threads to match core count usually gives best performance
    run_env["OMP_NUM_THREADS"] = num_cores
    run_env["OMP_NESTED"] = "TRUE"
    run_env["OMP_MAX_ACTIVE_LEVELS"] = "2"

    print(f"Running command: {' '.join(command_list)}")

    try:
        libraries.subprocess.run(
            command_list,
            env=run_env,
            text=True,
            check=True
        )
        print("\n--- Run complete. ---")

    except libraries.subprocess.CalledProcessError as e:
        print(f"--- ERROR: Run failed with return code {e.returncode} ---")
    except FileNotFoundError:
        print("--- ERROR: './search' executable not found. ---")