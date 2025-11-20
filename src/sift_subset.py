import libraries
from libraries import np

def subsample_fvecs(input_file, output_file, k=100000, seed=42):
    """
    Creates a new .fvecs file by randomly subsampling k vectors
    from an input .fvecs file in a memory-efficient way.

    Args:
        input_file (str): Path to the source .fvecs file (e.g., 'sift_base.fvecs').
        output_file (str): Path to write the new subsampled file.
        k (int): The number of random vectors to subsample.
        seed (int): Random seed for reproducibility.
    """

    if not libraries.os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        print("Please download 'sift_base.fvecs' and place it in the same directory.")
        return

    print(f"Opening {input_file} to get metadata...")

    try:
        with open(input_file, 'rb') as f_in:
            # --- 1. Get metadata ---
            # Read the dimension of the first vector
            d_bytes = f_in.read(4)
            d = libraries.struct.unpack('i', d_bytes)[0]

            # Calculate the size of one vector record in bytes
            # (4 bytes for the int 'd' + d * 4 bytes for the 'd' floats)
            record_size = 4 + (d * 4)

            # Get the total file size
            f_in.seek(0, 2)
            total_size = f_in.tell()

            # Get the total number of vectors
            n = total_size // record_size

            if k > n:
                print(f"Warning: k ({k}) is larger than total vectors ({n}).")
                print(f"Will copy all {n} vectors.")
                k = n

            print(f"Input file has {n} vectors of dimension {d}.")
            print(f"Record size is {record_size} bytes.")
            print(f"Creating subsample of {k} vectors...")

            # --- 2. Generate random indices ---
            np.random.seed(seed)
            # Get k unique random indices from the range [0, n)
            indices = np.random.permutation(n)[:k]

            # --- 3. Sort indices for efficient reading ---
            # Sorting allows us to read the file sequentially,
            # which is much faster than random disk seeks.
            indices.sort()

            print(f"Generated {k} random indices. Writing to {output_file}...")

            # --- 4. Read selected vectors and write to new file ---
            f_in.seek(0) # Rewind to start

            with open(output_file, 'wb') as f_out:
                for i in indices:
                    # Seek to the start of the i-th vector record
                    offset = i * record_size
                    f_in.seek(offset)

                    # Read the entire record (4 + d*4 bytes)
                    record_bytes = f_in.read(record_size)

                    # Write the exact same bytes to the new file
                    f_out.write(record_bytes)

            print("\n--- Subsampling Complete ---")
            print(f"Successfully created {output_file} with {k} vectors.")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Clean up partial file if it exists
        if libraries.os.path.exists(output_file):
            libraries.os.remove(output_file)

# --- Main execution ---
if __name__ == "__main__":
    INPUT_FILE = 'Data/SIFT/sift_base.fvecs'
    OUTPUT_FILE = './Data/SIFT/sift_base_100k.fvecs'
    N_SUBSAMPLE = 100000

    subsample_fvecs(INPUT_FILE, OUTPUT_FILE, k=N_SUBSAMPLE)