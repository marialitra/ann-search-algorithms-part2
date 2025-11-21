import libraries
from libraries import np, Tuple, Dict, List

def parse_neighbor_file(path: str) -> Tuple[Dict[int, List[int]], int]:
    """
        Parse the neighbor file.
        Returns: neighbors: dict query -> list of neighbor ids (ints) in order
    """

    neighbors: Dict[int, List[int]] = {}
    q_re = libraries.re.compile(r"^Query:\s*(\d+)")
    nn_re = libraries.re.compile(r"^Nearest neighbor-\d+:\s*(\d+)")

    current_q = None
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            m = q_re.match(line)
            if m:
                current_q = int(m.group(1))
                neighbors[current_q] = []
                continue

            m = nn_re.match(line)
            if m and current_q is not None:
                nid = int(m.group(1))
                neighbors[current_q].append(nid)
                continue

    return neighbors, len(neighbors)

def load_idx_images(filepath: str) -> Tuple[np.ndarray, int, int, int]:
    """
        Reads and extracts image vectors from a file in the IDX format (like MNIST).
        Returns:
            A tuple containing:
            - data_vectors: A NumPy array of shape (num_images, 1, rows, cols), 
            where 1 is the channel dimension for grayscale (0-255).
            - num_images: The total number of images found.
            - num_rows: The height of the image.
            - num_cols: The width of the image.
    """
    print(f"Loading IDX image file: {filepath}")

    if not libraries.os.path.exists(filepath):
        raise FileNotFoundError(f"Error: File not found at {filepath}")

    with open(filepath, 'rb') as f:
        header = f.read(16)

        # Unpack the header: Magic Number, Num Images, Rows, Cols
        magic, num_images, num_rows, num_cols = libraries.struct.unpack('>IIII', header)
        
        # Verify the magic number for IDX-3 (images) which should be 2051
        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")
        
        print(f"Found {num_images} images, each {num_rows}x{num_cols}.")
        vector_dimension = num_rows * num_cols
        print(f"Original flat dimension: {vector_dimension}.")

        # Calculate the total size of the pixel data to read (bytes)
        data_size = num_images * vector_dimension
        
        # Read the rest of the file contents (the pixel data)
        pixel_data = f.read(data_size)

        # Convert the byte data into a NumPy array of unsigned 8-bit integers (0-255)
        raw_data = np.frombuffer(pixel_data, dtype=np.uint8)

        # Ρeshape for CNN (Batch, Channels, Height, Width)
        # Channels=1 for grayscale
        data_vectors = raw_data.reshape(num_images, 1, num_rows, num_cols)

    print("Successfully extracted vectors.")
    return data_vectors, num_images, num_rows, num_cols

def read_fvecs(path: str) -> np.ndarray:
    """
        Read .fvecs file (each vector stored as: int32 dim, float32 dim values).
        Returns a numpy array of shape (n, dim) dtype float32.
    """
    vecs = []
    with open(path, 'rb') as fh:
        while True:
            hdr = fh.read(4)
            if not hdr:
                break
            d = libraries.struct.unpack('i', hdr)[0]
            bytes_needed = 4 * d
            buf = fh.read(bytes_needed)
            if len(buf) != bytes_needed:
                raise EOFError(f"Unexpected EOF while reading {path}")
            arr = np.frombuffer(buf, dtype=np.float32).copy()
            vecs.append(arr)
    if not vecs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(vecs)

def load_sift_vectors(filepath: str, dtype=np.float32) -> tuple[np.ndarray, int, int, int]:
    """
        Load SIFT vectors from common binary formats (.fvecs, .bvecs) and return
        a 4D tensor shaped (n, 1, 1, dim) to be compatible with the rest of the
        pipeline used by this script.
    """

    print(f"Loading SIFT vectors from {filepath} ...")

    if not libraries.os.path.exists(filepath):
        raise FileNotFoundError(f"SIFT file not found: {filepath}")

    lower = filepath.lower()
    if lower.endswith('.fvecs'):
        mat = read_fvecs(filepath).astype(np.float32)
    else:
        try:
            mat = np.load(filepath)
            if mat.ndim == 1:
                raise ValueError("Unsupported numpy shape for SIFT data")
        except Exception:
            mat = np.loadtxt(filepath)

    n, dim = mat.shape
    item_size = 4 * (dim + 1)
    data_vectors = mat.reshape(n, 1, 1, dim)
    file_size = libraries.os.path.getsize(filepath)
    n_vectors = file_size // item_size

    return data_vectors, n_vectors, 1, dim