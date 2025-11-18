import libraries
from libraries import Tuple, Dict, List, np, nn, counter, load_sift_vectors, load_idx_images, mnist_train, sift_train, parse_neighbor_file, build_csr_from_neighbors, save_output

def main():
    p = libraries.argparse.ArgumentParser(description="Build adjacency matrix (CSR) from neighbor TXT files.")
    p.add_argument("input", type=str, help="path to neighbor TXT file")
    p.add_argument("-d", type=str, help="path to dataset file")
    p.add_argument("-i", type=str, help="path to index file")
    p.add_argument("--type", type=str, help="dataset type (MNIST or SIFT)")
    p.add_argument("--nodes", type=int, default=128, help="number of neurons in FC layers (less important for CNN)")
    p.add_argument("--layers", type=int, default=3, help="number of layers (less important for CNN)")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("-m", type=int, required=True, help="number of blocks")
    p.add_argument("--epochs", type=int, default=10, help="number of epochs in training loop")
    p.add_argument("--batch_size", type=int, default=64, help="batch size")
    args = p.parse_args()

    inp = args.input
    if not libraries.os.path.exists(inp):
        raise SystemExit(f"File not found: {inp}")

    dataset_type = args.type
    print(f"The specified dataset type is: {dataset_type}")

    # Load data: MNIST-like IDX images or SIFT vectors
    if dataset_type and dataset_type.lower().startswith('sift'):
        data_vectors, num_images, img_rows, img_cols = load_sift_vectors(args.d)
    else:
        # default: IDX images (MNIST)
        data_vectors, num_images, img_rows, img_cols = load_idx_images(args.d)

    neighbors, datasetsize = parse_neighbor_file(inp)
    print(f"Parsed {datasetsize} queries. Building adjacency...")

    xadj, adjncy, adjwgt, vwgt = build_csr_from_neighbors(neighbors, datasetsize)

    IMBALANCE = 0.03
    SEED = 42

    # Call kahip partitioner
    edgecut, blocks = libraries.kahip.kaffpa(vwgt, xadj, adjwgt, adjncy, args.m, IMBALANCE, True, SEED, 1)
    print(f"Partitioned graph into {args.m} blocks with edgecut {edgecut}.")
    # print(blocks)

    X = data_vectors    # shape (n, 1, rows, cols)
    y = np.array(blocks)  # labels from KaHIP
    

    if dataset_type and dataset_type.lower().startswith('sift'):
        model = sift_train(args, img_rows, img_cols, X, y)
    else:
        # Train the new CNN model, passing image dimensions
        model = mnist_train(args, img_rows, img_cols, X, y)

    libraries.os.makedirs(args.i, exist_ok=True)

    # Save the model and the inverted index file
    save_output(model, args.i, X.copy(), y.copy(), img_rows, img_cols)

if __name__ == "__main__":
    main()