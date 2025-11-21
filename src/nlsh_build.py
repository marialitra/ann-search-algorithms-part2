import libraries
from libraries import Tuple, Dict, List, np, nn, counter, load_sift_vectors, load_idx_images, mnist_train, sift_train, parse_neighbor_file, build_csr_from_neighbors, save_builds_output, _slug, build_executable, run_ivfflat

def main():
    p = libraries.argparse.ArgumentParser(description="Build adjacency matrix (CSR) from neighbor TXT files.")
    p.add_argument("-d", "--dataset", required=True, type=str, help="Path to input file")
    p.add_argument("-i", "--index", required=True, type=str, help="Path to index file")
    p.add_argument("--type", required=True, type=str, help="Dataset type (MNIST or SIFT)")
    p.add_argument("--knn", type=int, default=10, help="Number of neighbors")
    p.add_argument("-m", type=int, default=100, help="Number of blocks/parts for KaHIP")
    p.add_argument("--imbalance", type=float, default=0.03, help="Imbalance for KaHIP")
    p.add_argument("--kahip_mode", type=int, default=2, help="Setting for KaHIP")
    p.add_argument("--layers", type=int, default=3, help="Number of layers")
    p.add_argument("--nodes", type=int, default=64, help="Number of neurons in FC layers (less important for CNN)")
    p.add_argument("--epochs", type=int, default=10, help="Number of epochs in training loop")
    p.add_argument("--batch_size", type=int, default=128, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--seed", type=int, default=1, help="Seed number for reproducibility")
    args = p.parse_args()
    
    if not libraries.os.path.exists(args.dataset):
        raise SystemExit(f"File not found: {args.dataset}")

    dataset_type = args.type
    print(f"The specified dataset type is: {dataset_type}")

    # Load data: MNIST-IDX images or SIFT vectors
    if dataset_type and dataset_type.lower().startswith('sift'):
        data_vectors, num_images, img_rows, img_cols = load_sift_vectors(args.dataset)
        dataset_type = "sift"
        root = int( np.sqrt(num_images) )
        kclusters = str(root * 2 )
        n_probe = str( int(kclusters)//400  if int(kclusters)//400 > 1 else 2 )
    elif dataset_type and dataset_type.lower().startswith('mnist'):
        data_vectors, num_images, img_rows, img_cols = load_idx_images(args.dataset)
        root = int( np.sqrt(num_images) )
        dataset_type = "mnist"
        kclusters = str(root * 2 )
        n_probe = str( int(kclusters)//95  if int(kclusters)//95 > 1 else 2 )
    else:
        print("Not acceptable dataset type")
        exit()

    # Build or load the kNN graph using IVFFLAT
    output_dir = "knngraphs"
    libraries.os.makedirs(output_dir, exist_ok=True)
    
    raw_knn_filename = f"knngraph_{_slug(args.dataset)}_N{args.knn}.txt"
    knn_graph = libraries.os.path.join(output_dir, raw_knn_filename)

    print(f"kngraph name is: {knn_graph}")

    command_list = [
        "./Ivfflat/search",
        "-d", args.dataset,
        "-q", args.dataset,
        "-kclusters", kclusters,
        "-nprobe", n_probe,
        "-o", knn_graph,
        "-N", str(args.knn + 1),
        "-R", "2",
        "-type", dataset_type,
        "-range", "false",
        "-ivfflat",
        "-seed", str(args.seed)
    ]
    
    if not libraries.os.path.exists(knn_graph):
        # Find the knn graph
        if build_executable():
            run_ivfflat(command_list)
        else:
            raise SystemExit("Build failed. Cannot proceed to run IVFFLAT.")

    # ---- Parse the kNN graph and build CSR adjacency ----
    neighbors, datasetsize = parse_neighbor_file(knn_graph)
    print(f"Parsed {datasetsize} queries. Building adjacency...")

    xadj, adjncy, adjwgt, vwgt = build_csr_from_neighbors(neighbors, datasetsize)


    # ---- Call kahip partitioner ----
    print("Start of KaHIP")

    edgecut, blocks = libraries.kahip.kaffpa(vwgt, xadj, adjwgt, adjncy, args.m, args.imbalance, True, args.seed, args.kahip_mode)
    
    # Debugging Reasons
    # print(f"Partitioned graph into {args.m} blocks with edgecut {edgecut}.")

    X = data_vectors
    y = np.array(blocks)

    # ---- Train the model based on dataset type ----
    if dataset_type and dataset_type.lower().startswith('sift'):
        model = sift_train(args, img_rows, img_cols, X, y)
    else:
        # Train the new CNN model, passing image dimensions
        model = mnist_train(args, img_rows, img_cols, X, y)

    libraries.os.makedirs(args.index, exist_ok=True)

    # Save the model and the inverted index file
    save_builds_output(model, args.index, X.copy(), y.copy(), img_rows, img_cols)

if __name__ == "__main__":
    main()