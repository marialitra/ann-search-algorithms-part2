import libraries
from libraries import load_data, neural_lsh, validate_args

def main():
    p = libraries.argparse.ArgumentParser(description="Neural LSH search phase.")
    p.add_argument("-d", "--dataset", required=True, type=str, help="Dataset file")
    p.add_argument("-q", "--query", required=True, type=str, help="Query file")
    p.add_argument("-i", "--index", required=True, type=str, help="Path to built index directory")
    p.add_argument("-o", "--output", required=True, type=str, help="Output results file")
    p.add_argument("-type", required=True, type=str, help="Type of the given dataset (MNIST or SIFT1M)")
    p.add_argument("-N", type=int, default=1, help="Number of nearest neighbors to report")
    p.add_argument("-R", type=float, default=-2, help="Distance for range search if enable")
    p.add_argument("-T", type=int, default=5, help="Number of bins to probe (multi-probe)")
    p.add_argument("-range", type=str, default="true", help="Range search flag")
    args = p.parse_args()
    validate_args(args)
    
    is_mnist = args.type and args.type.lower().startswith("mnist")
    is_sift = args.type and args.type.lower().startswith("sift")

    # --- Setup default R ---
    if args.R < 0:
        # Meaning that no given value has been given, or a wrong one detected,
        # We are going to use the default values for each dataset type
        if is_sift:
            args.R = 2800
        elif is_mnist:
            args.R = 2000
        else:
            print("Not acceptable dataset type")
            exit()

    model, X, Q, X_flat_raw, Q_flat_raw, inverted = load_data(args)
    neural_lsh(args, model, inverted, X, Q, X_flat_raw, Q_flat_raw)

if __name__ == "__main__":
    main()