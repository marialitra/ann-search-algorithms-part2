# Approximate Nearest Neighbor Search using Neural Network Classifiers

## **Authors:**  

- Lytra Maria
- Mylonaki Danai

## Main idea

This project is a continuation of Part 1, where we implemented classical Approximate Nearest Neighbor (ANN) search algorithms.

In this part, we extend that work by introducing a learning-based approach:

- **Locality sensitive Hashing using Neural  Classifiers**
The way this algorithm works is by computing a graph with the k-nearest neighbors and then using a graph partitioner, such as KaHIP, to provide a locality-sensitive division of the data space. This information is then used to train a neural classifier to predict partition (block) assignments for vectors so that at query time the model maps a query to a small set of candidate partitions (top‑T); those candidates are then re-ranked with exact distances to produce the final K nearest neighbours.

For the implementation and detailed description of the algorithms from Part 1, see:
[ANN Search Algorithms Part 1](https://github.com/PoiLson/ann-search-algorithms-part1)

## Dataset Structure

For more details about the dataset structure, see [datasets.md](datasets.md).

## Project Structure

This project is formatted in the following way:  

- _src_: contains all .c files
- _Ivfflat_: Contains a modified implementation of excersice 1
- _knngraphs_: contains the outputs from the C code for ANN using ivfflat
- _Data_: contains folders with data as follows:
  - MNIST: mnist train and query sets can be saved here
  - SIFT: sift base and query sets can be saved here
- _true Neighbors_: contains cashed versions of bruteforce calls
- _Makefile_

### **Providing a more in-depth analysis of every .py  we encounter:**

**Files included in the src/ folder:**

The src/ directory contains the implementation (.py) files with full documentation and structured modular design.
A brief description of each file is provided below:

- **_bruteforce.py:_** runs brute force K-NN algorithm for the train and query sets given and saves rerults to an file npy, as well as a meta file with the parameters it run and the average computation time.
- **_datasetUtils.py:_** contains to function specific to the sift dataset for training using a memmap-back dataset.
- **_libraries.py:_** contains both common and custom libraries used by the program
- **_neuralNet.py:_** defines MLP and CNN clasifier and the training for both datasets
- **_nlshBuild.py:_** parses the arguments along with quality checks calls on functions to create the knn_graph,  build the csr , run kahip ,train the model and save the inverted index file.
- **_nlshCore.py:_** does the query search and the metcis calculations
- **_nlshSearch.py:_** parses the arguments and calls the function from nlshCore
- **_parseFiles.py:_** functions to read ivfflats output as well as the mnist and sift datasets
- **_runSearchExe.py:_** given the arguments it uses subprocess to run the C code for ivfflat
- **_utils.py:_** contains function to do parsing checks, generate file names, build the csr format, save the output from training, load these data (model.pth and inverted_file) and normalize data for both dsatasets

## Installation Instructions

To install all required Python dependencies, run the following command inside the
Python environment you will use for this project:

```bash
pip install -r requirements.txt
```

## Compilation & Run (Makefile)

The repository `Makefile` provides convenience targets that run the Python build pipeline or delegate to the native `Ivfflat/` build when present. Use the targets below rather than calling low-level commands directly unless you need custom arguments.

- `make mnist` — runs the Python build pipeline for MNIST (`src/nlsh_build.py`) using the preset variables in the `Makefile` (dataset path, number of blocks, model hyperparameters, etc.).
- `make sift` — same as above but for the SIFT dataset.
- `make mnistSearch` / `make siftSearch` — run the search/evaluation step (`src/nlsh_search.py`) with the `Makefile` presets.
- `make search` — attempts to build the native `Ivfflat` executable; the target delegates to `make -C Ivfflat` and will fail if the `Ivfflat/` directory is not present.
- `make clean` — placeholder target for cleaning generated files (customize as needed).

Examples (copy-paste):

```bash
# Run the MNIST build pipeline (uses python script with Makefile presets)
make mnist

# Run the SIFT build pipeline
make sift

# Run evaluation/search for MNIST
make mnistSearch

# Run evaluation/search for SIFT
make siftSearch

# Basic clean placeholder
make clean
```

If you prefer to call the Python script directly, an example invocation equivalent to `make mnist` (using the Makefile defaults) is:

```bash
python3 ./src/nlsh_build.py \
  -d ./Data/MNIST/train-images.idx3-ubyte \
  -i nlsh_index_mnist \
  --type MNIST \
  --knn 5 \
  -m 2000 \
  --layers 4 \
  --nodes 512 \
  --epochs 3 \
  --batch_size 256 \
  --lr 0.001 \
  --seed 42
```

- Makefile variables at the top of the file control defaults such as `NUM_BLOCKS`, `BATCH_SIZE`, `LEARNING_RATE`, etc. Edit those values to customize runs.

## Usage Instructions

### **Files**

- **_Dataset file →_** Includes the data of the wanted dataset (MNIST or SIFT)
- **_Query file →_** Includes the query data of the wanted dataset (MNIST or SIFT)
- **_Output file →_** File to extract the output with the execution's metrics
- **_index path →_** Folder where the indexing details are stored

### **Available Datasets**

- **_mnist →_** for the MNIST dataset
- **_sift →_** for the SIFT dataset

### **NLSH Build FLags**

- **_knn →_** number of nearest neighbors
- **_m →_** number of partitions for KaHIP
- **_imbalance →_** imbalamce of KaHIP
- **_kahip_mode →_** mode for KaHIP (0,1 or 2)
- **_layers →_** total number of layers for classifiers
- **_nodes →_** number of hidden nodes in classifiers
- **_epochs →_** number training loops
- **_batch_size →_** size of batches
- **_lr →_** learning rate for Adam Optimizer
- **_seed →_** seed for reproducibility

### **NLSH Search FLags**

- **_N →_** number of nearest neighbors
- **_R →_** range for Range Search
- **_T →_** probing size
- **_range →_** activates range search (True or False)

### **Default Execution**

The Makefile already contains pre-tuned (optimized) parameter
However, these presets are meant for our best configurations and not for a “default” run.

## Version Control and Collaboration

The development of this project was managed using the Git version control system.

All source files, and experimental scripts were tracked through a dedicated Git repository to ensure collaborative development, change tracking, and reproducibility of results. The repository was hosted on a private GitHub project for version tracking and collaboration.
