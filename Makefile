# Makefile for graph partitioning and classification script

# --- Variables ---
PYTHON := python3
SCRIPT := ./src/nlsh_build.py

# --- SIFT Variables ---
INPUT_FILE_SIFT := ./Data/SIFT/sift_base.fvecs
INDEX_PATH_SIFT:= nlsh_index_sift
DATASET_TYPE_SIFT := sift
KNN_NEIGHBORS_SIFT := 5
NUM_BLOCKS_SIFT := 2000
IMBALANCE_SIFT := 0.1
KAHIP_MODE_SIFT := 0 # 1 equals ECO mode, 0 equals fast mode
NUM_LAYERS_SIFT := 3
NUM_NEURONS_SIFT := 128
EPOCHS_SIFT := 3
BATCH_SIZE_SIFT := 2048
LEARNING_RATE_SIFT := 0.001
SEED_SIFT := 42

# --- MNIST Variables ---
INPUT_FILE := ./Data/MNIST/train-images.idx3-ubyte
INDEX_PATH:= nlsh_index_mnist
DATASET_TYPE := MNIST
KNN_NEIGHBORS := 30
NUM_BLOCKS := 10000
IMBALANCE := 0.1
KAHIP_MODE := 0
NUM_LAYERS := 3
NUM_NEURONS := 512
EPOCHS := 3
BATCH_SIZE := 2048
LEARNING_RATE := 0.001
SEED := 42

# --- Targets ---

.PHONY: all run clean help

all: run

## run: Executes the script with example arguments.
sift:
	@echo "--- Running the script: $(SCRIPT) ---"
	$(PYTHON) $(SCRIPT) \
		-d $(INPUT_FILE_SIFT) \
		-i $(INDEX_PATH_SIFT) \
		--type $(DATASET_TYPE_SIFT) \
		--knn $(KNN_NEIGHBORS_SIFT) \
		-m $(NUM_BLOCKS_SIFT) \
		--imbalance $(IMBALANCE_SIFT) \
		--kahip_mode $(KAHIP_MODE_SIFT) \
		--layers $(NUM_LAYERS_SIFT) \
		--nodes $(NUM_NEURONS_SIFT) \
		--epochs $(EPOCHS_SIFT) \
		--batch_size $(BATCH_SIZE_SIFT) \
		--lr $(LEARNING_RATE_SIFT) \
		--seed $(SEED_SIFT) 
	@echo "-----------------------------------"


mnist:
	@echo "--- Running the script: $(SCRIPT) ---"
	$(PYTHON) $(SCRIPT) \
		-d $(INPUT_FILE) \
		-i $(INDEX_PATH) \
		--type $(DATASET_TYPE) \
		--knn $(KNN_NEIGHBORS) \
		-m $(NUM_BLOCKS) \
		--imbalance $(IMBALANCE) \
		--kahip_mode $(KAHIP_MODE) \
		--layers $(NUM_LAYERS) \
		--nodes $(NUM_NEURONS) \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--lr $(LEARNING_RATE) \
		--seed $(SEED) 
	@echo "-----------------------------------"

## clean: Placeholder for cleaning up generated files.
clean:
	@echo "Cleaning up generated files (if any)..."
	# Add commands here to remove temporary or output files, e.g.:
	# rm -f *.log *.out
	@echo "Done."

## help: Display this help message.
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'


siftSearch:
	python3 ./src/nlsh_search.py \
  -d ./Data/SIFT/sift_base.fvecs \
  -q ./Data/SIFT/sift_query.fvecs \
  -i nlsh_index_sift \
  -o output.txt \
  -type sift \
  -N 5 -R 98.0 -T 90 -range false

mnistSearch:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
	python3 ./src/nlsh_search.py \
  -d ./Data/MNIST/train-images.idx3-ubyte \
  -q ./Data/MNIST/t10k-images.idx3-ubyte \
  -i nlsh_index_mnist \
  -o output.txt \
  -type MNIST \
  -N 4 -R 300.0 -T 100 -range false


# ==============================================================
# Build executable: delegate to Ivfflat/Makefile (if present)
# ==============================================================

# building the native 'search' binary using the Ivfflat subdirectory
search:
	@echo "--- Delegating build to Ivfflat/Makefile (make -C Ivfflat) ---"
	@if [ -d Ivfflat ]; then \
		$(MAKE) -C Ivfflat; \
	else \
		echo "Ivfflat/ directory not found; cannot build 'search'"; exit 1; \
	fi