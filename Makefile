# Makefile for graph partitioning and classification script

# --- Variables ---
PYTHON := python3
SCRIPT := ./src/nlsh_build.py

# Example arguments for running the script.
# NOTE: You need to replace these paths and values with actual files and desired settings.

# INPUT_FILE_SIFT := ./knngraphs/siftfullivfflat_baseneighboring_graph.txt
# INPUT_FILE_SIFT := ./src/sift5knn.txt

DATASET_FILE_SIFT := ./Data/SIFT/sift_base.fvecs
# DATASET_FILE_SIFT := ./Data/SIFT/sift_base_100k.fvecs
OUTPUT_FILE_SIFT:= nlsh_index
DATASET_TYPE_SIFT := sift
NUM_BLOCKS_SIFT := 200
NUM_LAYERS_SIFT := 3
NUM_NEURONS_SIFT := 256
LEARNING_RATE_SIFT := 0.001
EPOCHS_SIFT := 20
BATCH_SIZE_SIFT := 1024
KNN_NEIGHBORS := 5

# INPUT_FILE := ./knngraphs/mnistfullivfflat5testset.txt
DATASET_FILE := ./Data/MNIST/train-images.idx3-ubyte
OUTPUT_FILE:= nlsh_index
DATASET_TYPE := MNIST
NUM_BLOCKS := 2000
NUM_LAYERS := 5
NUM_NEURONS := 512
LEARNING_RATE := 0.001
EPOCHS := 20
BATCH_SIZE := 4098

# --- Targets ---

.PHONY: all run clean help

all: run

## run: Executes the script with example arguments.
sift:
	@echo "--- Running the script: $(SCRIPT) ---"
	$(PYTHON) $(SCRIPT) \
		-d $(DATASET_FILE_SIFT) \
		-i $(OUTPUT_FILE_SIFT) \
		--type $(DATASET_TYPE_SIFT) \
		--knn $(KNN_NEIGHBORS) \
		-m $(NUM_BLOCKS_SIFT) \
		--layers $(NUM_LAYERS_SIFT) \
		--nodes $(NUM_NEURONS_SIFT) \
		--lr $(LEARNING_RATE_SIFT) \
		--epochs $(EPOCHS_SIFT) \
		--batch_size $(BATCH_SIZE_SIFT)
	@echo "-----------------------------------"


mnist:
	@echo "--- Running the script: $(SCRIPT) ---"
	$(PYTHON) $(SCRIPT) \
		-d $(DATASET_FILE) \
		-i $(OUTPUT_FILE) \
		--type $(DATASET_TYPE) \
		--knn $(KNN_NEIGHBORS) \
		-m $(NUM_BLOCKS) \
		--layers $(NUM_LAYERS) \
		--nodes $(NUM_NEURONS) \
		--lr $(LEARNING_RATE) \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE)
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
  -i nlsh_index \
  -o output.txt \
  -type sift \
  -N 4 -T 20 -range FALSE

mnistSearch:
	python3 ./src/nlsh_search.py \
  -d ./Data/MNIST/train-images.idx3-ubyte \
  -q ./Data/MNIST/t10k-images.idx3-ubyte \
  -i nlsh_index \
  -o output.txt \
  -N 4 -T 200 -range FALSE