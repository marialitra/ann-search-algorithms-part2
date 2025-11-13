# Makefile for graph partitioning and classification script

# --- Variables ---
PYTHON := python3
SCRIPT := builder.py

# Example arguments for running the script.
# NOTE: You need to replace these paths and values with actual files and desired settings.
INPUT_FILE := ./knngraphs/siftfullivfflat_baseneighboring_graph.txt
DATASET_FILE := ./Data/SIFT/sift_base.fvecs
OUTPUT_FILE:= nlsh_index
DATASET_TYPE := sift
NUM_BLOCKS := 200
NUM_LAYERS := 3
NUM_NEURONS := 256
LEARNING_RATE := 0.001
EPOCHS := 20
BATCH_SIZE := 1024


# --- Targets ---

.PHONY: all run clean help

all: run

## run: Executes the script with example arguments.
run:
	@echo "--- Running the script: $(SCRIPT) ---"
	$(PYTHON) $(SCRIPT) \
		$(INPUT_FILE) \
		-d $(DATASET_FILE) \
		-i $(OUTPUT_FILE) \
		--type $(DATASET_TYPE) \
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


search:
	python3 nlsh_search.py \
  -d ./Data/SIFT/sift_base.fvecs \
  -q ./Data/SIFT/sift_query.fvecs \
  -i nlsh_index \
  -o output.txt \
  -type sift \
  -N 4 -T 20 -range FALSE
