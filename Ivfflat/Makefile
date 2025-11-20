# ==============================================================
#  Makefile for Project (C)
# ==============================================================

# Compiler
CC = gcc

CFLAGS = -std=c11 -g -Iinclude -fopenmp -O3 \
		-funroll-loops -ffast-math
LDFLAGS = -lm

# Executable name
TARGET = search

# Directories (relative to Ivfflat/ so `make -C Ivfflat` works)
SRCDIR = src
OBJDIR = objectFiles
INCDIR = include

# Source files
# Source files (found relative to Ivfflat/)
SRCS = $(wildcard $(SRCDIR)/*.c)

# Object files
OBJS = $(SRCS:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# Pattern rule to compile any .c file in src/ to .o file in objectFiles/
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# =====================================================================
# User Input Parameters

# CHOOSE WHAT ALGORITHM YOU WANT TO USE
# AVAILABLE OPTIONS (algorithm -> pseudoname to use below):
# Locality Sensitive Hashing 			-> lsh
# Binary Hypercube 						-> hypercube
# InVerted File index 					-> ivfflat
# IVF with Product Quantization			-> ivfpq

# =====================================================================

ALGO = ivfflat
OUTPUT_FILE = mnistfullivfflat.txt

# ==============================================================
# Algorithm-specific parameters
# ==============================================================

# MNIST-optimized parameters
# ==============================================================

ifeq ($(ALGO), lsh)
    ALGO_PARAMS_MNIST = -k 4 -L 10 -w 120 -o $(OUTPUT_FILE) -N 1 -R 4 -type mnist -lsh -range false
endif

ifeq ($(ALGO), hypercube)
    ALGO_PARAMS_MNIST = -kproj 12 -w 40 -M 8500 -probes 850 -o $(OUTPUT_FILE) -N 1 -R 4 -type mnist -range false -hypercube
endif

ifeq ($(ALGO), ivfflat)
	ALGO_PARAMS_MNIST = -kclusters 1024 -nprobe 5 -o $(OUTPUT_FILE) -N 100 -R 700 -type mnist -range false -ivfflat -seed 42
endif

ifeq ($(ALGO), ivfpq)
	ALGO_PARAMS_MNIST = -kclusters 60 -nprobe 40 -M 98 -o $(OUTPUT_FILE) -N 10 -R 50000 -type mnist -nbits 8 -range false -ivfpq -seed 10
endif

# SIFT-optimized parameters
# ==============================================================

ifeq ($(ALGO), lsh)
    ALGO_PARAMS_SIFT = -k 6 -L 15 -w 60 -o $(OUTPUT_FILE) -N 1 -R 50000 -type sift -lsh -range false
endif

ifeq ($(ALGO), hypercube)
    ALGO_PARAMS_SIFT = -kproj 18 -w 50 -M 100000 -probes 1500 -o $(OUTPUT_FILE) -N 1 -R 50000 -type sift -range false -hypercube
endif

ifeq ($(ALGO), ivfflat)
    ALGO_PARAMS_SIFT = -kclusters 316 -nprobe 30 -o $(OUTPUT_FILE) -N 100 -R 2 -type sift -range false -ivfflat -seed 42
endif

ifeq ($(ALGO), ivfpq)
    ALGO_PARAMS_SIFT = -kclusters 60 -nprobe 5 -M 128 -o $(OUTPUT_FILE) -N 10 -R 50000 -type sift -nbits 8 -range false -ivfpq -seed 10
endif

# ==============================================================
# Build executable from object files
# ==============================================================

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# ==============================================================
# Compile each .c into .o
# ==============================================================

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# ==============================================================
# Create object directory if missing
# ==============================================================

$(OBJDIR):
	@mkdir -p $(OBJDIR)

# ==============================================================
# Cleaning
# ==============================================================

clean:
	@echo "Cleaning up..."
	rm -f $(OBJDIR)/*.o $(TARGET)

# ==============================================================
# Dataset-specific preset targets
# ==============================================================

mnist: $(OBJDIR) $(TARGET)
	@echo "Running $(ALGO) on MNIST dataset..."
	OMP_NUM_THREADS=8,4 \
	OMP_NESTED=TRUE \
	OMP_MAX_ACTIVE_LEVELS=2 \
	./$(TARGET) \
		-d Data/MNIST/train-images.idx3-ubyte \
		-q Data/MNIST/train-images.idx3-ubyte \
		$(ALGO_PARAMS_MNIST)
	@$(MAKE) clean

sift: $(OBJDIR) $(TARGET)
	@echo "Running $(ALGO) on SIFT dataset..."
	OMP_NUM_THREADS=8,4 \
	OMP_NESTED=TRUE \
	OMP_MAX_ACTIVE_LEVELS=2 \
	./$(TARGET) \
		-d Data/SIFT/sift_base.fvecs \
		-q Data/SIFT/sift_base.fvecs \
		$(ALGO_PARAMS_SIFT)
