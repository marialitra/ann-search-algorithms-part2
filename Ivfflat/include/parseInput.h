#ifndef PARSEINPUT_H
#define PARSEINPUT_H

// Enumeration for algorithm types: LSH, Hypercube, IVFFlat, IVFPQ and none
typedef enum
{
    ALG_NONE,
    ALG_LSH,
    ALG_HYPERCUBE,
    ALG_IVFFLAT,
    ALG_IVFPQ
} AlgorithmType;

// Enumeration for dataset types: MNIST, SIFT and none
typedef enum
{
    DATA_NONE,
    DATA_MNIST,
    DATA_SIFT
} DatasetType;

// Structure to hold all search parameters
typedef struct SearchParams
{
    // Common parameters
    char dataset_path[256];         // Path to dataset
    char query_path[256];           // Path to query set
    char output_path[256];          // Path to output file
    DatasetType dataset_type;       // Type of the dataset's data
    AlgorithmType algorithm;        // Type of current algorithm (lsh, hypersube, ivfflat, ivfpq)
    bool range_search;              // Variable to know if range search is enabled
    int N;                          // Number of nearest neighbors (topN)
    double R;                       // Range search radius

    // LSH params
    int k;                          // Number of hash functions
    int L;                          // Number of hash tables
    double w;                       // Window width

    // Hypercube params
    int kproj;                      // Number of projection
    int M;                          // Number of elements to be examined or Number of subvectors for IVFPQ
    int probes;                     // Number of vertices to be examined

    // IVFFlat / IVFPQ params
    int kclusters;                  // Number of centroids
    int nprobe;                     // Number of neighboring centroids to be examined
    unsigned int seed;              // Seed for reproducing results

    // IVFPQ extra params
    int nbits;                      // Size of centroids (s = 2^nbits)
} SearchParams;

// Function to parse command-line arguments and populate SearchParams structure
int parse_arguments(int argc, char** argv, SearchParams* params);

// Function to print usage instructions
void print_usage(AlgorithmType type);

#endif