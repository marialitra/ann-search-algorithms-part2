#ifndef HYPERCUBE_H
#define HYPERCUBE_H

// Forward declaration
struct Hypercube;

// Define function pointer
typedef uint64_t (*bin_hash)(const void* p, const struct Hypercube* hyper, uint64_t* ID);

//structure for hypercube hash function parameters
typedef struct Hypercube_hash_function
{
    float* v;   // The projection vector
    float t;    // Random offset
} Hypercube_hash_function;

// Hypercube structure for holding all relevant parameters and data
typedef struct Hypercube
{
    int d;                                  // Dimension of the input points
    int kproj;                              // Number of projections
    int M;                                  // Max number of points to check
    float w;                                // Window size
    int probes;                             // Number of probes
    int dataset_size;                       // Size of dataset
    DataType data_type;                     // Type of data (DATA_TYPE_UINT8 or DATA_TYPE_FLOAT)
    metric_func distance;                   // Distance metric function
    Hypercube_hash_function* hash_params;   // Array of hash parameters
    bin_hash binary_hash_function;          // Single hash function that computes all k bits
    float* thresholds;                      // Average threshold for each projection (for locality-preserving f)
    
    HashTable hash_table;                   // Hash table
} Hypercube;

//-----------------------Helper functions for hashing-----------------------------

// Deterministic threshold-based mapping to preserve locality
static inline bool f(int h_ip, float threshold)
{
    return (h_ip >= (int)threshold) ? 1 : 0;
}

// Compute the k-bit binary ID for a point in the hypercube
static inline uint64_t hash_func_impl_hyper(const void* p, const Hypercube* hyper, uint64_t* ID)
{
    uint64_t id = 0ULL;
    for (int i = 0; i < hyper->kproj; i++)
    {
        float func;
        func = dot_product(hyper->hash_params[i].v, p, hyper->d, hyper->data_type);

        // Compute bucket index using floorf for correctness
        float val = (func + hyper->hash_params[i].t) / hyper->w;
        int h_i = (int)floorf(val);

        // Apply threshold-based mapping to preserve locality: h_i >= threshold -> 1, else -> 0
        bool bit = f(h_i, hyper->thresholds[i]);

        // Shift left and add the new bit
        id = (id << 1) | (uint64_t)bit;
    }

    if (ID) *ID = id;
    
    return id;  // Return full uint64_t
}


// Modifies the hash function to be used in the hash table
static inline int hash_function_hyper(HashTable ht, void* data, uint64_t* ID)
{
    Hypercube* hyper_ctx = (Hypercube*)hash_table_get_algorithm_context(ht);
    if (!hyper_ctx) 
    { 
        if (ID) *ID = 0ULL; 
        return 0; 
    }

    return hyper_ctx->binary_hash_function(data, hyper_ctx, ID);
}

//--------------------------Hypercube main functions-----------------------------

// Initializes the hypercube structure, allocates memory, sets parameters, and stores data points
Hypercube* hyper_init(const struct SearchParams* params, const struct Dataset* dataset);

// Performs approximate nearest neighbor search using the hypercube index
void hyper_index_lookup(const void* q, const struct SearchParams* params, int* approx_neighbors,
                        double* approx_dists, int* approx_count, void* index_data);

// Performs range search using the hypercube index
void range_search_hyper(const void* q, const struct SearchParams* params, int** range_neighbors,
                        int* range_count, void* index_data);

// Frees all allocated memory associated with the hypercube structure
void hyper_destroy(struct Hypercube* hyper);

#endif