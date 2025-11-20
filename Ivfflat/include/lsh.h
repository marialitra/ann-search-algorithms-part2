#ifndef LSH_H
#define LSH_H

// Forward declarations
struct LSH;
struct SearchParams;
struct Dataset;

// Data structure for the hash function h(p)
typedef struct 
{
    float* v;       // Projection vector
    float t;        // Random offset
}LSH_hash_function;

// Define function pointer types
typedef int (*hash_func)(const void* p, const struct LSH* lsh, int table_index, uint64_t* ID);
typedef double (*metric_func)(const void* a, const void* b, const int dimension, DataType data_typea , DataType data_typeb);

// Data Structure for the full LSH
typedef struct LSH
{
    int d;                              // Dimension of the input points
    int L;                              // Number of hash tables
    int k;                              // Number of hash functions per table
    float w;                            // Window size
    int dataset_size;                   // Total number of points in the dataset (for visited array)
    int table_size;                     // Size of each hash table
    uint64_t num_of_buckets;            // Large modulus M for ID composition
    DataType data_type;                 // Type of data (DATA_TYPE_UINT8 or DATA_TYPE_FLOAT)
    metric_func distance;               // Distance function

    // Per-table hash parameters: for each table (L), we have k hash functions (v, t)
    LSH_hash_function** hash_params;    // 2D Array of hash parameters
    int** linear_combinations;          // Array of r[i][j], i in[L], j in[k] for g(p)
    
    HashTable* hash_tables;             // Array of hash tables
} LSH;

#define R_RANGE (1U << 29)  // values in [1, 2^29] to bound linear coefficients r_i

// ------------------- Helper functions for hashing -----------------------------

// Defines the LSH hash function
// g(p) = ID(p) mod table_size
// Where ID(p) = sum(h_i(p) * r_i) mod M
// r_i are random integers saved in the LSH struct
// M is the number of buckets also saved in the LSH struct
// h_i are the hash functions saved in the LSH struct
// g() needs to be a function stored in the LSH struct so it needs to return hash_func type
static inline int hash_func_impl_lsh(const void* p, const LSH* lsh, int table_index, uint64_t* outID)
{
    // The prime modulus for universal hashing
    const uint64_t M = lsh->num_of_buckets;
    uint64_t id_val = 0;
    const LSH_hash_function* table_hash_params = lsh->hash_params[table_index];

    for (int i = 0; i < lsh->k; i++)
    {
        // Compute LSH projection: h_i(v) = floor((a·v + b)/w)
        double func = 0.0;
        func = dot_product(table_hash_params[i].v, (const float *)p, lsh->d, lsh->data_type);

        double val = (func + (double)table_hash_params[i].t) / (double)lsh->w;
        
        // Universal hashing linear coefficients
        int64_t h_i = (int64_t)floor(val);
        int64_t r_i = lsh->linear_combinations[table_index][i];

        // Compute modular products safely in 128-bit space
        __int128 prod = (__int128)r_i * (__int128)h_i;

        // Reduce modulo prime
        uint64_t mod = (uint64_t)((prod % (int64_t)M + (int64_t)M) % (int64_t)M);
        id_val = (id_val + mod) % M;
    }

    // Output fingerprint
    if (outID)
        *outID = id_val;

    // Final hash table index
    int bucket_idx = (int)(id_val % (uint64_t)lsh->table_size);
    
    return bucket_idx;
}

// Wrapper function for hash function
static inline int hash_function_lsh(HashTable ht, void* data, uint64_t* ID)
{
    LSH* lsh_ctx = (LSH*)hash_table_get_algorithm_context(ht);
    if (!lsh_ctx || lsh_ctx->num_of_buckets == 0)
    {
        if (ID)
            *ID = 0;
        return 0;
    }
    int t_idx = hash_table_get_index(ht);

    return hash_func_impl_lsh(data, lsh_ctx, t_idx, ID);
}

// ------------------------- LSH main functions ---------------------------------

// Initializes the LSH structure, allocates memory, sets parameters, and stores data points
LSH* lsh_init(const struct SearchParams* params, const struct Dataset* dataset);

// Performs approximate nearest neighbor search using the LSH index
void lsh_index_lookup(const void* q, const struct SearchParams* params, int* approx_neighbors,
                        double* approx_dists, int* approx_count, void* index_data);

// Performs range search using the LSH index
void range_search_lsh(const void *q, const struct SearchParams *params,
                        int **range_neighbors, int *range_count, void *index_data);

// Frees all allocated memory associated with the LSH structure
void lsh_destroy(struct LSH* lsh);

#endif