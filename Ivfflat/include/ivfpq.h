#ifndef IVFPQ_H
#define IVFPQ_H

// Product Quantization parameters for each subspace
typedef struct
{
    int M;                          // Number of subspaces (parts)
    int nbits;                      // Bits per subspace
    int s;                          // Number of centroids per subspace (s = 2^nbits)
    int d_sub;                      // Dimensionality of each subspace (d/M)
    float*** subspace_centroids;    // [M][s][d_sub] - centroids for each subspace
} PQConfig;

// Inverted list entry for IVFPQ (stores compressed PQ codes instead of full vectors)
typedef struct
{
    int point_id;        // Original dataset index
    uint16_t* pq_code;   // Compressed representation: M codes, each indexing a subspace centroid
} IVFPQEntry;

// Inverted list which maps clusters to their points
typedef struct 
{
    int cluster_id;        // Centroid's ID
    int count;             // How many points we have currently
    int capacity;          // How many points can be stored
    IVFPQEntry* entries;   // Array of compressed entries
} IVFPQList;

// IVFPQ Index structure
typedef struct
{
    int k;                    // Number of clusters
    int d;                    // Original dimensionality
    float** centroids;        // Array of centroids
    DataType data_type;       // Underlying dataset type
    IVFPQList* lists;         // One list per cluster
    PQConfig pq;              // Product quantization configuration
    Dataset* dataset;         // Pointer to original dataset for exact distance computation
} IVFPQIndex;

// Initialize IVFPQ index
IVFPQIndex* ivfpq_init(Dataset* dataset, int k_clusters, int M, int nbits);

// Performs lookup using IVFPQ
void ivfpq_index_lookup(const void* q_void, const struct SearchParams* params, 
                        int* approx_neighbors, double* approx_dists, int* approx_count, void* index_data);

// Performs range search using IVFPQ
void range_search_ivfpq(const void *q_void, const struct SearchParams *params, int **range_neighbors, int *range_count, void *index_data);

// Destroy IVFPQIndex structure
void ivfpq_destroy(IVFPQIndex* index);

#endif