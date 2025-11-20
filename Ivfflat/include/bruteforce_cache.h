#ifndef BRUTEFORCE_CACHE_H
#define BRUTEFORCE_CACHE_H

// Structure to hold cached brute-force results
typedef struct
{
    int n_queries;        // Number of queries
    int N;                // Number of neighbors per query
    int** neighbors;      // neighbors[q][i] = index of i-th nearest neighbor for query q
    double** distances;   // distances[q][i] = distance to i-th nearest neighbor for query q
    double* query_times;  // query_times[q] = time taken for brute-force search for query q
} BruteForceCache;

// Compute brute-force nearest neighbors for all queries
// Returns cache structure with results
BruteForceCache* bruteforce_compute(const Dataset* dataset, const Dataset* query_set, int N);

// Save cache to disk
bool bruteforce_cache_save(const BruteForceCache* cache, const char* cache_path);

// Load cache from disk
// Returns NULL if file doesn't exist or is invalid
BruteForceCache* bruteforce_cache_load(const char* cache_path, int expected_queries, int expected_N);

// Free cache memory
void bruteforce_cache_free(BruteForceCache* cache);

// Generate cache filename based on dataset and query paths
// Format: <dataset_name>_<query_name>_N<N>_bruteforce.cache
char* bruteforce_cache_get_path(const char* dataset_path, const char* query_path, int N);

#endif