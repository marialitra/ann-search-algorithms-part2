#define _XOPEN_SOURCE 500  // For strdup
#include "../include/main.h"

BruteForceCache* bruteforce_compute(const Dataset* dataset, const Dataset* query_set, int N)
{
    if (!dataset || !query_set || N <= 0)
    {
        fprintf(stderr, "Invalid parameters for brute-force computation\n");

        return NULL;
    }

    // printf("Computing brute-force nearest neighbors (N=%d) for %d queries...\n", N, query_set->size);

    BruteForceCache* cache = (BruteForceCache*)malloc(sizeof(BruteForceCache));
    if (!cache)
    {
        fprintf(stderr, "Failed to allocate cache structure\n");

        return NULL;
    }

    cache->n_queries = query_set->size;
    cache->N = N;

    // Allocate arrays for neighbors, distances, and query times
    cache->neighbors = (int**)malloc(cache->n_queries * sizeof(int*));
    cache->distances = (double**)malloc(cache->n_queries * sizeof(double*));
    cache->query_times = (double*)malloc(cache->n_queries * sizeof(double));
    
    if (!cache->neighbors || !cache->distances || !cache->query_times)
    {
        fprintf(stderr, "Failed to allocate cache arrays\n");
        bruteforce_cache_free(cache);

        return NULL;
    }

    for (int q = 0; q < cache->n_queries; q++)
    {
        cache->neighbors[q] = (int*)malloc(N * sizeof(int));
        cache->distances[q] = (double*)malloc(N * sizeof(double));
        
        if (!cache->neighbors[q] || !cache->distances[q])
        {
            fprintf(stderr, "Failed to allocate arrays for query %d\n", q);
            bruteforce_cache_free(cache);

            return NULL;
        }
    }

    // Compute brute-force for each query
    #pragma omp parallel for schedule(dynamic)
    for (int q_idx = 0; q_idx < query_set->size; q_idx++)
    {
        void* query = query_set->data[q_idx];
        
        // Start timing for this query
        double start_time = omp_get_wtime();
        
        // Allocate temporary arrays for all distances
        double* all_dists = (double*)malloc(dataset->size * sizeof(double));
        int* all_indices = (int*)malloc(dataset->size * sizeof(int));
        
        // Compute distance to every point in dataset
        for (int i = 0; i < dataset->size; i++)
        {
            all_dists[i] = euclidean_distance(query, dataset->data[i], dataset->dimension, query_set->data_type, dataset->data_type);
            all_indices[i] = i;
        }
        
        // Partial sort to find top N (using selection sort for simplicity)
        for (int i = 0; i < N && i < dataset->size; i++)
        {
            int min_idx = i;
            for (int j = i + 1; j < dataset->size; j++)
            {
                if (all_dists[j] < all_dists[min_idx])
                    min_idx = j;

            }
            
            // Swap distances
            double temp_dist = all_dists[i];
            all_dists[i] = all_dists[min_idx];
            all_dists[min_idx] = temp_dist;
            
            // Swap indices
            int temp_idx = all_indices[i];
            all_indices[i] = all_indices[min_idx];
            all_indices[min_idx] = temp_idx;
        }
        
        // Copy top N to cache
        for (int i = 0; i < N && i < dataset->size; i++)
        {
            cache->neighbors[q_idx][i] = all_indices[i];
            cache->distances[q_idx][i] = all_dists[i];
        }
        
        // Record time taken for this query
        double end_time = omp_get_wtime();
        cache->query_times[q_idx] = end_time - start_time;
        
        free(all_dists);
        free(all_indices);
        
        if ((q_idx + 1) % 10 == 0)
        {
            // printf("  Processed %d/%d queries...\r", q_idx + 1, query_set->size);
            fflush(stdout);
        }
    }
    
    // printf("\nBrute-force computation complete!                    \n");
    return cache;
}

bool bruteforce_cache_save(const BruteForceCache *cache, const char *cache_path)
{
    if (!cache || !cache_path)
        return false;

    // printf("Saving brute-force cache to: %s\n", cache_path);

    FILE *fp = fopen(cache_path, "wb");
    if (!fp)
    {
        fprintf(stderr, "Failed to open cache file for writing: %s\n", cache_path);

        return false;
    }

    // Write header: magic number, version, n_queries, N
    uint32_t magic = 0xBF424643;  // "BFCF" = BruteForce Cache File
    uint32_t version = 1;
    
    fwrite(&magic, sizeof(uint32_t), 1, fp);
    fwrite(&version, sizeof(uint32_t), 1, fp);
    fwrite(&cache->n_queries, sizeof(int), 1, fp);
    fwrite(&cache->N, sizeof(int), 1, fp);

    // Write neighbors, distances, and query times for each query
    for (int q = 0; q < cache->n_queries; q++)
    {
        fwrite(cache->neighbors[q], sizeof(int), cache->N, fp);
        fwrite(cache->distances[q], sizeof(double), cache->N, fp);
        fwrite(&cache->query_times[q], sizeof(double), 1, fp);
    }

    fclose(fp);
    // printf("Cache saved successfully!\n");
    return true;
}

BruteForceCache* bruteforce_cache_load(const char* cache_path, int expected_queries, int expected_N)
{
    if (!cache_path)
        return NULL;

    FILE* fp = fopen(cache_path, "rb");
    if (!fp)
    {
        // File doesn't exist - not an error, just return NULL
        return NULL;
    }

    // printf("Loading brute-force cache from: %s\n", cache_path);

    // Read and validate header
    uint32_t magic, version;
    int n_queries, N;
    
    if (fread(&magic, sizeof(uint32_t), 1, fp) != 1 || magic != 0xBF424643)
    {
        fprintf(stderr, "Invalid cache file: bad magic number\n");
        fclose(fp);

        return NULL;
    }
    
    if (fread(&version, sizeof(uint32_t), 1, fp) != 1 || version != 1)
    {
        fprintf(stderr, "Invalid cache file: unsupported version\n");
        fclose(fp);

        return NULL;
    }
    
    if (fread(&n_queries, sizeof(int), 1, fp) != 1 || fread(&N, sizeof(int), 1, fp) != 1)
    {
        fprintf(stderr, "Error reading cache header from %s\n", cache_path);
        fclose(fp);

        return NULL;
    }

    // Validate parameters match
    if (n_queries != expected_queries || N != expected_N)
    {
        fprintf(stderr, "Cache mismatch: expected (%d queries, N=%d) but got (%d queries, N=%d)\n", expected_queries, expected_N, n_queries, N);
        fclose(fp);

        return NULL;
    }

    // Allocate cache structure
    BruteForceCache* cache = (BruteForceCache*)malloc(sizeof(BruteForceCache));
    if (!cache)
    {
        fclose(fp);

        return NULL;
    }

    cache->n_queries = n_queries;
    cache->N = N;
    cache->neighbors = (int**)malloc(n_queries * sizeof(int*));
    cache->distances = (double**)malloc(n_queries * sizeof(double*));
    cache->query_times = (double*)malloc(n_queries * sizeof(double));

    if (!cache->neighbors || !cache->distances || !cache->query_times)
    {
        bruteforce_cache_free(cache);
        fclose(fp);

        return NULL;
    }

    // Read data for each query
    for (int q = 0; q < n_queries; q++)
    {
        cache->neighbors[q] = (int*)malloc(N * sizeof(int));
        cache->distances[q] = (double*)malloc(N * sizeof(double));
        
        if (!cache->neighbors[q] || !cache->distances[q])
        {
            bruteforce_cache_free(cache);
            fclose(fp);

            return NULL;
        }
        
        if (fread(cache->neighbors[q], sizeof(int), N, fp) != (size_t)N ||
            fread(cache->distances[q], sizeof(double), N, fp) != (size_t)N ||
            fread(&cache->query_times[q], sizeof(double), 1, fp) != 1)
        {
            fprintf(stderr, "Error reading cache data for query %d\n", q);
            bruteforce_cache_free(cache);
            fclose(fp);

            return NULL;
        }
    }

    fclose(fp);
    // printf("Cache loaded successfully!\n");

    return cache;
}

void bruteforce_cache_free(BruteForceCache *cache)
{
    if (!cache)
        return;

    if (cache->neighbors)
    {
        for (int q = 0; q < cache->n_queries; q++)
        {
            if (cache->neighbors[q])
                free(cache->neighbors[q]);
        }

        free(cache->neighbors);
    }

    if (cache->distances)
    {
        for (int q = 0; q < cache->n_queries; q++)
        {
            if (cache->distances[q])
                free(cache->distances[q]);
        }

        free(cache->distances);
    }

    if (cache->query_times)
        free(cache->query_times);

    free(cache);
    
    return;
}

char* bruteforce_cache_get_path(const char* dataset_path, const char* query_path, int N)
{
    if (!dataset_path || !query_path)
        return NULL;

    // Extract filenames without paths
    char* dataset_copy = strdup(dataset_path);
    char* query_copy = strdup(query_path);
    char* dataset_name = basename(dataset_copy);
    char* query_name = basename(query_copy);

    // Remove extensions
    char* dot = strrchr(dataset_name, '.');

    if (dot) 
        *dot = '\0';
    
    dot = strrchr(query_name, '.');
    
    if (dot)
        *dot = '\0';

    // Build cache filename
    char* cache_path = (char*)malloc(512);
    snprintf(cache_path, 512, "Data/.cache/%s_%s_N%d_bruteforce.cache", dataset_name, query_name, N);

    free(dataset_copy);
    free(query_copy);

    return cache_path;
}