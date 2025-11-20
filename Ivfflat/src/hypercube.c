#include "../include/main.h"

Hypercube* hyper_init(const struct SearchParams* params, const struct Dataset* dataset)
{
    // Allocate memory for Hypercube structure
    // And set parameters
    Hypercube* hyper = (Hypercube*)malloc(sizeof(Hypercube));
    if (!hyper)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    hyper->d = dataset->dimension;
    hyper->kproj = params->kproj;
    hyper->w = params->w;
    hyper->M = params->M;
    hyper->probes = params->probes;
    hyper->dataset_size = dataset->size;
    
    // Store data type and select distance function
    hyper->data_type = dataset->data_type;
    hyper->distance = euclidean_distance; 

    // Initialize hash parameters
    hyper->hash_params = (Hypercube_hash_function*)malloc(hyper->kproj * sizeof(Hypercube_hash_function));
    if (!hyper->hash_params)
    {
        fprintf(stderr, "Failed to allocate hash aprameters for hypercube\n");
        hyper_destroy(hyper);

        exit(EXIT_FAILURE);
    }

    // Generate random vectors and offsets for each hash function
    for (int i = 0; i < hyper->kproj; i++)
    {
        hyper->hash_params[i].v = (float*)malloc(hyper->d * sizeof(float));
        if (!hyper->hash_params)
        {
            fprintf(stderr, "Failed to allocate hash parameters vector for hypercube\n");

            exit(EXIT_FAILURE);
        }

        generate_random_vector(hyper->hash_params[i].v, hyper->d);
        normalize_vector(hyper->hash_params[i].v, hyper->d);

        if(!hyper->hash_params[i].v)
        {
            fprintf(stderr, "Failed to allocate vector for hypercube\n");
            hyper_destroy(hyper);

            exit(EXIT_FAILURE);
        }

        int tmp = 0;
        hyper->hash_params[i].t = uniform_distribution(&tmp, &(hyper->w));
    }

    // Set the single binary hash function for hypercube
    // (computes all k bits to form the bucket index)
    hyper->binary_hash_function = hash_func_impl_hyper;

    // Compute average thresholds for each projection to preserve locality
    hyper->thresholds = (float*)malloc(hyper->kproj * sizeof(float));
    if (!hyper->thresholds)
    {
        fprintf(stderr, "Failed to allocate threshold for hypercube\n");
        hyper_destroy(hyper);
        
        exit(EXIT_FAILURE);
    }

    for (int proj = 0; proj < hyper->kproj; proj++)
    {
        // Compute average h_i value for all points for this projection
        double sum = 0.0;

        for (int i = 0; i < dataset->size; i++)
        {
            float func = dot_product(hyper->hash_params[proj].v, dataset->data[i], hyper->d, dataset->data_type);
            float val = (func + hyper->hash_params[proj].t) / hyper->w;
            float h_i = floorf(val);
            sum += h_i;
        }

        // Compute average
        hyper->thresholds[proj] = (float)(sum / dataset->size);
    }


    // Create hash table with algorithm context and index 0
    hyper->hash_table = hash_table_create(1 << hyper->kproj, sizeof(int), NULL, NULL, hash_function_hyper, hyper, 0, &(dataset->dimension));
    if (!hyper->hash_table)
    {
        fprintf(stderr, "Failed to allocate hashtable for hypercube\n");
        hyper_destroy(hyper);

        exit(EXIT_FAILURE);
    }

    // Insert dataset points into hash table
    for (int i = 0; i < dataset->size; i++)
    {
        if (!dataset->data || !dataset->data[i])
        {
            printf("Warning: dataset[%d] is NULL\n", i);

            continue;
        }

        // Insert the valid data point into the hash table
        hash_table_insert(hyper->hash_table, &i, dataset->data[i]);
    }

    return hyper;
}

void hyper_index_lookup(const void* q, const struct SearchParams* params, int* approx_neighbors, double* approx_dists, int* approx_count, void* index_data)
{
    // Retrieve hypercube structure from context
    struct Hypercube* hyper = (struct Hypercube*)index_data;

    // Create min-heap for top-N neighbors
    MinHeap *topN = heap_create(params->N);
    if (!topN)
    {
        fprintf(stderr, "Failed to allocate min-heap.\n");
        exit(EXIT_FAILURE);
    }

    // Allocate visited array for O(1) duplicate detection (instead of O(N) linear scan)
    // This dramatically speeds up queries when examining many candidates
    bool* visited = (bool*)calloc(hyper->dataset_size, sizeof(bool));
    if (!visited)
    {
        fprintf(stderr, "Failed to allocate visited array\n");
        heap_destroy(topN);

        exit(EXIT_FAILURE);
    }

    // Compute the bucket index for the query point
    uint64_t q_id;
    uint64_t bucket_idx = hyper->binary_hash_function(q, hyper, &q_id);

    // Access the bucket corresponding to the computed index
    int bucket_count = 0;
    const HTEntry* bucket = hash_table_get_bucket_entries(hyper->hash_table, bucket_idx, &bucket_count);
    
    uint64_t* neighbors = (uint64_t*)malloc(hyper->probes * sizeof(uint64_t));
    int neighbor_count = hyper->probes;

    // Get neighboring bucket indices based on Hamming distance
    // Only the required number of probes
    int checked = 0; // Number of distinct points examined
    int reached_m = 0;
    get_hamming_neighbors(bucket_idx, hyper->probes, hyper->kproj, neighbors);
    for (int n = 0; n < neighbor_count; n++)
    {
        uint64_t neighbor_idx = neighbors[n];
        int neighbor_count_entries = 0;
        const HTEntry* neighbor_bucket = hash_table_get_bucket_entries(hyper->hash_table, neighbor_idx, &neighbor_count_entries);

        // Process the neighbor bucket entries array
        for (int bi = 0; bi < neighbor_count_entries; ++bi)
        {
            int data_idx = *(int*)neighbor_bucket[bi].key;
            void* p = neighbor_bucket[bi].data;

            // O(1) duplicate check using visited array instead of O(N) linear scan
            if (visited[data_idx])
                continue;

            visited[data_idx] = true;

            // Count examined points and enforce M threshold
            checked++;
            if (checked >= hyper->M)
            {
                reached_m = 1;
                break;
            }

            // Use int-based distance computation for MNIST integer data
            double dist = hyper->distance(q, p, hyper->d, hyper->data_type, hyper->data_type);

            // Insert into min-heap (O(log N) instead of O(N) insertion sort)
            heap_insert(topN, data_idx, dist);
        }

        if (reached_m)
            break;
    }

    // Extract results from heap in sorted order
    *approx_count = topN->size;
    heap_extract_sorted(topN, approx_neighbors, approx_dists);
    heap_destroy(topN);

    // Clean up allocations
    free(visited);
    free(neighbors);

    return;
}

void range_search_hyper(const void* q, const struct SearchParams* params, int** range_neighbors, int* range_count, void* index_data)
{
    // Retrieve hypercube structure from context
    struct Hypercube* hyper = (struct Hypercube*)index_data;

    // Allocate visited array for O(1) duplicate detection (instead of O(N) linear scan)
    // This dramatically speeds up queries when examining many candidates
    bool* visited = (bool*)calloc(hyper->dataset_size, sizeof(bool));
    if (!visited)
    {
        fprintf(stderr, "Failed to allocate visited array\n");
        
        exit(EXIT_FAILURE);
    }

    // Range search optimization: allocate in chunks to avoid repeated realloc
    int range_capacity = 0;
    const int RANGE_ALLOC_CHUNK = 128; // Grow by 128 entries at a time

    // Compute the bucket index for the query point
    uint64_t q_id;
    uint64_t bucket_idx = hyper->binary_hash_function(q, hyper, &q_id);

    // Access the bucket corresponding to the computed index
    int bucket_count = 0;
    const HTEntry* bucket = hash_table_get_bucket_entries(hyper->hash_table, bucket_idx, &bucket_count);

    uint64_t* neighbors = (uint64_t*)malloc(hyper->probes * sizeof(uint64_t));
    int neighbor_count = hyper->probes;

    get_hamming_neighbors(bucket_idx, hyper->probes, hyper->kproj, neighbors);
    for (int n = 0; n < neighbor_count; n++)
    {
        uint64_t neighbor_idx = neighbors[n];
        int neighbor_count_entries = 0;
        const HTEntry* neighbor_bucket = hash_table_get_bucket_entries(hyper->hash_table, neighbor_idx, &neighbor_count_entries);

        // Process the neighbor bucket entries array
        for (int bi = 0; bi < neighbor_count_entries; ++bi)
        {
            int data_idx = *(int*)neighbor_bucket[bi].key;
            void* p = neighbor_bucket[bi].data;

            // O(1) duplicate check using visited array instead of O(N) linear scan
            if (visited[data_idx])
                continue;

            visited[data_idx] = true;

            // Use int-based distance computation for MNIST integer data
            double dist = hyper->distance(q, p, hyper->d, hyper->data_type, hyper->data_type);

            // Range search (visited array already handles deduplication)
            if (dist <= params->R)
            {
                // Allocate in chunks to avoid repeated realloc overhead
                if (*range_count >= range_capacity)
                {
                    range_capacity += RANGE_ALLOC_CHUNK;
                    int* new_range = (int*)realloc(*range_neighbors, range_capacity * sizeof(int));
                    if (!new_range)
                    {
                        fprintf(stderr, "Memory reallocation failed for range_neighbors\n");
                        free(*range_neighbors);
                        *range_neighbors = NULL;
                        *range_count = 0;
                        free(neighbors);
                        free(visited);
                    
                        exit(EXIT_FAILURE);
                    }

                    *range_neighbors = new_range;
                }
                (*range_neighbors)[(*range_count)++] = data_idx;
            }

        }
    }

    // Cleanup if no range neighbors found
    if (*range_count == 0 && *range_neighbors)
    {
        free(*range_neighbors);
        *range_neighbors = NULL;
    }

    // Clean up allocations
    free(visited);
    free(neighbors);

    return;
}

void hyper_destroy(struct Hypercube* hyper)
{
    if (!hyper)
        return;

    // Free hash table first
    if (hyper->hash_table)
        hash_table_destroy(hyper->hash_table);

    // Free hash parameters
    if (hyper->hash_params)
    {
        for (int i = 0; i < hyper->kproj; i++)
        {
            if (hyper->hash_params[i].v)
                free(hyper->hash_params[i].v);
        }

        free(hyper->hash_params);
    }

    // Free per-projection hashmaps (if any)
    // Free thresholds array
    if (hyper->thresholds)
        free(hyper->thresholds);

    // Finally free the structure itself
    free(hyper);

    return;
}
