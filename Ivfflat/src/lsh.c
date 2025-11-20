#include "../include/main.h"

// Helper function to initialize the linear combinations needed (r_i)
static void init_linear_combinations(LSH* lsh)
{
    lsh->linear_combinations = malloc(lsh->L * sizeof(int32_t *));
    if (!lsh->linear_combinations)
    {
        fprintf(stderr, "Memory allocation failed for linear combinations.\n");

        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < lsh->L; i++)
    {
        lsh->linear_combinations[i] = malloc(lsh->k * sizeof(int32_t));
        if (!lsh->linear_combinations[i])
        {
            fprintf(stderr, "Memory allocation failed for linear combinations row.\n");

            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < lsh->k; j++)
        {
            uint32_t lc = (uint32_t)(rand() % (R_RANGE - 1)) + 1;

            lsh->linear_combinations[i][j] = (int32_t)lc;
        }
    }

    return;
}

LSH* lsh_init(const struct SearchParams* params, const struct Dataset* dataset)
{
    // Allocate memory for LSH structure and set parameters
    LSH* lsh = (LSH*)malloc(sizeof(LSH));
    if (!lsh)
    {
        fprintf(stderr, "Memory allocation failed for lsh.\n");

        exit(EXIT_FAILURE);
    }

    lsh->d = dataset->dimension;
    lsh->L = params->L;
    lsh->k = params->k;
    lsh->w = params->w;
    lsh->dataset_size = dataset->size;

    lsh->table_size = nearest_prime(dataset->size / 4);
    // Set M to the largest 32-bit prime (2^32 - 5) to avoid overflow in older impls
    lsh->num_of_buckets = 4294967291ULL;

    // Store data type and select distance function
    lsh->data_type = dataset->data_type;
    lsh->distance = euclidean_distance; // uint8-based distance

    // Allocate memory for per-table hash parameters (L x k)
    lsh->hash_params = (LSH_hash_function**)malloc(lsh->L * sizeof(LSH_hash_function*));
    if (!lsh->hash_params)
    {
        fprintf(stderr, "Failed to allocate space for the hash parameters of LSH!\n");
        lsh_destroy(lsh);

        exit(EXIT_FAILURE);
    }

    // For each table, generate K independent hash functions (v, t)
    for (int tbl = 0; tbl < lsh->L; tbl++)
    {
        lsh->hash_params[tbl] = (LSH_hash_function*)malloc(lsh->k * sizeof(LSH_hash_function));
        if (!lsh->hash_params[tbl])
        {
            fprintf(stderr, "Failed to allocate space for hash parameters indices!\n");
            lsh_destroy(lsh);

            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < lsh->k; i++)
        {
            lsh->hash_params[tbl][i].v = (float*)malloc(lsh->d * sizeof(float));
            if (!lsh->hash_params[tbl][i].v)
            {
                fprintf(stderr, "Failed to allocate space for hash parameters indices!\n");
                lsh_destroy(lsh);

                exit(EXIT_FAILURE);
            }

            generate_random_vector(lsh->hash_params[tbl][i].v, lsh->d);
            normalize_vector(lsh->hash_params[tbl][i].v, lsh->d);

            // Calculate t uniformly in [0, w)
            float zero = 0.0f;
            float wval = lsh->w;
            lsh->hash_params[tbl][i].t = uniform_distribution(&zero, &wval);
        }
    }

    init_linear_combinations(lsh);

    // Create a hash table for each hash table in LSH
    lsh->hash_tables = (HashTable*)malloc(lsh->L * sizeof(HashTable));

    for (int i = 0; i < lsh->L; i++)
    {
        lsh->hash_tables[i] = hash_table_create(lsh->table_size, sizeof(int), NULL, NULL, hash_function_lsh, lsh, i, &(dataset->dimension));
        if (!lsh->hash_tables[i])
        {
            fprintf(stderr, "Failed to allocate space for LSH's hashtable!\n");
            lsh_destroy(lsh);
            exit(EXIT_FAILURE);
        }
    }

    // Insert all points in all hash tables
    for (int i = 0; i < dataset->size; i++)
    {
        // Verify point is valid before insertion
        if (!dataset->data || !dataset->data[i])
        {
            printf("Warning: dataset[%d] is NULL\n", i);
            continue;
        }

        for (int j = 0; j < lsh->L; j++)
            hash_table_insert(lsh->hash_tables[j], &i, dataset->data[i]);
    }

    return lsh;
}

void lsh_index_lookup(const void* q, const struct SearchParams* params, int* approx_neighbors, double* approx_dists, int* approx_count, void* index_data)
{
    // Cast index_data to LSH structure
    struct LSH* lsh = (struct LSH*)index_data;
    if(index_data == NULL)
    {
        fprintf(stderr, "LSH struct is not allocated!\n");

        exit(EXIT_FAILURE);
    }

    // Use a visited array for O(1) duplicate detection across all tables
    bool* visited = (bool*)calloc(lsh->dataset_size, sizeof(bool));
    if (!visited)
    {
        fprintf(stderr, "Failed to allocate visited array.\n");
        
        exit(EXIT_FAILURE);
    }

    // Create min-heap for top-N neighbors
    MinHeap* topN = heap_create(params->N);
    if (!topN)
    {
        fprintf(stderr, "Failed to allocate min-heap.\n");
        free(visited);
        
        exit(EXIT_FAILURE);
    }

    int flagged = 0;
    // For each hash table, compute the bucket index for the query point
    for (int tbl_idx = 0; tbl_idx < lsh->L; tbl_idx++)
    {
        uint64_t q_id = 0ULL;
        int bucket_idx = hash_func_impl_lsh(q, lsh, tbl_idx, &q_id);

        int bucket_count = 0;
        const HTEntry* bucket = hash_table_get_bucket_entries(lsh->hash_tables[tbl_idx], bucket_idx, &bucket_count);
        for (int bi = 0; bi < bucket_count; ++bi)
        {
            int data_idx = *(int*)bucket[bi].key;
            void* p = bucket[bi].data;

            if (bucket[bi].ID != q_id)
                continue;

            // Skip duplicate candidates across tables using visited array
            if (data_idx >= 0 && data_idx < lsh->dataset_size && visited[data_idx])
                continue;

            // Mark as visited now to avoid reprocessing and count as unique candidate
            if (data_idx >= 0 && data_idx < lsh->dataset_size)
                visited[data_idx] = true;

            // Compute distance for this candidate using type-aware function
            double dist = lsh->distance(q, p, lsh->d, lsh->data_type, lsh->data_type);

            // Insert into min-heap (O(log N) instead of O(N) insertion sort)
            heap_insert(topN, data_idx, dist);

        }
    }

    // Extract results from heap in sorted order
    *approx_count = topN->size;
    heap_extract_sorted(topN, approx_neighbors, approx_dists);
    heap_destroy(topN);
    free(visited);

    return;
}

void range_search_lsh(const void* q, const struct SearchParams* params, int** range_neighbors, int* range_count, void* index_data)
{
    // Cast index_data to LSH structure
    struct LSH* lsh = (struct LSH*)index_data;
    if(index_data == NULL)
    {
        fprintf(stderr, "LSH struct is not allocated!\n");

        exit(EXIT_FAILURE);
    }

    // Use a visited array for O(1) duplicate detection across all tables
    bool* visited = (bool*)calloc(lsh->dataset_size, sizeof(bool));
    if (!visited)
    {
        fprintf(stderr, "Failed to allocate visited array.\n");
        
        exit(EXIT_FAILURE);
    }

    // Range neighbors dynamic capacity (grow in chunks instead of per-item realloc)
    const int RANGE_ALLOC_CHUNK = 128;
    int range_capacity = (*range_neighbors && *range_count > 0) ? *range_count : 0;

    // For each hash table, compute the bucket index for the query point
    for (int tbl_idx = 0; tbl_idx < lsh->L; tbl_idx++)
    {
        uint64_t q_id = 0ULL;
        int bucket_idx = hash_func_impl_lsh(q, lsh, tbl_idx, &q_id);

        int bucket_count = 0;
        const HTEntry* bucket = hash_table_get_bucket_entries(lsh->hash_tables[tbl_idx], bucket_idx, &bucket_count);
        for (int bi = 0; bi < bucket_count; ++bi)
        {
            int data_idx = *(int*)bucket[bi].key;
            void* p = bucket[bi].data;

            if (bucket[bi].ID != q_id)
                continue;

            // Skip duplicate candidates across tables using visited array
            if (data_idx >= 0 && data_idx < lsh->dataset_size && visited[data_idx])
                continue;

            // Mark as visited now to avoid reprocessing and count as unique candidate
            if (data_idx >= 0 && data_idx < lsh->dataset_size)
                visited[data_idx] = true;

            // Compute distance for this candidate using type-aware function
            double dist = lsh->distance(q, p, lsh->d, lsh->data_type, lsh->data_type);

            if (params->range_search && dist <= params->R)
            {
                // Ensure capacity and append without per-item realloc
                if (*range_count >= range_capacity)
                {
                    int new_capacity = range_capacity + RANGE_ALLOC_CHUNK;
                    int *new_buf = (int *)realloc(*range_neighbors, new_capacity * sizeof(int));
                    if (!new_buf)
                    {
                        fprintf(stderr, "Memory reallocation failed for range neighbors.\n");
                        free(*range_neighbors);
                        *range_neighbors = NULL;
                        *range_count = 0;
                        free(visited);
                        
                        exit(EXIT_FAILURE);
                    }
                    *range_neighbors = new_buf;
                    range_capacity = new_capacity;
                }

                // Append this neighbor
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

    free(visited);

    return;
}

void lsh_destroy(struct LSH* lsh)
{
    if (!lsh)
        return;

    // Free hash params and their vectors
    if (lsh->hash_params)
    {
        for (int tbl = 0; tbl < lsh->L; tbl++)
        {
            if (lsh->hash_params[tbl])
            {
                for (int i = 0; i < lsh->k; i++)
                {
                    if (lsh->hash_params[tbl][i].v)
                        free(lsh->hash_params[tbl][i].v);
                }
                free(lsh->hash_params[tbl]);
            }
        }
        free(lsh->hash_params);
    }

    // Free linear combinations
    if (lsh->linear_combinations)
    {
        for (int i = 0; i < lsh->L; i++)
        {
            if (lsh->linear_combinations[i])
                free(lsh->linear_combinations[i]);
        }
        free(lsh->linear_combinations);
    }

    // Destroy hash tables (guard duplicates) and free array
    if (lsh->hash_tables)
    {
        for (int i = 0; i < lsh->L; i++)
        {
            if (lsh->hash_tables[i])
            {
                int duplicate = 0;
                for (int j = 0; j < i; j++)
                {
                    if (lsh->hash_tables[j] == lsh->hash_tables[i])
                    {
                        duplicate = 1;
                        break;
                    }
                }
                if (!duplicate)
                {
                    hash_table_destroy(lsh->hash_tables[i]);
                }
                lsh->hash_tables[i] = NULL;
            }
        }
        free(lsh->hash_tables);
    }

    free(lsh);

    return;
}