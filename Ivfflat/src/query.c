#include "../include/main.h"

void perform_query(const struct SearchParams* params, const struct Dataset* dataset, const struct Dataset* query_set, index_lookup lookup_func, range_search range_fun, void* index_data)
{
    // Open output file
    FILE* output_file = fopen(params->output_path, "w");
    if (!output_file)
    {
        fprintf(stderr, "Error opening output file: %s\n", params->output_path);
        
        exit(EXIT_FAILURE);
    }

    // Guard against mismatched dimensions between dataset and queries
    if (dataset->dimension != query_set->dimension)
    {
        fprintf(stderr, "dataset dimension (%d) != query dimension (%d). Using min dimension for distance.\n", dataset->dimension, query_set->dimension);
        exit(EXIT_FAILURE);
    }

    // ------------------------- Main query loop ------------------------------------
    // Iterate over each query in the query set

    // Print format to print in the output the name of the algorithm we are using
    fprintf(output_file, "%s\n", params->algorithm == ALG_LSH ? "LSH" :
            params->algorithm == ALG_HYPERCUBE ? "Hypercube" :
            params->algorithm == ALG_IVFFLAT ? "IVFFlat" : "IVFPQ");

    // Parallel query processing
    #pragma omp parallel for schedule(dynamic)
    for (int q_idx = 0; q_idx < query_set->size; q_idx++)
    {
        void* q = query_set->data[q_idx];
        int* approx_neighbors = (int*)malloc(params->N * sizeof(int));
        if (!approx_neighbors)
        {
            fprintf(stderr, "Failed to allocate space for approx_neighbors!\n");

            exit(EXIT_FAILURE);
        }

        double* approx_dists = (double*)malloc(params->N * sizeof(double));
        if (!approx_dists)
        {
            fprintf(stderr, "Failed to allocate space for approx_dists!\n");

            exit(EXIT_FAILURE);
        }

        int approx_count = 0;
        int* range_neighbors = NULL;
        int range_count = 0;

        for(int i = 0; i < params->N; i++)
        {
            approx_neighbors[i] = -1;
            approx_dists[i] = INFINITY;
        }

        // Algorithm-specific index lookup
        lookup_func(q, params, approx_neighbors, approx_dists, &approx_count, index_data);

        // Algorithm-specific range search function
        if(params->range_search)
            range_fun(q, params, &range_neighbors, &range_count, index_data);

        
        // Thread-local buffer for output
        char buffer[4096];
        int offset = 0;
        offset += snprintf(buffer + offset, sizeof(buffer) - offset, "\nQuery: %d\n", q_idx);

        for (int i = 0; i < params->N && i < approx_count; i++)
            offset += snprintf(buffer + offset, sizeof(buffer) - offset, "Nearest neighbor-%d: %d\n", i + 1, approx_neighbors[i]);

        if (params->range_search && range_count > 0)
        {
            offset += snprintf(buffer + offset, sizeof(buffer) - offset, "\nR-near neighbors:\n");
            for (int i = 0; i < range_count; i++)
                offset += snprintf(buffer + offset, sizeof(buffer) - offset, "%d\n", range_neighbors[i]);
        }

        // Serialize writes to the file
        #pragma omp critical
        {
            fputs(buffer, output_file);
        }

        free(approx_neighbors);
        free(approx_dists);
        free(range_neighbors);

    }

    return;
}