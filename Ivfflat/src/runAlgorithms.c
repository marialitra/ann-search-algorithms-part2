#include "../include/main.h"

void run_lsh(SearchParams* params, Dataset* dataset)
{
    struct LSH* lsh = lsh_init(params, dataset);
    if (!lsh)
    {
        fprintf(stderr, "Failed to build LSH index.\n");

        exit(EXIT_FAILURE);
    }
    
    Dataset* query_set = NULL;
    if (params->dataset_type == DATA_MNIST)
        query_set = read_data_mnist(params->query_path);
    else if (params->dataset_type == DATA_SIFT)
        query_set = read_data_sift(params->query_path);

    if (query_set)
    {
        // Perform queries using LSH
        perform_query(params, dataset, query_set, lsh_index_lookup, range_search_lsh, lsh);

        for (int i = 0; i < query_set->size; i++)
            free(query_set->data[i]);

        free(query_set->data);
        free(query_set);
    }

    lsh_destroy(lsh);

    return;
}

void run_hypercube(SearchParams* params, Dataset* dataset)
{
    struct Hypercube* hyper = hyper_init(params, dataset);
    if (!hyper)
    {
        fprintf(stderr, "Failed to build Hypercube index.\n");

        exit(EXIT_FAILURE);
    }

    Dataset* query_set = NULL;
    if (params->dataset_type == DATA_MNIST)
        query_set = read_data_mnist(params->query_path);
    else if (params->dataset_type == DATA_SIFT)
        query_set = read_data_sift(params->query_path);
        
    if (query_set)
    {
        // Perform queries using Hypercube
        perform_query(params, dataset, query_set, hyper_index_lookup, range_search_hyper, hyper);

        for (int i = 0; i < query_set->size; i++)
            free(query_set->data[i]);

        free(query_set->data);
        free(query_set);
    }

    hyper_destroy(hyper);

    return;
}

void run_ivfflat(SearchParams* params, Dataset* dataset)
{
    // Build the index
    IVFFlatIndex* ivf_index = ivfflat_init(dataset, params->kclusters);
    if (!ivf_index)
    {
        fprintf(stderr, "Failed to build IVFFlat index.\n");

        exit(EXIT_FAILURE);
    }

    // Compute silhouette (prints results internally)
    // computeSilhouette(ivf_index, dataset);

    Dataset* query_set = NULL;
    if (params->dataset_type == DATA_MNIST)
        query_set = read_data_mnist(params->query_path);
    else if (params->dataset_type == DATA_SIFT)
        query_set = read_data_sift(params->query_path);

    if (query_set)
    {
        // Perform queries using IVFFlat
        perform_query(params, dataset, query_set, ivfflat_index_lookup, range_search_ivfflat, ivf_index);

        for (int i = 0; i < query_set->size; i++)
            free(query_set->data[i]);

        free(query_set->data);
        free(query_set);
    }

    ivfflat_destroy(ivf_index);

    return;
}

void run_ivfpq(SearchParams* params, Dataset* dataset)
{
    // printf("Running IVFPQ with dataset: %s\n", params->dataset_path);
    
    // Use default PQ parameters: M=8 subspaces, nbits=8 (256 centroids per subspace)
    int M = params->M;
    int nbits = params->nbits;
    
    // printf("Building IVFPQ index with k=%d clusters, M=%d subspaces, nbits=%d...\n", params->kclusters, M, nbits);
    IVFPQIndex* ivfpq_index = ivfpq_init(dataset, params->kclusters, M, nbits);
    if (!ivfpq_index)
    {
        fprintf(stderr, "Failed to build IVFPQ index.\n");

        exit(EXIT_FAILURE);
    }
    
    // Compute silhouette (prints results internally)
    // computeSilhouette(ivf_index, dataset);
    
    Dataset* query_set = NULL;
    if (params->dataset_type == DATA_MNIST)
        query_set = read_data_mnist(params->query_path);
    else if (params->dataset_type == DATA_SIFT)
        query_set = read_data_sift(params->query_path);
    
    if (query_set)
    {
        // Perform queries using IVFPQ
        perform_query(params, dataset, query_set, ivfpq_index_lookup, range_search_ivfpq, ivfpq_index);
        
        for (int i = 0; i < query_set->size; i++)
            free(query_set->data[i]);

        free(query_set->data);
        free(query_set);
    }
    
    ivfpq_destroy(ivfpq_index);

    return;
}