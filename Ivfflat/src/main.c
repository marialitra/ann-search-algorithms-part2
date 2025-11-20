#include "../include/main.h"

int main(int argc, char **argv)
{
    SearchParams params;
    if (parse_arguments(argc, argv, &params) != 0)
        exit(EXIT_FAILURE);

    printf("Algorithm selected: ");

    Dataset* dataset = NULL;
    if (params.dataset_type == DATA_MNIST)
        dataset = read_data_mnist(params.dataset_path);
    else if(params.dataset_type == DATA_SIFT)
        dataset = read_data_sift(params.dataset_path);
        
    if (dataset == NULL)
    {
        perror("Dataset not correct\n");
        exit(EXIT_FAILURE);
    }

    if(params.dataset_type == DATA_MNIST)
        printf("MNIST Dataset loaded: %d points of dimension %d\n", dataset->size, dataset->dimension);
    else if(params.dataset_type == DATA_SIFT)
        printf("SIFT Dataset loaded: %d points of dimension %d\n", dataset->size, dataset->dimension);

    unsigned int seed = 42;
    if (params.seed)
    {
        seed = params.seed;
        srand(seed);
    }
    else
        srand(time(NULL));

    switch (params.algorithm)
    {
        case ALG_LSH:
            printf("LSH\n");
            run_lsh(&params, dataset);
            break;

        case ALG_HYPERCUBE:
            printf("Hypercube\n");
            run_hypercube(&params, dataset);
            break;

        case ALG_IVFFLAT:
            printf("IVFFlat\n");
            run_ivfflat(&params, dataset);
            break;

        case ALG_IVFPQ:
            printf("IVFPQ\n");
            run_ivfpq(&params, dataset);
            break;

        default:
            printf("Unknown\n");
            break;
    }

    // Free loaded dataset
    free_dataset(dataset);

    return EXIT_SUCCESS;
}