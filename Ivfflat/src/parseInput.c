#include "../include/main.h"

int parse_arguments(int argc, char** argv, SearchParams* params)
{
    // Default values
    memset(params, 0, sizeof(SearchParams));
    params->dataset_type = DATA_NONE;
    params->algorithm = ALG_NONE;
    params->range_search = false;
    params->N = -1;
    params->R = -1.0;
    params->seed = -1;
    params->k = -1;
    params->L = -1;
    params->w = -1.0;
    params->kproj = -1;
    params->M = -1;
    params->probes = -1;
    params->kclusters = -1;
    params->nprobe = -1;
    params->nbits = -1;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc)
            strncpy(params->dataset_path, argv[++i], sizeof(params->dataset_path));
        else if (strcmp(argv[i], "-q") == 0 && i + 1 < argc)
            strncpy(params->query_path, argv[++i], sizeof(params->query_path));
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
            strncpy(params->output_path, argv[++i], sizeof(params->output_path));
        else if (strcmp(argv[i], "-type") == 0 && i + 1 < argc)
        {
            i++;
            if (strcmp(argv[i], "mnist") == 0)
                params->dataset_type = DATA_MNIST;
            else if (strcmp(argv[i], "sift") == 0)
                params->dataset_type = DATA_SIFT;
        }
        else if (strcmp(argv[i], "-range") == 0 && i + 1 < argc)
            params->range_search = (strcmp(argv[++i], "true") == 0);
        else if (strcmp(argv[i], "-N") == 0 && i + 1 < argc)
        {
            params->N = atoi(argv[++i]);
            if(params->N < 0)
            {
                fprintf(stderr, "Error: Neighbors (N) cannot be a negative number.\n");
                print_usage(params->algorithm);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "-R") == 0 && i + 1 < argc)
        {
            params->R = atof(argv[++i]);
            if(params->R < 0)
            {
                fprintf(stderr, "Error: Range search (R) cannot be a negative number.\n");
                print_usage(params->algorithm);
                exit(EXIT_FAILURE);
            }
        }

        // Algorithm-specific
        else if (strcmp(argv[i], "-lsh") == 0)
            params->algorithm = ALG_LSH;
        else if (strcmp(argv[i], "-hypercube") == 0)
            params->algorithm = ALG_HYPERCUBE;
        else if (strcmp(argv[i], "-ivfflat") == 0)
            params->algorithm = ALG_IVFFLAT;
        else if (strcmp(argv[i], "-ivfpq") == 0)
            params->algorithm = ALG_IVFPQ;

        // LSH
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc)
        {
            params->k = atoi(argv[++i]);
            if(params->k < 0)
            {
                fprintf(stderr, "Error: k for LSH cannot be a negative number.\n");
                print_usage(params->algorithm);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "-L") == 0 && i + 1 < argc)
        {
            params->L = atoi(argv[++i]);
            if(params->L < 0)
            {
                fprintf(stderr, "Error: L for LSH cannot be a negative number.\n");
                print_usage(params->algorithm);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc)
        {
            params->w = atof(argv[++i]);
            if(params->w < 0)
            {
                fprintf(stderr, "Error: w for LSH cannot be a negative number.\n");
                print_usage(params->algorithm);
                exit(EXIT_FAILURE);
            }
        }

        // Hypercube
        else if (strcmp(argv[i], "-kproj") == 0 && i + 1 < argc)
        {
            params->kproj = atoi(argv[++i]);
            if(params->kproj < 0)
            {
                fprintf(stderr, "Error: kproj for Hypercube cannot be a negative number.\n");
                print_usage(params->algorithm);
                exit(EXIT_FAILURE);
            }
            else if(params->kproj >= 30)
            {
                fprintf(stderr, "Error: kproj for Hypercube cannot greater than or equal to 30.\n");
                print_usage(params->algorithm);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "-M") == 0 && i + 1 < argc)
        {
            params->M = atoi(argv[++i]);
            if(params->M < 0)
            {
                fprintf(stderr, "Error: M for Hypercube or IVFPQ cannot be a negative number.\n");
                print_usage(params->algorithm);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "-probes") == 0 && i + 1 < argc)
        {
            params->probes = atoi(argv[++i]);
            if(params->probes < 0)
            {
                fprintf(stderr, "Error: probes for Hypercube cannot be a negative number.\n");
                print_usage(params->algorithm);
                exit(EXIT_FAILURE);
            }
        }

        // IVF / IVFPQ
        else if (strcmp(argv[i], "-kclusters") == 0 && i + 1 < argc)
        {
            params->kclusters = atoi(argv[++i]);
            if(params->kclusters < 0)
            {
                fprintf(stderr, "Error: kclusters for IVFFLAT or IVFPQ cannot be a negative number.\n");
                print_usage(params->algorithm);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "-nprobe") == 0 && i + 1 < argc)
        {
            params->nprobe = atoi(argv[++i]);
            if(params->nprobe < 0)
            {
                fprintf(stderr, "Error: nprobe for IVFFLAT or IVFPQ cannot be a negative number.\n");
                print_usage(params->algorithm);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc)
        {
            params->seed = atoi(argv[++i]);
            if(params->seed < 0)
            {
                fprintf(stderr, "Error: seed for IVFFLAT or IVFPQ cannot be a negative number.\n");
                print_usage(params->algorithm);
                exit(EXIT_FAILURE);
            }
        }

        // IVFPQ only
        else if (strcmp(argv[i], "-nbits") == 0 && i + 1 < argc)
        {
            params->nbits = atoi(argv[++i]);
            if(params->nbits < 0)
            {
                fprintf(stderr, "Error: nbits for IVFPQ cannot be a negative number.\n");
                print_usage(params->algorithm);
                exit(EXIT_FAILURE);
            }
            else if(params->nbits > 16)
            {
                fprintf(stderr, "Error: nbits for IVFPQ cannot be greater than 16.\n");
                print_usage(params->algorithm);
                exit(EXIT_FAILURE);
            }
        }

        else
        {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            return -1;
        }
    }


    // Minimal checks
    if (params->algorithm == ALG_NONE)
    {
        fprintf(stderr, "Error: Missing algorithm.\n");
        print_usage(params->algorithm);
        exit(EXIT_FAILURE);
    }
    else if(params->dataset_type == DATA_NONE)
    {
        fprintf(stderr, "Error: Missing dataset type.\n");
        print_usage(params->algorithm);
        exit(EXIT_FAILURE);
    }

    // Now we can set the default values because
    // We need to knwo which algorithm we are using
    // And which dataset we have


    // Shared
    if (params->N == -1)
        params->N = 1;
    if (params->seed == -1)
        params->seed = 1;
    if(params->dataset_type == DATA_MNIST && params->R == -1.0)
        params->R = 2000;
    else if(params->R == -1.0)
        params->R = 2;

    // Specific
    if(params->algorithm == ALG_LSH && (params->k == -1 || params->L == -1 || params->w == -1.0))
    {
        params->k = 4;
        params->L = 5;
        params->w = 4.0;
    }
    else if(params->algorithm == ALG_HYPERCUBE && (params->kproj == -1 || params->w == -1.0 || params->M == -1 || params->probes == -1))
    {
        params->kproj = 14;
        params->w = 4.0;
        params->M = 10;
        params->probes = 2;
    }
    else if(params->algorithm == ALG_IVFFLAT && (params->kclusters == -1 || params->nprobe == -1))
    {
        params->kclusters = 50;
        params->nprobe = 5;
    }
    else if(params->algorithm == ALG_IVFPQ && (params->kclusters == -1 || params->nprobe == -1 || params->nbits == -1 || params->M == -1))
    {
        params->kclusters = 50;
        params->nprobe = 5;
        params->nbits = 8;
        params->M = 16;
    }
    printf("Using seed: %u\n", params->seed);
    printf("Number of nearest neighbors (N): %d\n", params->N);
    printf("Radius (R): %f\n", params->R);
    return 0;
}

void print_usage(AlgorithmType type)
{
    printf("Usage example:\n");

    if(type == ALG_LSH)
        printf("  /search -d <input file> -q <query file> -k <int> -L <int> -w <double> -o <output file> -N <number of nearest> -R <radius> -type <flag> -lsh -range <true|false>\n");
    else if(type == ALG_HYPERCUBE)
        printf("  ./search -d <input file> -q <query file> -kproj <int> -w <double> -M <int> -probes <int> -o <output file> -N <number of nearest> -R <radius> -type <flag> -range <true|false> -hypercube\n");
    else if(type == ALG_IVFFLAT)
        printf("  ./search -d <input file> -q <query file> -kclusters <int> -nprobe <int> -o <output file> -N <number of nearest> -R <radius> -type <flag> -range <true|false> -ivfflat -seed <int>\n");
    else if(type == ALG_IVFPQ)
        printf("  ./search -d <input file> -q <query file> -kclusters <int> -nprobe <int> -M <int> -o <output file> -N <number of nearest> -R <radius> -type <flag> -nbits <int> -range <true|false> -ivfpq -seed <int>\n");
    else if(type == ALG_NONE)
    {
        printf(" You have to indicate a specific algorithm in the Makefile in order to run it.\n The available options are the following:\n");
        printf("- lsh\n");
        printf("- hypercube\n");
        printf("- ivfflat\n");
        printf("- ivfpq\n");
    }

    return;
}