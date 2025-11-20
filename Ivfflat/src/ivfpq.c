#include "../include/main.h"

// Helper: Split vector into M parts
static inline void split_vector_into_parts(const void* vec, DataType data_type, int d, int M, float** parts)
{
    int d_sub = d / M;
    
    if (data_type == DATA_TYPE_FLOAT)
    {
        const float *fvec = (const float *)vec;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < d_sub; j++)
                parts[i][j] = fvec[i * d_sub + j];
        }
    }
    else
    {  // DATA_TYPE_UINT8
        const uint8_t *uvec = (const uint8_t *)vec;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < d_sub; j++)
                parts[i][j] = (float)uvec[i * d_sub + j];
        }
    }

    return;
}

// Helper: Compute residual vector r(x) = x - c(x)
static inline void compute_residual(const void* vec, const float* centroid, DataType data_type, int d, float* residual)
{
    if (data_type == DATA_TYPE_FLOAT)
    {
        const float *fvec = (const float *)vec;
        for (int j = 0; j < d; j++)
            residual[j] = fvec[j] - centroid[j];
    } 
    else
    {  // DATA_TYPE_UINT8
        const uint8_t *uvec = (const uint8_t *)vec;
        for (int j = 0; j < d; j++)
            residual[j] = (float)uvec[j] - centroid[j];
    }

    return;
}

// Helper: Run Lloyd's on a subspace to get s centroids
static float** run_lloyd_on_subspace(float** subspace_data, int n_points, int d_sub, int s, int max_iters)
{
    if (n_points == 0 || d_sub == 0)
        return NULL;
    
    // Allocate centroids
    float** centroids = (float**)malloc(s * sizeof(float *));
    if(!centroids)
    {
        fprintf(stderr, "Failed to allocate space for the centroids!\n");

        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < s; i++)
        centroids[i] = (float *)calloc(d_sub, sizeof(float));
    
    // Initialize centroids with KMeans++ for better initialization
    int* assignments = (int*)malloc(n_points * sizeof(int));
    if(!assignments)
    {
        fprintf(stderr, "Failed to allocate space for the assignments!\n");

        exit(EXIT_FAILURE);
    }

    bool* selected = (bool*)calloc(n_points, sizeof(bool));
    if(!selected)
    {
        fprintf(stderr, "Failed to allocate space for the selected!\n");

        exit(EXIT_FAILURE);
    }
    
    // Step 1: Choose first centroid uniformly at random
    int first_idx = rand() % n_points;
    memcpy(centroids[0], subspace_data[first_idx], d_sub * sizeof(float));
    selected[first_idx] = true;
    
    // Step 2: Choose remaining centroids using D² weighting
    double* min_dist_sq = (double*)malloc(n_points * sizeof(double));
    if(!min_dist_sq)
    {
        fprintf(stderr, "Failed to allocate space for the min_dist_sq!\n");

        exit(EXIT_FAILURE);
    }
    
    for (int t = 1; t < s && t < n_points; t++)
    {
        // Compute D²(x) = min distance squared to nearest existing centroid (parallel)
        double sum_dist_sq = 0.0;
        
        #pragma omp parallel for reduction(+:sum_dist_sq) schedule(static)
        for (int i = 0; i < n_points; i++)
        {
            if (selected[i])
            {
                min_dist_sq[i] = 0.0;
                continue;
            }
            
            // Find distance to nearest centroid so far
            double best_dist_sq = DBL_MAX;
            for (int c = 0; c < t; c++)
            {
                double dist_sq = 0.0;
                for (int j = 0; j < d_sub; j++)
                {
                    double diff = subspace_data[i][j] - centroids[c][j];
                    dist_sq += diff * diff;
                }

                if (dist_sq < best_dist_sq)
                    best_dist_sq = dist_sq;
            }
            min_dist_sq[i] = best_dist_sq;
            sum_dist_sq += best_dist_sq;
        }
        
        // Choose next centroid with probability proportional to D²(x)
        double random_val = ((double)rand() / RAND_MAX) * sum_dist_sq;
        double cumulative = 0.0;
        int chosen_idx = -1;
        
        for (int i = 0; i < n_points; i++)
        {
            if (selected[i])
                continue;

            cumulative += min_dist_sq[i];

            if (cumulative >= random_val)
            {
                chosen_idx = i;
                break;
            }
        }
        
        // Fallback in case of numerical issues
        if (chosen_idx < 0)
        {
            for (int i = 0; i < n_points; i++)
            {
                if (!selected[i])
                {
                    chosen_idx = i;
                    break;
                }
            }
        }
        
        memcpy(centroids[t], subspace_data[chosen_idx], d_sub * sizeof(float));
        selected[chosen_idx] = true;
    }
    
    free(min_dist_sq);
    free(selected);
    
    // Lloyd's iterations with convergence checking (like IVFFlat)
    double epsilon = 1e-4;
    bool changed_any = false;
    for (int iter = 0; iter < max_iters; iter++)
    {
        // Assign points to nearest centroid
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_points; i++)
        {
            float best_dist = FLT_MAX;
            int best_c = 0;

            for (int c = 0; c < s; c++)
            {
                float dist = 0.0f;
                for (int j = 0; j < d_sub; j++)
                {
                    float diff = subspace_data[i][j] - centroids[c][j];
                    dist += diff * diff;
                }

                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }
        
        // Recompute centroids and check for convergence
        int* counts = (int*)calloc(s, sizeof(int));
        if(!counts)
        {
            fprintf(stderr, "Failed to allocate space for the counts!\n");

            exit(EXIT_FAILURE);
        }

        float** new_centroids = (float**)malloc(s * sizeof(float*));
        if(!new_centroids)
        {
            fprintf(stderr, "Failed to allocate space for the new_centroids!\n");

            exit(EXIT_FAILURE);
        }

        for (int c = 0; c < s; c++)
        {
            new_centroids[c] = (float*)calloc(d_sub, sizeof(float));

            if(!new_centroids[c])
            {
                fprintf(stderr, "Failed to allocate space for the new_centroids!\n");

                exit(EXIT_FAILURE);
            }
        }
        
        // Sum points assigned to each centroid
        for (int i = 0; i < n_points; i++)
        {
            int c = assignments[i];
            counts[c]++;
            for (int j = 0; j < d_sub; j++)
                new_centroids[c][j] += subspace_data[i][j];
        }
        
        // Average and check for convergence (like IVFFlat recompute_centroids)
        changed_any = false;
        for (int c = 0; c < s; c++)
        {
            if (counts[c] > 0)
            {
                for (int j = 0; j < d_sub; j++)
                    new_centroids[c][j] /= counts[c];
            }
            
            // Check centroid shift (Euclidean distance)
            double shift = 0.0;
            for (int j = 0; j < d_sub; j++)
            {
                double diff = new_centroids[c][j] - centroids[c][j];
                shift += diff * diff;
            }

            shift = sqrt(shift);
            
            if (shift > epsilon)
                changed_any = true;
            
            // Replace old centroid with new
            free(centroids[c]);
            centroids[c] = new_centroids[c];
        }
        
        free(new_centroids);
        free(counts);
        
        // Early stopping if converged (like IVFFlat)
        if (!changed_any)
        {
            // printf("Lloyd's converged in %d iterations\n", iter + 1);
            break;
        }
    }

    // if (changed_any)
    //     printf("----Lloyd's converged in %d iterations\n", max_iters);

    free(assignments);

    return centroids;
}

// Helper: Find nearest centroid in a subspace
static inline int find_nearest_subspace_centroid(const float* subvec, float** centroids, int s, int d_sub)
{
    float best_dist = FLT_MAX;
    int best_idx = 0;
    
    for (int i = 0; i < s; i++)
    {
        float dist = 0.0f;
        for (int j = 0; j < d_sub; j++)
        {
            float diff = subvec[j] - centroids[i][j];
            dist += diff * diff;
        }

        if (dist < best_dist)
        {
            best_dist = dist;
            best_idx = i;
        }
    }

    return best_idx;
}

// IVFPQ Initialization following the algorithm from the image
IVFPQIndex* ivfpq_init(Dataset* dataset, int k_clusters, int M, int nbits)
{
    if (!dataset || k_clusters <= 0 || M <= 0 || nbits <= 0)
    {
        fprintf(stderr, "Invalid IVFPQ parameters\n");

        exit(EXIT_FAILURE);
    }
    
    if (dataset->dimension % M != 0)
    {
        fprintf(stderr, "Dimension %d must be divisible by M=%d\n", dataset->dimension, M);

        exit(EXIT_FAILURE);    
    }
    
    // printf("Building IVFPQ index: k=%d, M=%d, nbits=%d\n", k_clusters, M, nbits);
    
    IVFPQIndex* index = (IVFPQIndex*)malloc(sizeof(IVFPQIndex));
    if(!index)
    {
        fprintf(stderr, "Failed to allocate space for the IVFPQ index!\n");

        exit(EXIT_FAILURE);
    }
    
    index->k = k_clusters;
    index->d = dataset->dimension;
    index->data_type = dataset->data_type;
    index->dataset = dataset;
    
    // Step 1 & 2: Run Lloyd's to get coarse centroids (reuse IVFFlat logic)
    // printf("Step 1-2: Computing coarse centroids with Lloyd's algorithm...\n");
    IVFFlatIndex* ivf_temp = ivfflat_init(dataset, k_clusters);
    if (!ivf_temp)
    {
        free(index);
        
        exit(EXIT_FAILURE);
    }
    
    // Copy centroids from IVFFlat
    index->centroids = ivf_temp->centroids;
    ivf_temp->centroids = NULL;  // Prevent double-free
    
    // Initialize PQ configuration
    index->pq.M = M;
    index->pq.nbits = nbits;
    index->pq.s = 1 << nbits;  // s = 2^nbits
    index->pq.d_sub = dataset->dimension / M;
    
    // printf("PQ config: M=%d subspaces, s=%d centroids per subspace, d_sub=%d\n", index->pq.M, index->pq.s, index->pq.d_sub);
    
    // Allocate subspace centroids [M][s][d_sub]
    index->pq.subspace_centroids = (float***)malloc(M * sizeof(float**));
    if(!index->pq.subspace_centroids)
    {
        fprintf(stderr, "Failed to allocate space for the subspaces centroids!\n");

        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < M; i++)
        index->pq.subspace_centroids[i] = NULL;
    
    // Allocate inverted lists
    index->lists = (IVFPQList*)calloc(k_clusters, sizeof(IVFPQList));
    if(!index->lists)
    {
        fprintf(stderr, "Failed to allocate space for the lists!\n");

        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < k_clusters; i++)
    {
        index->lists[i].cluster_id = i;
        index->lists[i].count = 0;
        index->lists[i].capacity = 0;
        index->lists[i].entries = NULL;
    }
    
    // Step 3-5: Collect residuals from ALL clusters and train global PQ codebooks
    // printf("Step 3-5: Computing residuals from all clusters and training PQ codebooks...\n");
    
    // First, count total residuals and decide on sampling
    int total_available = dataset->size;
    
    // Sample at most 10000 residuals for faster training
    int max_train_samples = (total_available < 10000) ? total_available : 10000;
    int sample_rate = (total_available > max_train_samples) ? (total_available / max_train_samples) : 1;
    
    // printf("Collecting %d/%d residual vectors for PQ training (sample rate: 1/%d)...\n", max_train_samples, total_available, sample_rate);
    
    float** all_residuals = (float**)malloc(max_train_samples * sizeof(float*));
    if(!all_residuals)
    {
        fprintf(stderr, "Failed to allocate space for the all_residuals!\n");

        exit(EXIT_FAILURE);
    }

    int residual_idx = 0;
    
    for (int c = 0; c < k_clusters && residual_idx < max_train_samples; c++)
    {
        InvertedList* ivf_list = &ivf_temp->lists[c];
        for (int i = 0; i < ivf_list->count && residual_idx < max_train_samples; i++)
        {
            // Sample points based on sample_rate
            if (i % sample_rate != 0)
                continue;
            
            all_residuals[residual_idx] = (float*)malloc(dataset->dimension * sizeof(float));
            if(!all_residuals)
            {
                fprintf(stderr, "Failed to allocate space for the all_residuals!\n");

                exit(EXIT_FAILURE);
            }
            compute_residual(ivf_list->points[i], index->centroids[c], dataset->data_type, dataset->dimension, all_residuals[residual_idx]);
            residual_idx++;
        }
    }
    
    int actual_samples = residual_idx;
    // printf("Collected %d residual samples.\n", actual_samples);
    
    // Train each subspace quantizer on the sampled residuals
    // printf("Training %d subspace quantizers...\n", M);
    for (int m = 0; m < M; m++)
    {
        // printf("  Training subspace %d/%d...\n", m+1, M);
        fflush(stdout);
        
        // Extract subspace m from all residuals
        float** subspace_data = (float**)malloc(actual_samples * sizeof(float*));
        if(!subspace_data)
        {
            fprintf(stderr, "Failed to allocate space for the subspace_data!\n");

            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < actual_samples; i++)
        {
            subspace_data[i] = (float*)malloc(index->pq.d_sub * sizeof(float));
            if(!subspace_data)
            {
                fprintf(stderr, "Failed to allocate space for the subspace_data!\n");

                exit(EXIT_FAILURE);
            }

            for (int j = 0; j < index->pq.d_sub; j++)
                subspace_data[i][j] = all_residuals[i][m * index->pq.d_sub + j];
        }
        
        // Train Lloyd's on this subspace (use fewer iterations for speed)
        index->pq.subspace_centroids[m] = run_lloyd_on_subspace(subspace_data, actual_samples, index->pq.d_sub, index->pq.s, 100);
        
        // Free subspace data
        for (int i = 0; i < actual_samples; i++)
            free(subspace_data[i]);

        free(subspace_data);
    }
    
    // Free all residuals
    for (int i = 0; i < actual_samples; i++)
        free(all_residuals[i]);

    free(all_residuals);
    // printf("PQ codebooks trained successfully!\n");
    
    // Step 6-8: Encode all points and build inverted lists
    // printf("Step 6-8: Encoding points with PQ and building inverted lists...\n");
    
    for (int c = 0; c < k_clusters; c++)
    {
        InvertedList* ivf_list = &ivf_temp->lists[c];
        IVFPQList* pq_list = &index->lists[c];
        
        pq_list->capacity = ivf_list->count;
        pq_list->entries = (IVFPQEntry*)malloc(ivf_list->count * sizeof(IVFPQEntry));
        if(!pq_list->entries)
        {
            fprintf(stderr, "Failed to allocate space for the entries of the PQ list!\n");

            exit(EXIT_FAILURE);
        }
        
        for (int i = 0; i < ivf_list->count; i++)
        {
            // Compute residual
            float *residual = (float*)malloc(dataset->dimension * sizeof(float));
            if(!residual)
            {
                fprintf(stderr, "Failed to allocate space for the residual!\n");

                exit(EXIT_FAILURE);
            }

            compute_residual(ivf_list->points[i], index->centroids[c], dataset->data_type, dataset->dimension, residual);
            
            // Split into M parts and encode
            uint16_t* pq_code = (uint16_t*)malloc(M * sizeof(uint16_t));
            if(!pq_code)
            {
                fprintf(stderr, "Failed to allocate space for the pq_code!\n");

                exit(EXIT_FAILURE);
            }

            for (int m = 0; m < M; m++)
            {
                float* subvec = &residual[m * index->pq.d_sub];
                pq_code[m] = (uint16_t)find_nearest_subspace_centroid(subvec, index->pq.subspace_centroids[m], index->pq.s, index->pq.d_sub);
            }
            
            pq_list->entries[i].point_id = ivf_list->point_ids[i];
            pq_list->entries[i].pq_code = pq_code;
            pq_list->count++;
            
            free(residual);
        }
    }
    
    // Clean up temporary IVFFlat index
    ivfflat_destroy(ivf_temp);
    
    // printf("IVFPQ index built successfully\n");
    return index;
}

void ivfpq_index_lookup(const void* q_void, const struct SearchParams* params, int* approx_neighbors, double* approx_dists, int* approx_count, void* index_data)
{
    IVFPQIndex* index = (IVFPQIndex*)index_data;
    int d = index->d;
    int k = index->k;
    int nprobe = params->nprobe;
    int N = params->N;
    
    if (nprobe > k) nprobe = k;
    
    // Step 1: Find nearest coarse centroids using a heap to keep top-nprobe
    double* centroid_dists = (double*)malloc(nprobe * sizeof(double));
    if(!centroid_dists)
    {
        fprintf(stderr, "Failed to allocate space for the centroid_dists!\n");

        exit(EXIT_FAILURE);
    }

    int* centroid_ids = (int*)malloc(nprobe * sizeof(int));
    if(!centroid_ids)
    {
        fprintf(stderr, "Failed to allocate space for the centroid_ids!\n");

        exit(EXIT_FAILURE);
    }
    
    // Convert query to float
    float* q_float = (float*)malloc(d * sizeof(float));
    if(!q_float)
    {
        fprintf(stderr, "Failed to allocate space for the q_float!\n");

        exit(EXIT_FAILURE);
    }

    if (index->data_type == DATA_TYPE_FLOAT)
        memcpy(q_float, q_void, d * sizeof(float));
    else
    {
        const uint8_t* q_u8 = (const uint8_t*)q_void;
        for (int i = 0; i < d; i++)
            q_float[i] = (float)q_u8[i];
    }
    
    // Build a heap of size nprobe with the smallest centroid distances
    MinHeap* centroid_heap = heap_create(nprobe);
    for (int i = 0; i < k; i++)
    {
        double dist = norm(q_void, index->centroids[i], d, index->data_type, DATA_TYPE_FLOAT);
        heap_insert(centroid_heap, i, dist);
    }
    // Extract in ascending order of distance (nearest first)
    heap_extract_sorted(centroid_heap, centroid_ids, centroid_dists);
    heap_destroy(centroid_heap);
    
    // Step 2: Create min-heap for top-N
    MinHeap* topN = heap_create(N);
    
    // Step 3: For each selected cluster, compute asymmetric distances
    for (int p = 0; p < nprobe; p++)
    {
        int cid = centroid_ids[p];
        IVFPQList* list = &index->lists[cid];
        
        // Compute query residual: q - c(cid)
        float* q_residual = (float*)malloc(d * sizeof(float));
        if(!q_residual)
        {
            fprintf(stderr, "Failed to allocate space for the q_residual!\n");

            exit(EXIT_FAILURE);
        }
        
        for (int i = 0; i < d; i++)
            q_residual[i] = q_float[i] - index->centroids[cid][i];
        
        // Compute approximate distance for each point using PQ codes
        for (int i = 0; i < list->count; i++)
        {
            double approx_dist = 0.0;
            uint16_t* pq_code = list->entries[i].pq_code;
            
            for (int m = 0; m < index->pq.M; m++)
            {
                float* q_sub = &q_residual[m * index->pq.d_sub];
                float dist = 0.0;

                for(int j = 0; j < index->pq.d_sub; j++)
                {
                    float diff = q_sub[j] - index->pq.subspace_centroids[m][pq_code[m]][j];
                    dist += diff * diff;

                }

                approx_dist += dist;
            }
            
            heap_insert(topN, list->entries[i].point_id, (double)approx_dist);
        }
        
        free(q_residual);
    }
    
    // Step 4: Extract results
    *approx_count = topN->size;
    int* approx_neighbors_adc = (int*)malloc((*approx_count) * sizeof(int));
    if(!approx_neighbors_adc)
    {
        fprintf(stderr, "Failed to allocate space for the approx_neighbors_adc!\n");

        exit(EXIT_FAILURE);
    }

    double* approx_dists_adc = (double*)malloc((*approx_count) * sizeof(double));
    if(!approx_dists_adc)
    {
        fprintf(stderr, "Failed to allocate space for the approx_dists_adc!\n");

        exit(EXIT_FAILURE);
    }

    heap_extract_sorted(topN, approx_neighbors_adc, approx_dists_adc);
    heap_destroy(topN);

    for (int i = 0; i < *approx_count; i++)
    {
        approx_neighbors[i] = approx_neighbors_adc[i];
        // Compute exact Euclidean distance to the selected neighbor
        void* point = index->dataset->data[approx_neighbors_adc[i]];
        approx_dists[i] = euclidean_distance(q_void, point, d, index->data_type, index->data_type);
    }

    
    // Sort aprox_dists and approx_neighbors based on approx_dists
    for (int i = 0; i < *approx_count - 1; i++)
    {
        for (int j = 0; j < *approx_count - i - 1; j++)
        {
            if (approx_dists[j] > approx_dists[j + 1])
            {
                // Swap distances
                double temp_dist = approx_dists[j];
                approx_dists[j] = approx_dists[j + 1];
                approx_dists[j + 1] = temp_dist;

                // Swap corresponding neighbors
                int temp_neighbor = approx_neighbors[j];
                approx_neighbors[j] = approx_neighbors[j + 1];
                approx_neighbors[j + 1] = temp_neighbor;
            }
        }
    }
    
    free(approx_neighbors_adc);
    free(approx_dists_adc);
    free(centroid_dists);
    free(centroid_ids);
    free(q_float);

    return;
}

void range_search_ivfpq(const void* q_void, const struct SearchParams* params, int** range_neighbors, int* range_count, void* index_data)
{
    IVFPQIndex* index = (IVFPQIndex*)index_data;
    int d = index->d;
    int k = index->k;
    int nprobe = params->nprobe;
    int R = params->R;
    
    if (nprobe > k)
        nprobe = k;
    
    // Step 1: Find nearest coarse centroids using a heap to keep top-nprobe
    double* centroid_dists = (double*)malloc(nprobe * sizeof(double));
    if(!centroid_dists)
    {
        fprintf(stderr, "Failed to allocate space for the centroid_dists!\n");

        exit(EXIT_FAILURE);
    }

    int* centroid_ids = (int*)malloc(nprobe * sizeof(int));
    if(!centroid_ids)
    {
        fprintf(stderr, "Failed to allocate space for the centroid_ids!\n");

        exit(EXIT_FAILURE);
    }

    
    // Convert query to float
    float* q_float = (float*)malloc(d * sizeof(float));
    if(!q_float)
    {
        fprintf(stderr, "Failed to allocate space for the q_float!\n");

        exit(EXIT_FAILURE);
    }

    if (index->data_type == DATA_TYPE_FLOAT)
        memcpy(q_float, q_void, d * sizeof(float));
    else
    {
        const uint8_t* q_u8 = (const uint8_t*)q_void;
        for (int i = 0; i < d; i++)
            q_float[i] = (float)q_u8[i];
    }
    
    // Build a heap of size nprobe with the smallest centroid distances
    MinHeap* centroid_heap = heap_create(nprobe);
    for (int i = 0; i < k; i++)
    {
        double dist = euclidean_distance(q_void, index->centroids[i], d, index->data_type, DATA_TYPE_FLOAT);
        heap_insert(centroid_heap, i, dist);
    }

    // Extract in ascending order of distance (nearest first)
    heap_extract_sorted(centroid_heap, centroid_ids, centroid_dists);
    heap_destroy(centroid_heap);
    
    // Range search optimization: allocate in chunks to avoid repeated realloc
    int range_capacity = 0;
    const int RANGE_ALLOC_CHUNK = 128; // Grow by 128 entries at a time

    // Step 3: For each selected cluster, compute asymmetric distances
    for (int p = 0; p < nprobe; p++)
    {
        int cid = centroid_ids[p];
        IVFPQList* list = &index->lists[cid];
        
        // Compute query residual: q - c(cid)
        float* q_residual = (float*)malloc(d * sizeof(float));
        if(!q_residual)
        {
            fprintf(stderr, "Failed to allocate space for the q_residual!\n");

            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < d; i++)
            q_residual[i] = q_float[i] - index->centroids[cid][i];
        
        // Compute approximate distance for each point using PQ codes
        for (int i = 0; i < list->count; i++)
        {
            double approx_dist = 0.0;
            uint16_t* pq_code = list->entries[i].pq_code;
            
            for (int m = 0; m < index->pq.M; m++)
            {
                float* q_sub = &q_residual[m * index->pq.d_sub];
                float dist = 0.0;
                for(int j = 0; j < index->pq.d_sub; j++)
                {
                    float diff = q_sub[j] - index->pq.subspace_centroids[m][pq_code[m]][j];
                    dist += diff * diff;

                }

                approx_dist += dist;
            }
            

            if (approx_dist <= params->R)
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
                        free(centroid_dists);
                        free(centroid_ids);
                        free(q_float);


                        exit(EXIT_FAILURE);
                    }
                    *range_neighbors = new_range;
                }
                (*range_neighbors)[(*range_count)++] = list->entries[i].point_id;
            }

        }
        
        free(q_residual);
    }
    
    // Cleanup if no range neighbors found
    if (*range_count == 0 && *range_neighbors)
    {
        free(*range_neighbors);
        *range_neighbors = NULL;
    }
    
    free(centroid_dists);
    free(centroid_ids);
    free(q_float);

    return;
}

void ivfpq_destroy(IVFPQIndex* index)
{
    if (!index)
        return;
    
    // Free subspace centroids
    if (index->pq.subspace_centroids)
    {
        for (int m = 0; m < index->pq.M; m++)
        {
            if (index->pq.subspace_centroids[m])
            {
                for (int s = 0; s < index->pq.s; s++)
                    free(index->pq.subspace_centroids[m][s]);

                free(index->pq.subspace_centroids[m]);
            }
        }
        free(index->pq.subspace_centroids);
    }
    
    // Free inverted lists
    if (index->lists)
    {
        for (int i = 0; i < index->k; i++)
        {
            if (index->lists[i].entries)
            {
                for (int j = 0; j < index->lists[i].count; j++)
                    free(index->lists[i].entries[j].pq_code);

                free(index->lists[i].entries);
            }
        }
        free(index->lists);
    }
    
    // Free centroids
    if (index->centroids)
    {
        for (int i = 0; i < index->k; i++)
            free(index->centroids[i]);

        free(index->centroids);
    }
    
    free(index);

    return;
}