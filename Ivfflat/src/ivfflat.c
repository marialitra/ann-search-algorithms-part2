#include "../include/main.h"

void assign_points_to_clusters(IVFFlatIndex* index, Dataset* dataset, int start, int end)
{
    // Two-pass parallel assignment to avoid contention on list growth
    // Pass 1: compute best cluster per point in parallel
    int n = end - start;
    if (n <= 0)
    {
        clear_lists(index);
        return;
    }

    int k = index->k;
    int d = index->d;
    DataType data_type = dataset->data_type;

    int* assign = (int*)malloc(n * sizeof(int));
    if (!assign)
    {
        perror("malloc assign");
        exit(EXIT_FAILURE);
    }

    clear_lists(index);

    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii++)
    {
        int i = start + ii;
        void *vec = dataset->data[i];
        double best_dist = DBL_MAX;
        int best_cluster = -1;

        // Find nearest centroid
        for (int t = 0; t < k; t++)
        {
            double dist = norm(vec, index->centroids[t], d, data_type, DATA_TYPE_FLOAT);
            if (dist < best_dist)
            {
                best_dist = dist;
                best_cluster = t;
            }
        }
        assign[ii] = best_cluster;
    }

    // Pass 2: count per cluster (single-thread; cheap compared to distance work)
    int* counts = (int*)calloc(k, sizeof(int));
    if (!counts)
    {
        perror("calloc counts");
        exit(EXIT_FAILURE);
    }

    for (int ii = 0; ii < n; ii++)
    {
        int c = assign[ii];
        if (c >= 0)
            counts[c]++;
    }

    // Ensure capacity and prepare write positions
    int* write_pos = (int*)malloc(k * sizeof(int));
    if (!write_pos)
    {
        perror("malloc write_pos");
        exit(EXIT_FAILURE);
    }

    for (int t = 0; t < k; t++)
    {
        InvertedList* list = &index->lists[t];
        if (list->capacity < counts[t])
        {
            list->capacity = counts[t];
            list->points = (void**)realloc(list->points, list->capacity * sizeof(void*));
            list->point_ids = (int*)realloc(list->point_ids, list->capacity * sizeof(int));
            if (!list->points || !list->point_ids)
            {
                perror("realloc in ensure capacity");

                exit(EXIT_FAILURE);
            }
        }

        list->count = 0;
        write_pos[t] = 0;
    }

    // Pass 3: fill lists in parallel using atomic capture to get slot index
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii++)
    {
        int c = assign[ii];
        int i = start + ii;
        if (c < 0)
            continue;

        int pos;
        #pragma omp atomic capture
        { pos = write_pos[c]; write_pos[c]++; }

        InvertedList *list = &index->lists[c];
        list->points[pos] = dataset->data[i];
        list->point_ids[pos] = i;
    }

    // Finalize counts
    for (int t = 0; t < k; t++)
    {
        index->lists[t].count = write_pos[t];
        index->lists[t].cluster_id = t;
    }

    // Free temporary buffers
    free(assign);
    free(counts);
    free(write_pos);

    return;
}

bool recompute_centroids(IVFFlatIndex* index, int d, double epsilon)
{
    int k = index->k;
    int changed_any = 0;

    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < k; t++)
    {
        InvertedList* list = &index->lists[t];
        if (list->count == 0)
            continue;

        float* new_centroid = (float*)calloc(d, sizeof(float));
        if (!new_centroid)
        {
            perror("calloc failed in recompute_centroids");

            exit(EXIT_FAILURE);
        }

        // Sum points for this cluster
        for (int i = 0; i < list->count; i++)
        {
            if (index->data_type == DATA_TYPE_FLOAT)
            {
                float* vec = (float*)list->points[i];
                for (int j = 0; j < d; j++)
                    new_centroid[j] += vec[j];
            }
            else
            {
                uint8_t* u8vec = (uint8_t*)list->points[i];
                for (int j = 0; j < d; j++)
                    new_centroid[j] += (float)u8vec[j];
            }
        }

        // Average
        for (int j = 0; j < d; j++)
            new_centroid[j] /= (list->count > 0 ? list->count : 1);

        // Check centroid shift
        double shift = euclidean_distance(index->centroids[t], new_centroid, d, DATA_TYPE_FLOAT, DATA_TYPE_FLOAT);
        if (shift > epsilon)
        {
            #pragma omp atomic write
            changed_any = 1;
        }

        // Replace old centroid
        float* old = index->centroids[t];
        index->centroids[t] = new_centroid;
        free(old);
    }

    return changed_any != 0;
}

int findSubsetSize(int datasetSize)
{
    int size = sqrt(datasetSize);

    return size;
}

void fisher_yates_shuffle(void** array, size_t n)
{
    for (size_t i = n - 1; i > 0; i--)
    {
        size_t j = rand() % (i + 1);
        // Swap the pointers
        void* tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }

    return;
}

Dataset* createSubset(Dataset* dataset, int subsetSize)
{
    // Create a copy of the data pointers to shuffle without modifying the original dataset
    void** data_copy = (void**)malloc(dataset->size * sizeof(void*));
    if (!data_copy)
    {
        fprintf(stderr, "Memory allocation failed for data_copy in createSubset\n");

        exit(EXIT_FAILURE);
    }
    
    // Copy the pointers (not the data itself)
    for (size_t i = 0; i < dataset->size; i++)
    {
        data_copy[i] = (void*)malloc(dataset->dimension * sizeof(float));
        if (!data_copy[i])
        {
            fprintf(stderr, "Memory allocation failed for data_copy[%zu] in createSubset\n", i);
            // Free previously allocated pointers
            for (size_t m = 0; m < i; m++)
                free(data_copy[m]);

            free(data_copy);

            exit(EXIT_FAILURE);
        }
        if (dataset->data_type == DATA_TYPE_FLOAT)
        {
            float* src = (float*)dataset->data[i];
            float* dst = (float*)data_copy[i];
            memcpy(dst, src, dataset->dimension * sizeof(float));
        }
        else
        {
            uint8_t* src = (uint8_t*)dataset->data[i];
            float* dst = (float*)data_copy[i];
            for (int j = 0; j < dataset->dimension; j++)
                dst[j] = (float)src[j];
        }
    }
    
    // Shuffle the copy, not the original
    fisher_yates_shuffle(data_copy, dataset->size);

    Dataset* subset = (Dataset*)malloc(sizeof(Dataset));
    if (!subset)
    {
        fprintf(stderr, "Memory allocation failed for subset dataset struct in Kmeans++\n");
        free(data_copy);

        exit(EXIT_FAILURE);
    }

    subset->size = subsetSize;
    subset->dimension = dataset->dimension;
    subset->data_type = dataset->data_type;
    subset->data = data_copy;  // Use the shuffled copy

    return subset;
}

void printClusters(float** centroids, int kclusters, int d)
{
    for (int dx = 0; dx < kclusters; dx++)
    {
        printf("centroid[%d] = { ", dx);
        for (int x = 0; x < d; x++)
        {
            printf("%f, ", centroids[dx][x]);
        }
        printf("}\n");
    }

    return;
}

centroidInfo* runKmeans(Dataset* subset, int kclusters)
{
    int n = subset->size;
    // printf("From the subset size: %d we want to find %d centroids!\n", n, kclusters);
    int d = subset->dimension;
    int t = 0; // Count of centroids
    int i = 0; // Count of non-centroids
    double distance = 0.0;
    int idx = 0;
    double sumDistances = 0.0;
    double chooseRandomNumber = 0.0;
    double density = 0.0;

    centroidInfo* info = (centroidInfo*)malloc(sizeof(centroidInfo));
    if (!info)
    {
        fprintf(stderr, "Memory allocation failed for centroid info\n");

        exit(EXIT_FAILURE);
    }

    bool* is_centroid = (bool*)calloc(n, sizeof(bool));
    if (!is_centroid)
    {
        fprintf(stderr, "Memory allocation failed for is_centroid\n");

        exit(EXIT_FAILURE);
    }

    double* best_distances_square = (double*)calloc(n, sizeof(double));
    if (!best_distances_square)
    {
        fprintf(stderr, "Memory allocation failed for best_distances_square\n");

        exit(EXIT_FAILURE);
    }

    double* probabilities = (double*)calloc(n, sizeof(double));
    if (!probabilities)
    {
        fprintf(stderr, "Memory allocation failed for probabilities\n");

        exit(EXIT_FAILURE);
    }

    /* Unified initialization: handle kclusters > n, then KMeans++ selection using
     * Distance point to centroid for both float and int subset types. */
    if (kclusters > n)
    {
        float** centroids = malloc(n * sizeof(float*));
        if (!centroids)
        {
            fprintf(stderr, "Memory allocation failed for centroids\n");

            exit(EXIT_FAILURE);
        }

        for (int l = 0; l < n; l++)
        {
            centroids[l] = (float*)malloc(d * sizeof(float));
            if (!centroids[l])
            {
                fprintf(stderr, "Memory allocation failed for centroid copy\n");

                exit(EXIT_FAILURE);
            }

            if (subset->data_type == DATA_TYPE_FLOAT)
            {
                float* src = (float*)subset->data[l];
                for (int _j = 0; _j < d; ++_j)
                    centroids[l][_j] = src[_j];
            }
            else
            {
                uint8_t* src = (uint8_t*)subset->data[l];
                for (int _j = 0; _j < d; ++_j)
                    centroids[l][_j] = (float)src[_j];
            }

            is_centroid[l] = 1;
        }
        info->centroids = centroids;
        info->is_centroid = is_centroid;

        return info;
    }

    float** centroids = malloc(kclusters * sizeof(float*));
    if (!centroids)
    {
        fprintf(stderr, "Memory allocation failed for centroids\n");

        exit(EXIT_FAILURE);
    }

    idx = rand() % n;
    centroids[t] = (float*)malloc(d * sizeof(float));
    
    if (!centroids[t])
    {
        fprintf(stderr, "Memory allocation failed for centroid copy\n");

        exit(EXIT_FAILURE);
    }

    if (subset->data_type == DATA_TYPE_FLOAT)
    {
        float *src = (float *)subset->data[idx];
        for (int _j = 0; _j < d; ++_j)
            centroids[t][_j] = src[_j];
    }
    else
    {
        uint8_t *src = (uint8_t *)subset->data[idx];
        for (int _j = 0; _j < d; ++_j)
            centroids[t][_j] = (float)src[_j];
    }

    is_centroid[idx] = 1;
    t++;
    for (; t < kclusters; t++)
    {
        for (int b = 0; b < n; b++)
            best_distances_square[b] = -1.0;

        sumDistances = 0.0;

        for (int ii = 0; ii < n; ii++)
        {
            if (is_centroid[ii])
                continue;

            void* vec = subset->data[ii];
            double best_dist = DBL_MAX;
            for (int l = 0; l < t; l++)
            {
                double dist = norm(vec, centroids[l], d, subset->data_type, DATA_TYPE_FLOAT);
                if (dist < best_dist)
                    best_dist = dist;
            }
            best_distances_square[ii] = best_dist * best_dist;
            sumDistances += best_distances_square[ii];
        }

        for (int ii = 0; ii < n; ii++)
        {
            if (is_centroid[ii])
                continue;

            probabilities[ii] = (sumDistances > 0.0) ? (best_distances_square[ii] / sumDistances) : 0.0;
        }

        chooseRandomNumber = (double)rand() / (double)RAND_MAX;
        density = 0.0;
        for (int ii = 0; ii < n; ii++)
        {
            if (is_centroid[ii])
                continue;

            density += probabilities[ii];
            if (density >= chooseRandomNumber)
            {
                centroids[t] = (float*)malloc(d * sizeof(float));
                if (!centroids[t])
                {
                    fprintf(stderr, "Memory allocation failed for centroid copy\n");

                    exit(EXIT_FAILURE);
                }

                if (subset->data_type == DATA_TYPE_FLOAT)
                {
                    float *src = (float *)subset->data[ii];
                    for (int _j = 0; _j < d; ++_j)
                        centroids[t][_j] = src[_j];
                }
                else
                {
                    uint8_t *src = (uint8_t *)subset->data[ii];
                    for (int _j = 0; _j < d; ++_j)
                        centroids[t][_j] = (float)src[_j];
                }

                is_centroid[ii] = 1;
                break;
            }
        }
    }

    info->centroids = centroids;
    info->is_centroid = is_centroid;

    // Free temporary working buffers allocated in this function
    free(probabilities);
    free(best_distances_square);

    return info;
}

IVFFlatIndex* lloydAlgorithm(Dataset* subset, int kclusters)
{
    centroidInfo* info = runKmeans(subset, kclusters);
    int max_iters = 50;
    double epsilon = 1e-4;

    // Take care of the edge case
    if (kclusters > subset->size)
        kclusters = subset->size;

    // Create IVFFlat index structure
    IVFFlatIndex* index = malloc(sizeof(IVFFlatIndex));
    index->k = kclusters;
    index->d = subset->dimension;
    index->centroids = info->centroids;
    index->data_type = subset->data_type;
    index->lists = calloc(kclusters, sizeof(InvertedList));
    if (!index->lists)
    {
        perror("calloc lists");

        exit(EXIT_FAILURE);
    }

    for (int t = 0; t < kclusters; ++t)
    {
        index->lists[t].points = NULL;
        index->lists[t].point_ids = NULL;
        index->lists[t].count = 0;
        index->lists[t].capacity = 0;
        index->lists[t].cluster_id = t;
    }

    for (int iter = 0; iter < max_iters; iter++)
    {
        assign_points_to_clusters(index, subset, 0, subset->size);
        bool changed = recompute_centroids(index, subset->dimension, epsilon);
        if (!changed)
            break;
    }

    // Clean up temporary memory we no longer need
    free(info->is_centroid);
    free(info);

    // Return built index (centroids and lists)
    return index;
}

IVFFlatIndex* ivfflat_init(Dataset* dataset, int kclusters)
{
    int subsetSize = findSubsetSize(dataset->size);
    Dataset* subset = createSubset(dataset, subsetSize); // Produces the X'

    IVFFlatIndex* ivfflat_index = lloydAlgorithm(subset, kclusters);


    // Now assign ALL points from the full dataset to the corresponding centroids
    assign_points_to_clusters(ivfflat_index, dataset, 0, dataset->size);
    
    // Free data
    for (int i = 0; i < dataset->size; i++)
    {
        free(subset->data[i]);
    }

    free(subset->data);
    free(subset);

    return ivfflat_index;
}

void ivfflat_index_lookup(const void* q_void, const struct SearchParams* params, int* approx_neighbors, double* approx_dists, int* approx_count, void* index_data)
{
    IVFFlatIndex* index = (IVFFlatIndex*)index_data;
    int d = index->d;
    int k = index->k;
    int nprobe = params->nprobe;
    int N = params->N; // Number of neighbors to return

    // --- Step 1: Compute distances from query to all centroids ---
    if (nprobe > k)
        nprobe = k;
        
    double* centroid_dists = malloc(nprobe * sizeof(double));
    int* centroid_ids = malloc(nprobe * sizeof(int));
    int selected = 0;

  
    const float* qf = NULL;
    const uint8_t* qi = NULL;
    if (index->data_type == DATA_TYPE_FLOAT)
        qf = (const float*)q_void;
    else
        qi = (const uint8_t*)q_void;

    for (int i = 0; i < k; i++)
    {
        double cent;
        if (qf)
            cent = norm(qf, index->centroids[i], d, DATA_TYPE_FLOAT, DATA_TYPE_FLOAT);
        else
            cent = norm(qi, index->centroids[i], d, DATA_TYPE_UINT8, DATA_TYPE_FLOAT);

        int j = 0;
        if (selected < nprobe)
        {
            j = selected;
            selected++;
        }
        else
        {
            if (cent >= centroid_dists[nprobe - 1])
                continue;

            j = nprobe - 1;
        }

        for (; j > 0 && cent < centroid_dists[j - 1]; j--)
        {
            centroid_dists[j] = centroid_dists[j - 1];
            centroid_ids[j] = centroid_ids[j - 1];
        }
        centroid_dists[j] = cent;
        centroid_ids[j] = i;
    }

    // --- Step 2: Create min-heap for top-N candidates ---
    MinHeap* topN = heap_create(N);

    // --- Step 3: Search within selected (nprobe) clusters ---
    int total_candidates = 0;
    for (int p = 0; p < nprobe && p < k; p++)
    {
        int cid = centroid_ids[p];
        InvertedList* list = &index->lists[cid];

        for (int i = 0; i < list->count; i++)
        {
            void* vec = list->points[i];
            double dist;

            if (index->data_type == DATA_TYPE_FLOAT)
                dist = euclidean_distance(qf, vec, d, DATA_TYPE_FLOAT, DATA_TYPE_FLOAT);
            else
                dist = euclidean_distance(qi, vec, d, DATA_TYPE_UINT8, DATA_TYPE_UINT8);

            total_candidates++;

            // Insert into min-heap (O(log N) instead of O(N) insertion sort)
            heap_insert(topN, list->point_ids[i], dist);
        }
    }

    // --- Step 4: Extract results from heap in sorted order ---
    *approx_count = topN->size;
    heap_extract_sorted(topN, approx_neighbors, approx_dists);
    heap_destroy(topN);

    // --- Cleanup ---
    free(centroid_dists);
    free(centroid_ids);

    return;
}

void range_search_ivfflat(const void* q_void, const struct SearchParams* params, int** range_neighbors, int* range_count, void* index_data)
{
    IVFFlatIndex* index = (IVFFlatIndex*)index_data;
    int d = index->d;
    int k = index->k;
    int nprobe = params->nprobe;
    int R = params->R; // Range distance for range search

    // --- Step 1: Compute distances from query to all centroids ---
    if (nprobe > k)
        nprobe = k;

    double* centroid_dists = malloc(nprobe * sizeof(double));
    int* centroid_ids = malloc(nprobe * sizeof(int));
    int selected = 0;

  
    const float* qf = NULL;
    const uint8_t* qi = NULL;
    if (index->data_type == DATA_TYPE_FLOAT)
        qf = (const float*)q_void;
    else
        qi = (const uint8_t*)q_void;

    for (int i = 0; i < k; i++)
    {
        double cent;
        if (qf)
            cent = norm(qf, index->centroids[i], d, DATA_TYPE_FLOAT, DATA_TYPE_FLOAT);
        else
            cent = norm(qi, index->centroids[i], d, DATA_TYPE_UINT8, DATA_TYPE_FLOAT);

        int j = 0;
        if (selected < nprobe)
        {
            j = selected;
            selected++;
        }
        else
        {
            if (cent >= centroid_dists[nprobe - 1])
                continue;

            j = nprobe - 1;
        }

        for (; j > 0 && cent < centroid_dists[j - 1]; j--)
        {
            centroid_dists[j] = centroid_dists[j - 1];
            centroid_ids[j] = centroid_ids[j - 1];
        }
        centroid_dists[j] = cent;
        centroid_ids[j] = i;
    }

    // Range search optimization: allocate in chunks to avoid repeated realloc
    int range_capacity = 0;
    const int RANGE_ALLOC_CHUNK = 128; // Grow by 128 entries at a time

    // --- Step 3: Search within selected (nprobe) clusters ---
    int total_candidates = 0;
    for (int p = 0; p < nprobe && p < k; p++)
    {
        int cid = centroid_ids[p];
        InvertedList* list = &index->lists[cid];

        for (int i = 0; i < list->count; i++)
        {
            void* vec = list->points[i];
            double dist;

            if (index->data_type == DATA_TYPE_FLOAT)
                dist = euclidean_distance(qf, vec, d, DATA_TYPE_FLOAT, DATA_TYPE_FLOAT);
            else
                dist = euclidean_distance(qi, vec, d, DATA_TYPE_UINT8, DATA_TYPE_UINT8);

            total_candidates++;

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
                        free(centroid_dists);
                        free(centroid_ids);         

                        exit(EXIT_FAILURE);
                    }
                    *range_neighbors = new_range;
                }
                (*range_neighbors)[(*range_count)++] = list->point_ids[i];
            }

        }
    }

    // Cleanup if no range neighbors found
    if (*range_count == 0 && *range_neighbors)
    {
        free(*range_neighbors);
        *range_neighbors = NULL;
    }

    // Cleanup
    free(centroid_dists);
    free(centroid_ids);

    return;
}

void ivfflat_destroy(IVFFlatIndex* index)
{
    if (!index)
        return;

    // Free all lists (only point_ids and list arrays; the points belong to the dataset)
    for (int t = 0; t < index->k; ++t)
    {
        if (index->lists[t].points)
            free(index->lists[t].points);
        if (index->lists[t].point_ids)
            free(index->lists[t].point_ids);
    }

    free(index->lists);

    // Free centroids
    if (index->centroids)
    {
        for (int t = 0; t < index->k; ++t)
            if (index->centroids[t])
                free(index->centroids[t]);
        free(index->centroids);
    }

    free(index);

    return;
}
