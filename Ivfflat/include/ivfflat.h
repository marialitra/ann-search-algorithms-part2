#ifndef IVFFLAT_H
#define IVFFLAT_H

// Info for the centroids
typedef struct CentroidInfo
{
    bool* is_centroid; // Array to depict which data is a centroid
    float** centroids; // Centroids stored as float arrays for both float and int datasets
} centroidInfo;

// List to map each cluster (centroids to their data)
typedef struct
{
    int cluster_id; // Index of the centroid
    int count;      // Number of assigned points
    int capacity;   // Current capacity for dynamic growth
    void** points;  // Actual point vectors
    int* point_ids; // Dataset indices
} InvertedList;

// Structure to implement the IVFFLAT algorithm
typedef struct
{
    int k;               // Number of clusters
    int d;               // Dimensionality
    float** centroids;   // Centroid vectors
    DataType data_type;  // Underlying dataset type (DATA_TYPE_UINT8 or DATA_TYPE_FLOAT)
    InvertedList* lists; // One list per cluster
} IVFFlatIndex;

// Assigns points from the dataset in range [start, end) to their corresponding cluster
void assign_points_to_clusters(IVFFlatIndex* index, Dataset* dataset, int start, int end);

// Adds a point to their corresponding cluster list
static inline void add_point_to_list(InvertedList* list, void* point, int point_id, int cluster_id)
{
    if (list->count == list->capacity)
    {
        list->capacity = (list->capacity == 0) ? 128 : list->capacity * 2;
        list->points = realloc(list->points, list->capacity * sizeof(void*));
        list->point_ids = realloc(list->point_ids, list->capacity * sizeof(int));
        if (!list->points || !list->point_ids)
        {
            perror("realloc failed in add_point_to_list");
            exit(EXIT_FAILURE);
        }
    }
    
    list->points[list->count] = point;
    list->point_ids[list->count] = point_id;
    list->count++;
    list->cluster_id = cluster_id;

    return;
}

// Restarts the count for all cluster lists, starting from 0
static inline void clear_lists(IVFFlatIndex* index)
{
    for (int t = 0; t < index->k; t++)
        index->lists[t].count = 0;
        
    return;
}

// Finds the size for the subset using the square root of the datasetSize
int findSubsetSize(int datasetSize);

// Shuffles the array given
void fisher_yates_shuffle(void** array, size_t n);

// Creates subset preserving the original
Dataset* createSubset(Dataset* dataset, int subsetSize);

// Implements the Kmeans++ algorithm
centroidInfo* runKmeans(Dataset* dataset, int kclusters);

// Implements the Lloyd's algorithm (finds kclusters from subset)
IVFFlatIndex* lloydAlgorithm(Dataset* subset, int kclusters);

// Initializes the ivfflat algorithm
IVFFlatIndex* ivfflat_init(Dataset* dataset, int kclusters);

// Performs query for ivfflat
void ivfflat_index_lookup(const void* q_void, const struct SearchParams* params, int* approx_neighbors,
                        double* approx_dists, int* approx_count, void* index_data);

// Performs range search for ivfflat
void range_search_ivfflat(const void *q_void, const struct SearchParams *params,
                        int **range_neighbors, int *range_count, void *index_data);

// Destroys the IVFFlatIndex structure
void ivfflat_destroy(IVFFlatIndex* index);

#endif