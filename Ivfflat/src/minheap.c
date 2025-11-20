#include "../include/main.h"

// Helper function to swap elements (Neighbors) inside of the heap
static inline void heap_swap(Neighbor* a, Neighbor* b)
{
    Neighbor temp = *a;
    *a = *b;
    *b = temp;

    return;
}

// Helper function to sift down in heap
static inline void heap_sift_down(MinHeap* h, int idx)
{
    int largest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    
    if (left < h->size && h->heap[left].dist > h->heap[largest].dist)
        largest = left;
    if (right < h->size && h->heap[right].dist > h->heap[largest].dist)
        largest = right;
    
    if (largest != idx)
    {
        heap_swap(&h->heap[idx], &h->heap[largest]);
        heap_sift_down(h, largest);
    }

    return;
}

// Helper function to sift up in heap
static inline void heap_sift_up(MinHeap* h, int idx)
{
    while (idx > 0)
    {
        int parent = (idx - 1) / 2;
        if (h->heap[idx].dist <= h->heap[parent].dist)
            break;

        heap_swap(&h->heap[idx], &h->heap[parent]);
        idx = parent;
    }

    return;
}

MinHeap* heap_create(int capacity)
{
    MinHeap* h = malloc(sizeof(MinHeap));
    if (!h)
    {
        fprintf(stderr, "Failed to allocate heap!\n");

        exit(EXIT_FAILURE);
    }
    
    h->heap = malloc(capacity * sizeof(Neighbor));
    if (!h->heap)
    {
        fprintf(stderr, "Failed to allocate heap!\n");
        free(h);

        exit(EXIT_FAILURE);
    }
    
    h->size = 0;
    h->capacity = capacity;
    
    return h;
}

void heap_insert(MinHeap* h, int id, double dist)
{
    if (h->size < h->capacity)
    {
        // Heap not full - add element
        h->heap[h->size].id = id;
        h->heap[h->size].dist = dist;
        heap_sift_up(h, h->size);
        h->size++;
    }
    else if (dist < h->heap[0].dist)
    {
        // Replace root (largest distance) with new element
        h->heap[0].id = id;
        h->heap[0].dist = dist;
        heap_sift_down(h, 0);
    }

    return;
}

void heap_extract_sorted(MinHeap* h, int* ids, double* dists) 
{
    // Extract elements in descending order and reverse to get ascending
    int n = h->size;
    for (int i = n - 1; i >= 0; i--) 
    {
        ids[i] = h->heap[0].id;
        dists[i] = h->heap[0].dist;
        h->heap[0] = h->heap[h->size - 1];
        h->size--;
        heap_sift_down(h, 0);
    }

    return;
}

void heap_destroy(MinHeap* h)
{
    if (!h)
        return;

    free(h->heap);
    free(h);

    return;
}

double heap_max_dist(const MinHeap* h)
{
    if (!h || h->size == 0)
        return -1.0;

    return h->heap[0].dist;
}

bool heap_is_full(const MinHeap* h)
{
    if (!h)
        return false;

    return h->size >= h->capacity;
}