#ifndef MINHEAP_H
#define MINHEAP_H

// Neighbor structure for min-heap (max-heap for top-N)
typedef struct
{
    int id;         // ID of the neighbor
    double dist;    // Distance from point
} Neighbor;

// MinHeap structure for neighbors
typedef struct
{
    Neighbor* heap;  // Array of neighbors
    int size;        // Size of heap
    int capacity;    // Capacity of heap
} MinHeap;

// Create a new min-heap with given capacity
MinHeap* heap_create(int capacity);

// Insert a neighbor into the heap
// If heap is full and dist < max_dist in heap, replace root
void heap_insert(MinHeap* h, int id, double dist);

// Extract all elements in sorted order (ascending by distance)
// Results are written to ids[] and dists[] arrays
void heap_extract_sorted(MinHeap* h, int* ids, double* dists);

// Destroy the heap and free memory
void heap_destroy(MinHeap* h);

// Get the maximum distance in the heap (root element)
// Returns -1.0 if heap is empty
double heap_max_dist(const MinHeap* h);

// Check if heap is full
bool heap_is_full(const MinHeap* h);

#endif