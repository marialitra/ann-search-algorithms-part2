#ifndef HASHTABLE_H
#define HASHTABLE_H

// Forward Declarations
struct LSH; 
typedef struct hash_table* HashTable;

// Function Pointer Types
typedef void (*funtion)(void*);
typedef int (*Compare_fun)(const void*, const void*, const void*);
typedef int (*Hash_fun)(HashTable, void*, uint64_t*);

// Structure to store an entry in the bucket
typedef struct HTEntry
{
    void* key;      // The key, representing the dataset index
    void* data;     // Pointer to vector data
    uint64_t ID;    // ID produced by hash function
} HTEntry;

// Internal bucket representation
typedef struct HTBucket
{
    HTEntry* items; // Array of items (entries) inside of a bucket
    int count;      // Currently stored number of entries
    int capacity;   // Capacity of bucket
} HTBucket;

// Structure for Hashtable Implementation
// Notice that the way it handles collisions is by chaining with dynamin arrays,
// So it has fixed capacity
typedef struct hash_table
{
    int size;                   // Number of elements in the table
    int capacity;               // Number of buckets
    int key_size;               // Size of the key in bytes (can be any type)
    funtion destroy;            // Function to destroy elements of the table
    Compare_fun compare;        // Function to compare keys
    Hash_fun hash_function;     // Function to hash data to buckets

    void* algorithmContext;     // Pointer to algorithm context for hashing (LSH*, Hypercube*, ...)
    int table_index;            // Index of this table in a family only for LSH (the others have 0)
    const void* metricContext;  // Stores the dimension of the element or any metric context

    HTBucket* buckets;          // Array of buckets implemented as dynamic arrays
}hash_table;

// Finds the prime number before a given number, n
int nearest_prime(int n);

// Creates a hash table with a given capacity
HashTable hash_table_create(int capacity, int key_size, funtion destroy, Compare_fun compare, Hash_fun hash_function,
                            void* algorithmContext, int table_index, const void* metricContext);

// Inserts a key and data in the hash table
int hash_table_insert(HashTable hash_table, void* key, void* data);

// Searches for a key in the hash table and returns the associated data
void* hash_table_search(HashTable hash_table, void* key);

// Destroys the hash table
void hash_table_destroy(HashTable hash_table);

// Returns the number of elements in the hash table
int hash_table_size(HashTable hash_table);

// Removes the element with the given key from the hash table
int hash_table_remove(HashTable hash_table, void* key);

// Returns the capacity of the hash table
int hash_table_capacity(HashTable hash_table); 

// Returns a pointer to the contiguous entries array for a bucket and sets out_count
const HTEntry* hash_table_get_bucket_entries(HashTable hash_table, uint64_t index, int* out_count);

// Print the hashtable given
void print_hashtable(HashTable hash_table, int table_size, int dimension);

// Print all hashtables
void print_hashtables(int L, int table_size, HashTable* hash_tables, int dimension);

// Retrieve hash table's algorithm context
static inline void* hash_table_get_algorithm_context(HashTable ht)
{
    return ((struct hash_table*)ht)->algorithmContext;
}

// Retrieve hash table's index
static inline int hash_table_get_index(HashTable ht)
{
    return ((struct hash_table*)ht)->table_index;
}

#endif