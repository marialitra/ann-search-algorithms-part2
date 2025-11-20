#include "../include/main.h"

int nearest_prime(int n)
{
    if(n < 2)
        return 2;

    int* prime = (int*)malloc((n + 1) * sizeof(int));
    if (!prime)
        return 2; // Fallback

    for (int i = 0; i <= n; i++)
        prime[i] = 1;

    for (int i = 2; i * i <= n; i++)
    {
        if (prime[i] == 1)
        {
            for (int j = i * i; j <= n; j += i)
            {
                prime[j] = 0;
            }
        }
    }

    // Return the nearest prime number
    for (int i = n; i >= 2; i--)
    {
        if (prime[i] == 1)
        {
            free(prime);

            return i;
        }
    }

    free(prime);

    return 2;
}

HashTable hash_table_create(int capacity, int key_size, funtion destroy, Compare_fun compare, Hash_fun hash_function, void* algorithmContext, int table_index, const void* metricContext)
{
    // Initialize the hash table structure
    HashTable hash_table = (HashTable)malloc(sizeof(struct hash_table));
    if (hash_table == NULL)
    {
        fprintf(stderr, "Failed to allocate hashtable\n");

        exit(EXIT_FAILURE);
    }

    hash_table->size = 0;
    hash_table->capacity = capacity;
    hash_table->key_size = key_size;
    hash_table->destroy = destroy;
    hash_table->compare = compare;
    hash_table->hash_function = hash_function;
    hash_table->algorithmContext = algorithmContext;
    hash_table->table_index = table_index;
    hash_table->metricContext = metricContext;

    // Initialize buckets array
    hash_table->buckets = (HTBucket*)calloc(hash_table->capacity, sizeof(HTBucket));
    if (hash_table->buckets == NULL)
    {
        fprintf(stderr, "Failed to allocate hashtable buckets\n");

        exit(EXIT_FAILURE);
    }

    return hash_table;
}

int hash_table_insert(HashTable hash_table, void* key, void* data)
{
    // Get the ID value
    uint64_t ID = 0ULL;

    // Get the hash value
    int hash_value = hash_table->hash_function(hash_table, data, &ID);

    // Ensure capacity in the target bucket
    HTBucket* b = &hash_table->buckets[hash_value];
    if (b->capacity == 0)
    {
        b->capacity = 4;
        b->items = (HTEntry*)malloc(b->capacity * sizeof(HTEntry));
        if (!b->items)
        {
            fprintf(stderr, "Failed to allocate buket items\n");

            exit(EXIT_FAILURE);
        }
    }
    else if (b->count >= b->capacity)
    {
        int new_cap = b->capacity * 2;
        HTEntry* new_items = (HTEntry*)realloc(b->items, new_cap * sizeof(HTEntry));
        if (!new_items)
        {
            fprintf(stderr, "Failed to allocate space for new items\n");

            exit(EXIT_FAILURE);
        }

        b->items = new_items;
        b->capacity = new_cap;
    }

    // Append entry at end
    HTEntry* e = &b->items[b->count++];
    e->data = data;
    e->key = (void*)malloc(hash_table->key_size);
    if (!e->key)
    {
        fprintf(stderr, "Failed to allocate space for key\n");

        exit(EXIT_FAILURE);
    }

    memcpy(e->key, key, hash_table->key_size);
    e->ID = ID;
    hash_table->size++;

    return 0;
}

void* hash_table_search(HashTable hash_table, void* key)
{
    uint64_t ID = 0ULL;

    // Get the hash value
    int hash_value = hash_table->hash_function(hash_table, key, &ID);

    HTBucket* b = &hash_table->buckets[hash_value];
    if (!b || b->count == 0) return NULL;

    for (int i = 0; i < b->count; ++i)
    {
        if (hash_table->compare && hash_table->compare(b->items[i].key, key, hash_table->metricContext) == 0)
            return b->items[i].data;
    }

    return NULL;
}

void hash_table_destroy(HashTable hash_table)
{
    // Destroy the table and the elements in it
    for (int i = 0; i < hash_table->capacity; i++)
    {
        HTBucket* b = &hash_table->buckets[i];
        if (b->items)
        {
            for (int j = 0; j < b->count; ++j)
            {
                if (hash_table->destroy)
                    hash_table->destroy(b->items[j].data);

                free(b->items[j].key);
            }

            free(b->items);
            b->items = NULL;
            b->count = b->capacity = 0;
        }
    }

    free(hash_table->buckets);
    free(hash_table);

    return;
}

int hash_table_size(HashTable hash_table)
{
    return hash_table->size;
}

int hash_table_remove(HashTable hash_table, void *key)
{
    uint64_t ID = 0ULL;

    // Get the hash value
    int hash_value = hash_table->hash_function(hash_table, key, &ID);

    HTBucket* b = &hash_table->buckets[hash_value];
    if (!b || b->count == 0) return -1;

    for (int i = 0; i < b->count; ++i)
    {
        if (hash_table->compare && hash_table->compare(b->items[i].key, key, hash_table->metricContext) == 0)
        {
            // Remove by shifting last into i (order not preserved) or memmove
            free(b->items[i].key);
            if (hash_table->destroy)
                hash_table->destroy(b->items[i].data);

            // Shift tail left
            if (i < b->count - 1)
                memmove(&b->items[i], &b->items[i+1], (b->count - i - 1) * sizeof(HTEntry));

            b->count--;
            hash_table->size--;

            return 0;
        }
    }

    return -1;
}

int hash_table_capacity(HashTable hash_table)
{
    return hash_table->capacity;
}

const HTEntry* hash_table_get_bucket_entries(HashTable hash_table, uint64_t index, int* out_count)
{
    if (index >= (uint64_t)hash_table->capacity)
    {
        if (out_count)
            *out_count = 0;
        
            return NULL;
    }

    HTBucket* b = &hash_table->buckets[index];
    if (out_count)
        *out_count = b->count;

    return b->items;
}

void print_hashtable(HashTable hash_table, int table_size, int dimension)
{
    if (hash_table == NULL)
    {
        printf("Hash table is not initialized.\n");

        return;
    }

    // Print the contents of the hash table
    for (int j = 0; j < table_size; j++)
    {
        int cnt = 0;
        const HTEntry* bucket = hash_table_get_bucket_entries(hash_table, j, &cnt);

        if (bucket != NULL && cnt > 0)
        {
            printf(" Bucket %d: ", j);

            for (int bi = 0; bi < cnt; ++bi)
            {
                if (bucket[bi].data == NULL)
                    printf("(NULL_DATA) -> ");
                else
                {
                    float* point = (float*)bucket[bi].data;

                    // Check if we can safely read the point
                    printf("(");

                    for (int d = 0; d < dimension; d++)
                    {
                        printf("%f", point[d]);

                        if (d < dimension - 1)
                            printf(", ");

                    }

                    printf(") -> ");
                }
            }

            printf("NULL\n");
        }
        else
            printf(" Bucket %d: NULL\n", j);

    }

    printf("----------------------------------------------------------------\n");

    return;
}

void print_hashtables(int L, int table_size, HashTable* hash_tables, int dimension)
{
    if (hash_tables == NULL)
    {
        printf("Hashtables are not initialized.\n");
        return;
    }

    // Print the contents of each hash table
    for (int i = 0; i < L; i++)
    {
        printf("Hash table %d:\n", i);
        print_hashtable(hash_tables[i], table_size, dimension);

        printf("----------------------------------------------------------------\n");
    }

    return;
}