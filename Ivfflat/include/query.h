#ifndef QUERY_H
#define QUERY_H

// Function pointer type for index lookup functions
typedef void (*index_lookup)(const void*, const struct SearchParams*, int*, double*, int*, void*);

// Function pointer type for index lookup functions
typedef void (*range_search)(const void*, const struct SearchParams*, int**, int*, void*);

// Performs queries on the query set using the provided index lookup function
void perform_query(const struct SearchParams* params, const struct Dataset* dataset,
                const struct Dataset* query_set, index_lookup lookup_func, range_search range_fun, void* index_data);

#endif