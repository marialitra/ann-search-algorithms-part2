#ifndef RUNALGORITHMS_H
#define RUNALGORITHMS_H

struct SearchParams;

// Function declaration for running LSH
void run_lsh(SearchParams* params, Dataset* dataset);

// Function declaration for running Hypercube
void run_hypercube(SearchParams* params, Dataset* dataset);

// Function declaration for running IVFFLAT
void run_ivfflat(SearchParams* params, Dataset* dataset);

// Function declaration for running IVFPQ
void run_ivfpq(SearchParams* params, Dataset* dataset);

#endif