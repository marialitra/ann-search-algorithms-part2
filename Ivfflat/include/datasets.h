#ifndef DATASETS_H
#define DATASETS_H

// Data type enum to distinguish between int and float datasets
typedef enum
{
    DATA_TYPE_UINT8,  // MNIST: uint8_t values 0-255
    DATA_TYPE_FLOAT   // SIFT: float32 values
} DataType;

// Definition of Dataset struct
typedef struct Dataset
{
    void** data;         // Stores the data of the dataset
    int size;            // Number of points inside of the dataset
    int dimension;       // The dimension of the points
    DataType data_type;  // Type of data stored (DATA_TYPE_UINT8 or DATA_TYPE_FLOAT)
} Dataset;

// Reads MNIST IDX3 images file (big-endian) and returns a Dataset of integers [0..255]
Dataset* read_data_mnist(const char* images_path);

// Reads SIFT IDX3 image descriptor file (little-endian) and returns a Dataset of floats
Dataset* read_data_sift(const char* images_path);

// Prints first 'size' points of the dataset.
void printPartialDataset(int size, const Dataset* dataset);

// Frees all memory associated with a Dataset
void free_dataset(Dataset* dataset);

#endif