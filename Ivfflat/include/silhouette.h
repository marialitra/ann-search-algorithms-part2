#ifndef SILHOUETTE_H
#define SILHOUETTE_H

// Computes the silhouette scores for each cluster given the IVFFlat index
void computeSilhouette(IVFFlatIndex* index, Dataset* dataset);

#endif