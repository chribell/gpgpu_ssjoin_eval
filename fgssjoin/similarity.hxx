#ifndef FSSJOIN_SIMILARITY_HXX
#define FSSJOIN_SIMILARITY_HXX

#define PMAXSIZE_EPS 1e-10

#include "utils.hxx"

__forceinline__ __host__ __device__ unsigned int jaccard_maxprefix(unsigned int len, double threshold)
{
    unsigned int minsize = (unsigned int)(ceil(threshold * len));
    return myMin(len, len - minsize + 1);
}

__forceinline__ __host__ __device__ unsigned int jaccard_minoverlap(unsigned int len1, unsigned int len2, double threshold)
{
    unsigned int overlap = (unsigned int)(ceil((len1 + len2) * threshold / (1 + threshold)));
    return myMin(len2, myMin(len1, overlap));
}

__forceinline__ __host__ __device__ unsigned int jaccard_minsize(unsigned int len, double threshold)
{
    return (unsigned int)(ceil(threshold * len));
}

__forceinline__ __host__ __device__  unsigned int jaccard_maxsize(unsigned int len, double threshold)
{
    return (unsigned int)((len / threshold));
}

__forceinline__ __host__ __device__  unsigned int jaccard_maxsize(unsigned int len, unsigned int pos, double threshold)
{
    return (unsigned int)((len - ((1.0 - PMAXSIZE_EPS) + threshold) * pos) / threshold);
}

__forceinline__ __host__ __device__  unsigned int jaccard_midprefix(unsigned int len, double threshold)
{
    unsigned int minoverlap = jaccard_minoverlap(len, len, threshold);
    return myMin(len, len - minoverlap + 1);
}

#endif //BITMAP_GPUSSJOIN_SIMILARITY_HXX