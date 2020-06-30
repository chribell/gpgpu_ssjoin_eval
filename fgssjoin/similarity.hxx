#ifndef SIMILARITY_HXX
#define SIMILARITY_HXX

#pragma once

#define PMAXSIZE_EPS 1e-10

__forceinline__ __host__ __device__  int myMax(unsigned int a, unsigned int b)
{
    return (a < b) ? b : a;
}

__forceinline__ __host__ __device__  int myMin(unsigned int a, unsigned int b)
{
    return (a > b) ? b : a;
}

template <typename Similarity>
struct GenericSimilarity {
    __forceinline__ __host__ __device__ static unsigned int maxprefix(unsigned int len, double threshold) {
        return myMin(len, len - minsize(len, threshold) + 1);
    }

    __forceinline__ __host__ __device__ static unsigned int midprefix(unsigned int len, double threshold) {
        return myMin(len, len - minoverlap(len, len, threshold) + 1);
    }

    __forceinline__ __host__ __device__ static unsigned int minoverlap(unsigned int len1, unsigned int len2, double threshold) {
        return myMin(len2, myMin(len1, Similarity::minoverlap(len1, len2, threshold)));
    }

    __forceinline__ __host__ __device__ static unsigned int minsize(unsigned int len, double threshold) {
        return Similarity::minsize(len, threshold);
    }

    __forceinline__ __host__ __device__ static unsigned int maxsize(unsigned int len, double threshold) {
        return Similarity::maxsize(len, threshold);
    }

    __forceinline__ __host__ __device__ static unsigned int maxsize(unsigned int len, unsigned int pos, double threshold) {
        return Similarity::maxsize(len, pos, threshold);
    }
};

struct JaccardSimilarity {

    __forceinline__ __host__ __device__ static unsigned int minoverlap(unsigned int len1, unsigned int len2, double threshold) {
        return (unsigned int)(ceil((len1 + len2) * threshold / (1 + threshold)));
    }

    __forceinline__ __host__ __device__ static unsigned int minsize(unsigned int len, double threshold) {
        return (unsigned int)(ceil(threshold * len));
    }

    __forceinline__ __host__ __device__ static unsigned int maxsize(unsigned int len, double threshold) {
        return (unsigned int)((len / threshold));
    }

    __forceinline__ __host__ __device__ static unsigned int maxsize(unsigned int len, unsigned int pos, double threshold) {
        return (unsigned int)((len - ((1.0 - PMAXSIZE_EPS) + threshold) * pos) / threshold);
    }
};

typedef GenericSimilarity<JaccardSimilarity> Jaccard;

#endif
