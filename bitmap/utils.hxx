#ifndef BITMAP_GPUSSJOIN_UTILS_HXX
#define BITMAP_GPUSSJOIN_UTILS_HXX

inline void __gpuAssert(cudaError_t stat, int line, std::string file) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "Error %s at line %d in file %s\n",
                cudaGetErrorString(stat), line, file.c_str());
        exit(1);
    }
}

#define gpuAssert(value)  __gpuAssert((value),(__LINE__),(__FILE__))

__forceinline__ __host__ __device__  int myMax(int a, int b)
{
    return (a < b) ? b : a;
}

__forceinline__ __host__ __device__  int myMin(int a, int b)
{
    return (a > b) ? b : a;
}


#endif //BITMAP_GPUSSJOIN_UTILS_HXX
