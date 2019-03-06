#ifndef BITMAP_GPUSSJOIN_HANDLER_HXX
#define BITMAP_GPUSSJOIN_HANDLER_HXX

#pragma once
#include "classes.hxx"
#include "structs.hxx"
#include "device_timing.hxx"

template <class T>
class Handler
{
private: // members
    DeviceTiming _deviceTimings;

    size_t _memory = 0;
    double _threshold = 0.0;
    unsigned int _totalSimilars = 0;
    unsigned int _maxNumberOfEntries = 0;

    Collection<T> _hostCollection;
    DeviceCollection<T> _deviceCollection;
    DeviceArray<unsigned int> _deviceFilter;
    DeviceArray<Pair> _devicePairs;
    std::vector<Block> _blocks;
    unsigned int* _deviceCount;
    unsigned int _blockSize;

    dim3 _grid, _threads;

public:
    Handler(double threshold) : _threshold(threshold) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        _grid = dim3(devProp.multiProcessorCount * 16);
        _threads = dim3(devProp.maxThreadsPerBlock / 2);
        size_t totalMem;
        cudaMemGetInfo(&_memory, &totalMem);
    };

    void transferCollection(Collection<T>& collection);
    void constructBitmaps(unsigned int words);
    void join();
    void freeCollection();
    DeviceTiming getDeviceTimings();
private:
    void partitionCollectionIntoBlocks();
    void probeBlocks();
    void allocateFilterAndOutput();
    void clearFilter();
    void freeFilterAndOutput();
    void callBitmapFilter(Block probe, Block candidate);
    void processPairs(Block probe, Block candidate);
};

#endif //BITMAP_GPUSSJOIN_HANDLER_HXX
