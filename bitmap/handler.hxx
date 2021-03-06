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
    bool _binaryJoin = false;
    unsigned int _totalSimilars = 0;
    unsigned int _maxNumberOfEntries = 0;

    // Collections
    Collection<T> _hostInput;
    DeviceCollection<T> _deviceInput;
    Collection<T> _hostForeignInput;
    DeviceCollection<T> _deviceForeignInput;

    DeviceArray<unsigned int> _deviceFilter;
    DeviceArray<Pair> _devicePairs;
    std::vector<Block> _inputBlocks;
    std::vector<Block> _foreignInputBlocks;
    unsigned int* _deviceCount;
    unsigned int _blockSize;

    dim3 _grid, _threads;

public:
    Handler(double threshold, unsigned int blockSize, bool binaryJoin) : _threshold(threshold), _blockSize(blockSize), _binaryJoin(binaryJoin) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        _grid = dim3(devProp.multiProcessorCount * 16);
        _threads = dim3(devProp.maxThreadsPerBlock / 2);
        size_t totalMem;
        cudaMemGetInfo(&_memory, &totalMem);
    };

    void transferInputCollection(Collection<T>& collection);
    void transferForeignInputCollection(Collection<T>& collection);
    void constructBitmaps(unsigned int words);
    void join();
    void freeInputCollection();
    void freeForeignInputCollection();
    DeviceTiming getDeviceTimings();
    float getDeviceTotalTime();
    unsigned int getResult();
private:
    void partitionCollectionIntoBlocks();
    void selfJoin();
    void binaryJoin();
    void allocateFilterAndOutput();
    void clearFilter();
    void freeFilterAndOutput();
    void callBitmapFilter(Block probe, Block candidate, bool inverse);
    void processPairs(Block probe, Block candidate, bool inverse);
};

#endif //BITMAP_GPUSSJOIN_HANDLER_HXX
