#ifndef BITMAP_GPUSSJOIN_STRUCTS_HXX
#define BITMAP_GPUSSJOIN_STRUCTS_HXX

#pragma once
#include <numeric>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "classes.hxx"
#include "utils.hxx"


template <class T>
struct DeviceArray {
    T* array;
    size_t length = 0;

    inline void init(size_t len, T* arrayPtr)
    {
        length = len;
        gpuAssert(cudaMalloc((void**) &array, length * sizeof(T)));
        gpuAssert(cudaMemcpy(array, arrayPtr, length * sizeof(T), cudaMemcpyHostToDevice));
    }

    inline void init(std::vector<T>& vector)
    {
        length = vector.size();
        gpuAssert(cudaMalloc((void**) &array, length * sizeof(T)));
        gpuAssert(cudaMemcpy(array, &vector[0], length * sizeof(T), cudaMemcpyHostToDevice));
    }

    inline void init(size_t len)
    {
        length = len;
        gpuAssert(cudaMalloc((void**) &array, length * sizeof(T)));
        gpuAssert(cudaMemset(array, 0, length * sizeof(T)));
    }

    inline void zero()
    {
        gpuAssert(cudaMemset(array, 0, length * sizeof(T)));
    }

    inline void free()
    {
        gpuAssert(cudaFree(array));
        length = 0;
    }

    __forceinline__ __host__ __device__ T& operator[] (unsigned int i)
    {
        return array[i];
    }
};

template <class T>
struct DeviceCollection {
    DeviceArray<T> tokens;
    DeviceArray<unsigned int> offsets;
    DeviceArray<unsigned int> cardinalities;
    DeviceArray<Entry<T>> entries;
    DeviceArray<word> bitmaps;

    size_t numberOfSets = 0;
    size_t numberOfTerms = 0;
    size_t numberOfEntries = 0;

    unsigned int bitmapSize = 0;
    unsigned int bitmapWords = 0;

    inline void init(Collection<T>& collection)
    {
        numberOfSets    = collection.cardinalities.size();
        numberOfTerms   = collection.numberOfTerms;
        numberOfEntries = collection.entries.size();

        tokens.init(collection.tokens);
        cardinalities.init(collection.cardinalities);
        offsets.init(collection.offsets);
        entries.init(collection.entries);
    }

    inline void initBitmaps(unsigned int words)
    {
        // allocate memory space and set bitmaps to 0
        bitmapWords = words;
        bitmapSize  = bitmapWords * WORD_BITS;
        bitmaps.init(numberOfSets * bitmapWords);
    }

    __forceinline__ __host__ __device__ word* bitmapAt(size_t pos)
    {
        return bitmaps.array + (pos * bitmapWords);
    }

    __forceinline__ __device__ Entry<T>* entryAt(size_t pos)
    {
        return entries.array + pos;
    }

    inline void free() {
        tokens.free();
        cardinalities.free();
        offsets.free();
        entries.free();
        if (bitmaps.length > 0) {
            bitmaps.free();
        }
    }

    inline size_t requiredMemory()
    {
        size_t memory = 0;
        memory += tokens.length  * sizeof(T);
        memory += cardinalities.length * sizeof(unsigned int);
        memory += offsets.length * sizeof(unsigned int);
        memory += entries.length * sizeof(Entry<T>);
        memory += bitmaps.length * sizeof(word);
        return memory;
    }
};

struct Block
{
    unsigned int id;
    unsigned int startID;
    unsigned int endID;
    unsigned int firstEntryPosition;
    unsigned int lastEntryPosition;
    unsigned int entries;
    unsigned int size;
    Block() = default;
    Block(unsigned int id, unsigned int startID, unsigned int endID, unsigned int firstEntryPosition, unsigned int lastEntryPosition)
        : id(id), startID(startID), endID(endID), firstEntryPosition(firstEntryPosition), lastEntryPosition(lastEntryPosition)
    {
        size = endID - startID;
        entries = lastEntryPosition - firstEntryPosition;
    }
};

struct Pair
{
    unsigned int firstID;
    unsigned int secondID;
    Pair() = default;
    __host__ __device__ Pair(unsigned int firstID, unsigned int secondID) : firstID(firstID), secondID(secondID) {}
};

#endif //BITMAP_GPUSSJOIN_STRUCTS_HXX
