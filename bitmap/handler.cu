#include "handler.hxx"
#include "bitmap.hxx"
#include "similarity.hxx"
#include "verification.hxx"

/**
 * Transfer dataset collection to device
 */
template <class T>
void Handler<T>::transferCollection(Collection<T>& collection)
{
    _hostCollection = collection;
    DeviceTiming::EventPair* collectionTransfer = _deviceTimings.add("Transfer input collection to device (allocate + copy)", 0);
    _deviceCollection.init(collection);
    _deviceTimings.finish(collectionTransfer);
}

/**
 * Free device memory used for dataset collection
 */
template <class T>
void Handler<T>::freeCollection()
{
    DeviceTiming::EventPair* freeCollection = _deviceTimings.add("Free input collection device memory", 0);
    _deviceCollection.free();
    _deviceTimings.finish(freeCollection);
}


template <class T>
DeviceTiming  Handler<T>::getDeviceTimings()
{
    gpuAssert(cudaDeviceSynchronize());
    return _deviceTimings;
}

/**
 * Construct bitmap signatures on the device
 */
template <class T>
void Handler<T>::constructBitmaps(unsigned int bitmapWords)
{
    DeviceTiming::EventPair* bitmapsAllocation = _deviceTimings.add("Allocate bitmap space (with initialization)", 0);
    _deviceCollection.initBitmaps(bitmapWords);
    _deviceTimings.finish(bitmapsAllocation);

    // generate bitmaps
    DeviceTiming::EventPair* bitmapsGeneration = _deviceTimings.add("Generate bitmaps", 0);
    generateBitmaps<<<_grid, _threads>>>(_deviceCollection);
    _deviceTimings.finish(bitmapsGeneration);
}



template <class T>
void Handler<T>::join()
{
    // split the input collection in order to be able to store any output in the device memory
    partitionCollectionIntoBlocks();

    // allocate filter and output arrays device memory space
    allocateFilterAndOutput();

    // conduct the join on the device
    probeBlocks();

    // free filter and output device memory space
    freeFilterAndOutput();
}


template <class T>
void Handler<T>::partitionCollectionIntoBlocks()
{
    // first calculate the required memory for the input collection + (optional bitmaps)
    size_t requiredMemory = _deviceCollection.requiredMemory();

    size_t availableMemory = _memory - requiredMemory - 100000000; // subtract another 100M for sanity
    int bytesPerCell = sizeof(Pair) + sizeof(unsigned int) * 2;
    _blockSize = (unsigned int) std::sqrt(availableMemory / bytesPerCell);

    unsigned int numberOfBlocks = (unsigned int)
            _deviceCollection.numberOfSets / _blockSize + (_deviceCollection.numberOfSets % _blockSize == 0 ? 0 : 1);

    for (unsigned int i = 0; i < numberOfBlocks; i++) {
        unsigned int startID = i * _blockSize;
        unsigned int endID = ((i * _blockSize) + _blockSize) - 1;
        if (endID >= _deviceCollection.numberOfSets) endID = (unsigned int) _deviceCollection.numberOfSets - 1;

        Block block = {i, startID, endID, _hostCollection.offsets[startID], _hostCollection.offsets[endID] + _hostCollection.cardinalities[endID]};
        if (block.entries > _maxNumberOfEntries) _maxNumberOfEntries = block.entries;
        _blocks.push_back(block);
    }
}

template <class T>
void Handler<T>::probeBlocks()
{
    std::vector<Block>::const_iterator indexedBlock;
    std::vector<Block>::const_iterator probeBlock;

    std::cout << "Block size: " << _blockSize << std::endl;
    std::cout << "Number of blocks: " << _blocks.size() << std::endl;

    int totalProbes = 0;

    for(indexedBlock = _blocks.begin(); indexedBlock != _blocks.end(); ++indexedBlock) {

        for(probeBlock = _blocks.begin(); probeBlock != _blocks.end(); ++probeBlock) {

            unsigned int probeBlockID     = (*probeBlock).id;
            unsigned int candidateBlockID = (*indexedBlock).id;

            unsigned int firstIndexedSetSize  = _hostCollection.cardinalities[(*indexedBlock).startID];
            unsigned int lastProbeSetSize     = _hostCollection.cardinalities[(*probeBlock).endID];

            if ( (probeBlockID <= candidateBlockID)  && (lastProbeSetSize >= jaccard_minsize(firstIndexedSetSize, _threshold))) {
                callBitmapFilter(*probeBlock, *indexedBlock);
                processPairs(*probeBlock, *indexedBlock);
                clearFilter();
                gpuAssert(cudaMemset(_deviceCount, 0, sizeof(unsigned int)));
                totalProbes++;
            }
        }
    }
    std::cout << "Total block probes: " << totalProbes << std::endl;
    std::cout << "Total similars: " << _totalSimilars << std::endl;
}

template <class T>
void Handler<T>::allocateFilterAndOutput()
{
    DeviceTiming::EventPair* allocateOutput = _deviceTimings.add("Allocate filter & output arrays", 0);
    _deviceFilter.init(_blockSize * _blockSize);
    _devicePairs.init(_blockSize * _blockSize);
    gpuAssert(cudaMalloc((void**) &_deviceCount, sizeof(unsigned int)));
    gpuAssert(cudaMemset(_deviceCount, 0, sizeof(unsigned int)));
    _deviceTimings.finish(allocateOutput);
}

template <class T>
void Handler<T>::clearFilter()
{
    DeviceTiming::EventPair* clearFilterSpace = _deviceTimings.add("Clear filter space", 0);
    _deviceFilter.zero();
    _deviceTimings.finish(clearFilterSpace);
}

template <class T>
void Handler<T>::freeFilterAndOutput()
{
    DeviceTiming::EventPair* freeOutput = _deviceTimings.add("Free filter arrays & output pairs space", 0);
    _deviceFilter.free();
    _devicePairs.free();
    gpuAssert(cudaFree(_deviceCount));
    _deviceTimings.finish(freeOutput);
}

template <class T>
void Handler<T>::callBitmapFilter(Block probe, Block candidate)
{
    DeviceTiming::EventPair* bFilter= _deviceTimings.add("Bitmap filter", 0);
    bitmapFilter<<<_grid, _threads>>>(probe, candidate, _deviceCollection, _deviceFilter, _blockSize, _threshold);
    _deviceTimings.finish(bFilter);
}

template <class T>
void Handler<T>::processPairs(Block probe, Block candidate)
{
    DeviceTiming::EventPair* pairsVerification = _deviceTimings.add("Verify pairs", 0);
    verifyPairs<<<_grid, _threads>>>(probe, candidate, _deviceCollection, _devicePairs, _deviceFilter, _blockSize, _threshold, _deviceCount);
    unsigned int* count = new unsigned int;
    gpuAssert(cudaMemcpy(count, _deviceCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    _totalSimilars += *count;
    _deviceTimings.finish(pairsVerification);
}

template class Handler<unsigned int>;