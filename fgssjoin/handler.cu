#include "handler.hxx"
#include "prefix.hxx"
#include "similarity.hxx"
#include "verification.hxx"
#include <thrust/reduce.h>

/**
 * Transfer input dataset collection to device
 */
template <class T>
void Handler<T>::transferInputCollection(Collection<T> &collection)
{
    _hostInput = collection;
    DeviceTiming::EventPair* collectionTransfer = _deviceTimings.add( "Transfer input collection to device (allocate + copy)", 0);
    _deviceInput.init(collection);
    _deviceTimings.finish(collectionTransfer);
}

/**
 * Transfer input dataset collection to device
 */
template <class T>
void Handler<T>::transferForeignInputCollection(Collection<T> &collection)
{
    _hostForeignInput = collection;
    DeviceTiming::EventPair* collectionTransfer = _deviceTimings.add( "Transfer foreign-input collection to device (allocate + copy)", 0);
    _deviceForeignInput.init(collection);
    _deviceTimings.finish(collectionTransfer);
}

/**
 * Free device memory used for input collection
 */
template <class T>
void Handler<T>::freeInputCollection()
{
    DeviceTiming::EventPair* freeCollection = _deviceTimings.add("Free input collection device memory", 0);
    _deviceInput.free();
    _deviceTimings.finish(freeCollection);
}

/**
 * Free device memory used for foreign-input collection
 */
template <class T>
void Handler<T>::freeForeignInputCollection()
{
    DeviceTiming::EventPair* freeCollection = _deviceTimings.add("Free foreign-input collection device memory", 0);
    _deviceForeignInput.free();
    _deviceTimings.finish(freeCollection);
}


template <class T>
DeviceTiming  Handler<T>::getDeviceTimings()
{
    gpuAssert(cudaDeviceSynchronize());
    return _deviceTimings;
}

template <class T>
void Handler<T>::initInvertedIndex()
{
    DeviceTiming::EventPair* initIndex = _deviceTimings.add("Allocate inverted index device memory", 0);
    _deviceIndex.init(_hostInput,
            _binaryJoin
            ? myMax(_hostInput.universeSize, _hostForeignInput.universeSize)
            : _hostInput.universeSize);
    _deviceTimings.finish(initIndex);
}

template <class T>
void Handler<T>::makeInvertedIndex(Block block)
{
    DeviceTiming::EventPair* makeIndex = _deviceTimings.add("Make inverted index ", 0);
    _deviceIndex.count.zero();
    unsigned int entriesOffset = _hostInput.prefixStarts[block.startID];
    unsigned int entriesSize = _hostInput.prefixStarts[block.endID] +
            (_binaryJoin
                ? jaccard_maxprefix(_hostInput.sizes[block.endID], _threshold)
                : jaccard_midprefix(_hostInput.sizes[block.endID], _threshold)) -
            entriesOffset;

    countOccurences<<<_grid, _threads>>>(_deviceIndex, entriesOffset, entriesSize);

    thrust::device_ptr<unsigned int> thrustCount(_deviceIndex.count.array);
    thrust::device_ptr<unsigned int> thrustIndex(_deviceIndex.index.array);
    thrust::exclusive_scan(thrustCount, thrustCount + _deviceIndex.universeSize, thrustIndex);

    createIndex<<<_grid, _threads>>>(_deviceIndex, entriesOffset, entriesSize);

    _deviceTimings.finish(makeIndex);
}


template <class T>
void Handler<T>::join()
{
    // allocate device memory for inverted index
    initInvertedIndex();

    // split the input collection in order to be able to store any output in the device memory
    partitionCollectionsIntoBlocks();

    // allocate filter and output arrays device memory space
    allocateFilterAndOutput();

    // conduct the join on the device
    if (_binaryJoin) {
        binaryJoin();
    } else {
        selfJoin();
    }

    // free filter and output device memory space
    freeFilterAndOutput();
}


template <class T>
void Handler<T>::partitionCollectionsIntoBlocks()
{
    size_t availableMemory =
            _memory
            - _deviceInput.requiredMemory()
            - _deviceForeignInput.requiredMemory()
            - _deviceIndex.requiredMemory()
            - 100000000; // subtract another 100M for sanity
    int bytesPerCell = sizeof(Pair) + sizeof(unsigned int) * 2;
    unsigned int maxBlockSize = (unsigned int) std::sqrt(availableMemory / bytesPerCell);

    // if the given block size is larger than the maximum supported block size, decrease it
    if (_blockSize > maxBlockSize) _blockSize = maxBlockSize;

    unsigned int numberOfInputBlocks = (unsigned int)
            _deviceInput.numberOfSets / _blockSize + (_deviceInput.numberOfSets % _blockSize == 0 ? 0 : 1);

    for (unsigned int i = 0; i < numberOfInputBlocks; i++) {
        unsigned int startID = i * _blockSize;
        unsigned int endID = ((i * _blockSize) + _blockSize) - 1;
        if (endID >= _deviceInput.numberOfSets) endID = (unsigned int) _deviceInput.numberOfSets - 1;

        Block block = {i, startID, endID, _hostInput.starts[startID], _hostInput.starts[endID] + _hostInput.sizes[endID]};
        if (block.entries > _maxNumberOfEntries) _maxNumberOfEntries = block.entries;
        _inputBlocks.push_back(block);
    }

    if (_binaryJoin) {
        unsigned int numberOfForeignInputBlocks = (unsigned int)
            _deviceForeignInput.numberOfSets / _blockSize + (_deviceForeignInput.numberOfSets % _blockSize == 0 ? 0 : 1);

        for (unsigned int i = 0; i < numberOfForeignInputBlocks; i++) {
            unsigned int startID = i * _blockSize;
            unsigned int endID = ((i * _blockSize) + _blockSize) - 1;
            if (endID >= _deviceForeignInput.numberOfSets) endID = (unsigned int) _deviceForeignInput.numberOfSets - 1;

            Block block = {i, startID, endID, _hostForeignInput.starts[startID], _hostForeignInput.starts[endID] + _hostForeignInput.sizes[endID]};
            if (block.entries > _maxNumberOfEntries) _maxNumberOfEntries = block.entries;
            _foreignInputBlocks.push_back(block);
        }
    }

    std::cout << "Number of input blocks: " << _inputBlocks.size() << "\n";
    std::cout << "Number of foreign input blocks: " << _foreignInputBlocks.size() << "\n";
    std::cout << "Number of blocks: " << _inputBlocks.size() + _foreignInputBlocks.size() << "\n";
}

template <class T>
void Handler<T>::selfJoin()
{
    std::vector<Block>::const_iterator indexedBlock;
    std::vector<Block>::const_iterator probeBlock;

    int totalProbes = 0;

    // We incrementally process each block pair

    for(indexedBlock = _inputBlocks.begin(); indexedBlock != _inputBlocks.end(); ++indexedBlock) {

        unsigned int indexedID = (*indexedBlock).id;
        makeInvertedIndex(*indexedBlock);

        for(probeBlock = _inputBlocks.begin(); probeBlock != _inputBlocks.end(); ++probeBlock) {

            unsigned int probeID  = (*probeBlock).id;

            // in case of self join stop iterating right blocks that are after current left block
            if (probeID > indexedID) break;

            unsigned int firstIndexedSetSize  = _hostInput.sizes[(*indexedBlock).startID];
            unsigned int lastProbeSetSize   = _hostInput.sizes[(*probeBlock).endID];

            if ( lastProbeSetSize >= jaccard_minsize(firstIndexedSetSize, _threshold)) {
                filter(*indexedBlock, *probeBlock);
                verify(*indexedBlock, *probeBlock);
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
void Handler<T>::binaryJoin()
{
    std::vector<Block>::const_iterator indexedBlock;
    std::vector<Block>::const_iterator probeBlock;

    int totalProbes = 0;

    // We incrementally process each block pair

    for(indexedBlock = _inputBlocks.begin(); indexedBlock != _inputBlocks.end(); ++indexedBlock) {

        makeInvertedIndex(*indexedBlock);

        for(probeBlock = _foreignInputBlocks.begin(); probeBlock != _foreignInputBlocks.end(); ++probeBlock) {

            unsigned int leftFirstSetSize  = _hostInput.sizes[(*indexedBlock).startID];
            unsigned int leftLastSetSize  = _hostInput.sizes[(*indexedBlock).endID];
            unsigned int rightFirstSetSize  = _hostForeignInput.sizes[(*probeBlock).startID];
            unsigned int rightLastSetSize  = _hostForeignInput.sizes[(*probeBlock).endID];

            // ensure length filter works both ways (if right block has bigger set sizes)
            if (rightLastSetSize >= jaccard_minsize(leftFirstSetSize, _threshold) &&
                leftLastSetSize >= jaccard_minsize(rightFirstSetSize, _threshold)) {
                filter(*indexedBlock, *probeBlock);
                verify(*indexedBlock, *probeBlock);
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
void Handler<T>::filter(Block indexedBlock, Block probeBlock)
{
    DeviceTiming::EventPair* bFilter= _deviceTimings.add("Generate Candidates", 0);
    if (_binaryJoin) {
        binaryJoinPrefixFilter<<<_grid, _threads>>>(_deviceIndex, indexedBlock, probeBlock, _deviceInput, _deviceForeignInput, _deviceFilter, _blockSize, _threshold);
    } else {
        prefixFilter<<<_grid, _threads>>>(_deviceIndex, indexedBlock, probeBlock, _deviceInput, _deviceInput, _deviceFilter, _blockSize, _threshold);
    }
    _deviceTimings.finish(bFilter);
}

template <class T>
void Handler<T>::verify(Block indexedBlock, Block probeBlock)
{
    DeviceTiming::EventPair* pairsVerification = _deviceTimings.add("Verify pairs", 0);

    if (_binaryJoin) {
        verifyPairs<<<_grid, _threads>>>(indexedBlock, probeBlock, _deviceInput, _deviceForeignInput, _devicePairs, _deviceFilter, _blockSize, _threshold, _deviceCount);
    } else { // self-join
        verifyPairs<<<_grid, _threads>>>(indexedBlock, probeBlock, _deviceInput, _deviceInput, _devicePairs, _deviceFilter, _blockSize, _threshold, _deviceCount);
    }

    auto* count = new unsigned int;
    gpuAssert(cudaMemcpy(count, _deviceCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    _totalSimilars += *count;
    _deviceTimings.finish(pairsVerification);
}

template class Handler<unsigned int>;