#include "handler.hxx"
#include "bitmap.hxx"
#include "similarity.hxx"
#include "verification.hxx"

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

/**
 * Construct bitmap signatures on the device
 */
template <class T>
void Handler<T>::constructBitmaps(unsigned int bitmapWords)
{
    DeviceTiming::EventPair* bitmapsAllocation = _deviceTimings.add("Allocate input collection bitmaps (with initialization)", 0);
    _deviceInput.initBitmaps(bitmapWords);
    _deviceTimings.finish(bitmapsAllocation);

    // generate bitmaps
    DeviceTiming::EventPair* bitmapsGeneration = _deviceTimings.add("Generate input collection bitmaps", 0);
    generateBitmaps<<<_grid, _threads>>>(_deviceInput);
    _deviceTimings.finish(bitmapsGeneration);

    if (_binaryJoin) {
        bitmapsAllocation = _deviceTimings.add("Allocate foreign-input collection bitmaps (with initialization)", 0);
        _deviceForeignInput.initBitmaps(bitmapWords);
        _deviceTimings.finish(bitmapsAllocation);

        // generate bitmaps
        bitmapsGeneration = _deviceTimings.add("Generate foreign-input collection bitmaps", 0);
        generateBitmaps<<<_grid, _threads>>>(_deviceForeignInput);
        _deviceTimings.finish(bitmapsGeneration);
    }
}



template <class T>
void Handler<T>::join()
{
    // split the input collection in order to be able to store any output in the device memory
    partitionCollectionIntoBlocks();

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
void Handler<T>::partitionCollectionIntoBlocks()
{
    // first calculate the required memory for the input collection + (optional bitmaps)
    size_t inputRequiredMemory = _deviceInput.requiredMemory();
    size_t foreignInputRequiredMemory = _deviceForeignInput.requiredMemory();

    size_t availableMemory = _memory - inputRequiredMemory - foreignInputRequiredMemory - 100000000; // subtract another 100M for sanity
    int bytesPerCell = sizeof(Pair) + sizeof(unsigned int) * 2;
    _blockSize = (unsigned int) std::sqrt(availableMemory / bytesPerCell);

    unsigned int numberOfInputBlocks = (unsigned int)
            _deviceInput.numberOfSets / _blockSize + (_deviceInput.numberOfSets % _blockSize == 0 ? 0 : 1);

    for (unsigned int i = 0; i < numberOfInputBlocks; i++) {
        unsigned int startID = i * _blockSize;
        unsigned int endID = ((i * _blockSize) + _blockSize) - 1;
        if (endID >= _deviceInput.numberOfSets) endID = (unsigned int) _deviceInput.numberOfSets - 1;

        Block block = {i, startID, endID, _hostInput.offsets[startID], _hostInput.offsets[endID] + _hostInput.cardinalities[endID]};
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

            Block block = {i, startID, endID, _hostForeignInput.offsets[startID], _hostForeignInput.offsets[endID] + _hostForeignInput.cardinalities[endID]};
            if (block.entries > _maxNumberOfEntries) _maxNumberOfEntries = block.entries;
            _foreignInputBlocks.push_back(block);
        }
    }
}

template <class T>
void Handler<T>::selfJoin()
{
    std::vector<Block>::const_iterator leftBlock;
    std::vector<Block>::const_iterator rightBlock;

    int totalProbes = 0;

    // We incrementally process each block pair

    for(leftBlock = _inputBlocks.begin(); leftBlock != _inputBlocks.end(); ++leftBlock) {

        for(rightBlock = _inputBlocks.begin(); rightBlock != _inputBlocks.end(); ++rightBlock) {

            unsigned int rightID = (*rightBlock).id;
            unsigned int leftID  = (*leftBlock).id;

            // in case of self join stop iterating right blocks that are after current left block
            if (rightID > leftID) break;

            unsigned int leftFirstSetSize  = _hostInput.cardinalities[(*leftBlock).startID];
            unsigned int rightLastSetSize   = _hostInput.cardinalities[(*rightBlock).endID];

            if ( rightLastSetSize >= jaccard_minsize(leftFirstSetSize, _threshold)) {
                callBitmapFilter(*leftBlock, *rightBlock, false);
                processPairs( *leftBlock, *rightBlock, false);
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
    std::vector<Block>::const_iterator leftBlock;
    std::vector<Block>::const_iterator rightBlock;

    int totalProbes = 0;

    // We incrementally process each block pair

    for(leftBlock = _inputBlocks.begin(); leftBlock != _inputBlocks.end(); ++leftBlock) {

        for(rightBlock = _foreignInputBlocks.begin(); rightBlock != _foreignInputBlocks.end(); ++rightBlock) {

            unsigned int leftFirstSetSize  = _hostInput.cardinalities[(*leftBlock).startID];
            unsigned int leftLastSetSize  = _hostInput.cardinalities[(*leftBlock).endID];
            unsigned int rightFirstSetSize  = _hostForeignInput.cardinalities[(*rightBlock).startID];
            unsigned int rightLastSetSize  = _hostForeignInput.cardinalities[(*rightBlock).endID];

            // ensure length filter works both ways (if right block has bigger set sizes)
            if (rightLastSetSize >= jaccard_minsize(leftFirstSetSize, _threshold) &&
                leftLastSetSize >= jaccard_minsize(rightFirstSetSize, _threshold)) {
                if (rightFirstSetSize < leftLastSetSize) {
                    callBitmapFilter(*leftBlock, *rightBlock, false);
                    processPairs( *leftBlock, *rightBlock, false);
                } else {
                    callBitmapFilter(*leftBlock, *rightBlock, true);
                    processPairs( *leftBlock, *rightBlock, true);
                }
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
void Handler<T>::callBitmapFilter(Block leftBlock, Block rightBlock, bool inverse)
{
    DeviceTiming::EventPair* bFilter= _deviceTimings.add("Bitmap filter", 0);
    if (_binaryJoin) {
        if (inverse) {
            binaryJoinBitmapFilter<<<_grid, _threads>>>(rightBlock, leftBlock, _deviceForeignInput, _deviceInput, _deviceFilter, _blockSize, _threshold);
        } else {
            binaryJoinBitmapFilter<<<_grid, _threads>>>(leftBlock, rightBlock, _deviceInput, _deviceForeignInput, _deviceFilter, _blockSize, _threshold);
        }
    } else { // self-join
        bitmapFilter<<<_grid, _threads>>>(leftBlock, rightBlock, _deviceInput, _binaryJoin ? _deviceForeignInput : _deviceInput, _deviceFilter, _blockSize, _threshold);
    }
    _deviceTimings.finish(bFilter);
}

template <class T>
void Handler<T>::processPairs(Block leftBlock, Block rightBlock, bool inverse)
{
    DeviceTiming::EventPair* pairsVerification = _deviceTimings.add("Verify pairs", 0);
    if (_binaryJoin) {
        if (inverse) {
            verifyPairs<<<_grid, _threads>>>(rightBlock, leftBlock, _deviceInput, _deviceForeignInput, _devicePairs, _deviceFilter, _blockSize, _threshold, _deviceCount);
        } else {
            verifyPairs<<<_grid, _threads>>>(leftBlock, rightBlock, _deviceForeignInput, _deviceInput,  _devicePairs, _deviceFilter, _blockSize, _threshold, _deviceCount);
        }
    } else { // self-join
        verifyPairs<<<_grid, _threads>>>(leftBlock, rightBlock, _deviceInput, _deviceInput, _devicePairs, _deviceFilter, _blockSize, _threshold, _deviceCount);
    }

    auto* count = new unsigned int;
    gpuAssert(cudaMemcpy(count, _deviceCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    _totalSimilars += *count;
    _deviceTimings.finish(pairsVerification);
}

template class Handler<unsigned int>;