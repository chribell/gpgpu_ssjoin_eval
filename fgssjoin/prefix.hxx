#ifndef FGSSJOIN_PREFIX_HXX
#define FGSSJOIN_PREFIX_HXX

#pragma once

#include "inverted_index.hxx"
#include "structs.hxx"
#include "similarity.hxx"

__global__ void prefixFilter(InvertedIndex invertedIndex,
        Block indexedBlock,
        Block probeBlock,
        DeviceCollection<unsigned int> input,
        DeviceCollection<unsigned int> foreign,
        DeviceArray<unsigned int> filter,
        unsigned int blockSize,
        double threshold)
{
    for (unsigned int i = blockIdx.x; i < probeBlock.size + 1; i += gridDim.x) {
        unsigned int probeID = i + probeBlock.startID;
        unsigned int probeStart = foreign.starts[probeID];
        unsigned int probeSize = foreign.sizes[probeID];

        unsigned int maxsize = ceil(((float) probeSize)/threshold) + 1; // why does maxsize differs ?
        unsigned int maxprefix = probeSize - ceil(threshold * ((float) probeSize)) + 1;

        for (unsigned int j = 0; j < maxprefix; j++) {

            unsigned int probeToken = foreign.entries[probeStart + j].token;
            unsigned int listSize = invertedIndex.count[probeToken];
            unsigned int listEnd = invertedIndex.index[probeToken];
            unsigned int listStart = listEnd - listSize;

            for (int k = listStart + threadIdx.x; k < listEnd; k += blockDim.x) {

                int indexedID = invertedIndex.invertedIndex[k].setID;

                if (indexedID > probeID && input.sizes[indexedID] < maxsize) {
                    atomicAdd(&filter[i * blockSize + indexedID - indexedBlock.startID], 1);
                }
            }
        }
    }
}
__global__ void binaryJoinPrefixFilter(InvertedIndex invertedIndex,
        Block indexedBlock,
        Block probeBlock,
        DeviceCollection<unsigned int> input,
        DeviceCollection<unsigned int> foreign,
        DeviceArray<unsigned int> filter,
        unsigned int blockSize,
        double threshold)
{
    for (unsigned int i = blockIdx.x; i < probeBlock.size + 1; i += gridDim.x) {
        unsigned int probeID = i + probeBlock.startID;
        unsigned int probeStart = foreign.starts[probeID];
        unsigned int probeSize = foreign.sizes[probeID];

        unsigned int maxsize = ceil(((float) probeSize)/threshold) + 1; // why does maxsize differs ?
        unsigned int maxprefix = probeSize - ceil(threshold * ((float) probeSize)) + 1;

        for (unsigned int j = 0; j < maxprefix; j++) {

            unsigned int probeToken = foreign.entries[probeStart + j].token;
            unsigned int listSize = invertedIndex.count[probeToken];
            unsigned int listEnd = invertedIndex.index[probeToken];
            unsigned int listStart = listEnd - listSize;

            for (int k = listStart + threadIdx.x; k < listEnd; k += blockDim.x) {

                int indexedID = invertedIndex.invertedIndex[k].setID;

                if (input.sizes[indexedID] < maxsize) {
                    atomicAdd(&filter[i * blockSize + indexedID - indexedBlock.startID], 1);
                }
            }
        }
    }
}


#endif // FGSSJOIN_PREFIX_HXX