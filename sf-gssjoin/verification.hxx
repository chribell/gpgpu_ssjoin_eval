#ifndef SFGSSJOIN_VERIFICATION_HXX
#define SFGSSJOIN_VERIFICATION_HXX

#pragma once

#include "inverted_index.hxx"
#include "structs.hxx"
#include "similarity.hxx"


__global__ void calculateIntersection(InvertedIndex invertedIndex,
                                      Block indexedBlock,
                                      Block probeBlock,
                                      DeviceCollection<unsigned int> input,
                                      DeviceCollection<unsigned int> foreign,
                                      DeviceArray<unsigned int> intersection,
                                      unsigned int blockSize,
                                      double threshold,
                                      bool binaryJoin)
{
    for (int i = blockIdx.x; i < probeBlock.size + 1; i += gridDim.x) {
        int probeID = i + probeBlock.startID;
        int probeStart = foreign.starts[probeID];
        int probeSize = foreign.sizes[probeID];

        unsigned int maxsize = ceil(((float) probeSize)/threshold) + 1;

        for (int j = 0; j < probeSize; j++) {
            unsigned int probeToken = foreign.entries[probeStart + j].token;
            unsigned int listSize = invertedIndex.count[probeToken];
            unsigned int listEnd = invertedIndex.index[probeToken];
            unsigned int listStart = listEnd - listSize;

            for (int k = listStart + threadIdx.x; k < listEnd; k += blockDim.x) {
                int indexedID = invertedIndex.invertedIndex[k].setID;

                if ( (binaryJoin || indexedID > probeID) && input.sizes[indexedID] < maxsize)
                    atomicAdd(&intersection[i * blockSize + indexedID - indexedBlock.startID], 1);
            }
        }
    }
}

__global__ void verifyPairs(Block indexedBlock,
                            Block probeBlock,
                            DeviceCollection<unsigned int> input,
                            DeviceCollection<unsigned int> foreign,
                            DeviceArray<Pair> pairsArray,
                            DeviceArray<unsigned int> intersection,
                            unsigned int blockSize,
                            double threshold,
                            unsigned int* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < pairsArray.length; i += gridDim.x * blockDim.x) {

        if (intersection[i]) {

            int x = i / blockSize + probeBlock.startID;
            int y = i % blockSize + indexedBlock.startID;

            unsigned int probeSetSize = foreign.sizes[x];
            unsigned int indexedSetSize = input.sizes[y];

            float similarity = (float) intersection[i]/(probeSetSize + indexedSetSize - intersection[i]);


            if (similarity >= threshold) {
                unsigned int pos = atomicAdd(count, 1);
                pairsArray[pos].firstID = x;
                pairsArray[pos].secondID = y;
            }
        }
    }
}
#endif // SFGSSJOIN_VERIFICATION_HXX