#ifndef FGSSJOIN_VERIFICATION_HXX
#define FGSSJOIN_VERIFICATION_HXX

#pragma once
#include "structs.hxx"
#include "similarity.hxx"

__global__ void verifyPairs(Block indexedBlock,
                            Block probeBlock,
                            DeviceCollection<unsigned int> input,
                            DeviceCollection<unsigned int> foreign,
                            DeviceArray<Pair> pairsArray,
                            DeviceArray<unsigned int> filter,
                            unsigned int blockSize,
                            double threshold,
                            unsigned int* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < pairsArray.length; i += gridDim.x * blockDim.x) {

        if (filter[i]) {

            int x = i / blockSize + probeBlock.startID;
            int y = i % blockSize + indexedBlock.startID;

            Entry<unsigned int>* probeSet = foreign.entries.array + foreign.starts[x];
            Entry<unsigned int>* indexedSet = input.entries.array + input.starts[y];

            unsigned int probeSetSize = foreign.sizes[x];
            unsigned int indexedSetSize = input.sizes[y];

            float minoverlap = (threshold * ((float) (probeSetSize + indexedSetSize)) / (1 + threshold));
            unsigned int overlap = filter[i], m = 0, n = 0;

            unsigned int probeMaxPrefix = jaccard_maxprefix(probeSetSize, threshold);
            unsigned int indexedMidPrefix = jaccard_midprefix(indexedSetSize, threshold); // TODO check if maxprefix is required for binary join


            if (probeSet[probeMaxPrefix].token < indexedSet[indexedMidPrefix].token) {
                m = probeMaxPrefix;
            } else {
                n = indexedMidPrefix;
            }

            while(m < probeSetSize && n < indexedSetSize && probeSetSize + overlap - m >= minoverlap && indexedSetSize + overlap - n >= minoverlap)	{
                if (probeSet[m].token == indexedSet[n].token) {
                    overlap++;
                    n++;
                    m++;
                } else if (probeSet[m].token > indexedSet[n].token) {
                    n++;
                } else {
                    m++;
                }
                if (overlap >= minoverlap)
                    break;
            }

            if (overlap >= minoverlap) {
                unsigned int pos = atomicAdd(count, 1);
                pairsArray[pos].firstID = x;
                pairsArray[pos].secondID = y;
            }
        }
    }
}
#endif // FGSSJOIN_VERIFICATION_HXX