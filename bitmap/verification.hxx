#ifndef BITMAP_GPUSSJOIN_VERIFICATION_HXX
#define BITMAP_GPUSSJOIN_VERIFICATION_HXX

#pragma once
#include "structs.hxx"
#include "similarity.hxx"

__forceinline__ __device__  unsigned int intersection(unsigned int* tokens, unsigned int recordStart, unsigned int recordEnd,
                                                      unsigned int candidateStart, unsigned int candidateEnd)
{
    unsigned int count = 0;
    unsigned int i = recordStart, j = candidateStart;
    while (i < recordEnd && j < candidateEnd)
    {
        if (tokens[i] < tokens[j])
            i++;
        else if (tokens[j] < tokens[i])
            j++;
        else // intersect
        {
            count++;
            i++;
        }
    }
    return count;
}

__global__ void verifyPairs(Block probe, Block candidate, DeviceCollection<unsigned int> collection,
                             DeviceArray<Pair> pairsArray, DeviceArray<unsigned int> filter,
                             unsigned int blockSize, double threshold, unsigned int* count) {
    unsigned int pairsPerBlock = filter.length / gridDim.x + (filter.length % gridDim.x == 0 ? 0 : 1);
    unsigned int pairsStart = pairsPerBlock * (blockIdx.x);
    unsigned int pairsEnd = pairsStart + pairsPerBlock;
    if (pairsEnd >= filter.length) pairsEnd = filter.length - 1;

    for (unsigned int i = threadIdx.x; i < pairsEnd - pairsStart; i += blockDim.x) {
        unsigned int index = pairsStart + i;
        if (filter[index] > 0) {
            // extract probe, candidate ids from 1D index
            unsigned int probeID = (index / blockSize) + (probe.id * blockSize);
            unsigned int candidateID = (index % blockSize) + (candidate.id * blockSize);
            // get set offsets and cardinalities
            unsigned int probeStart = collection.offsets[probeID];
            unsigned int probeSize = collection.cardinalities[probeID];
            unsigned int probePrefix = probeStart + jaccard_maxprefix(probeSize, threshold);
            unsigned int candidateStart = collection.offsets[candidateID];
            unsigned int candidateSize = collection.cardinalities[candidateID];

            unsigned int jaccardEqovelarp = jaccard_minoverlap(probeSize, candidateSize,
                                                                            threshold);

            unsigned int inter = intersection(collection.tokens.array, probeStart, probeStart + probeSize,
                                              candidateStart, candidateStart + candidateSize);

            if (inter >= jaccardEqovelarp) {
                unsigned int position = atomicAdd(count, 1);
                pairsArray[position] = Pair(probeID, candidateID);
            }

            filter[index] = 0;

        }
    }
}
#endif //BITMAP_GPUSSJOIN_VERIFICATION_HXX