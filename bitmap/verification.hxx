#ifndef BITMAP_GPUSSJOIN_VERIFICATION_HXX
#define BITMAP_GPUSSJOIN_VERIFICATION_HXX

#pragma once
#include "structs.hxx"
#include "similarity.hxx"

__forceinline__ __device__  unsigned int intersection(
        const unsigned int* inputTokens, const unsigned int* foreignTokens,
        unsigned int firstSetStart, unsigned int rightSetEnd,
        unsigned int secondSetStart, unsigned int leftSetEnd)
{
    unsigned int count = 0;
    unsigned int i = firstSetStart, j = secondSetStart;
    while (i < rightSetEnd && j < leftSetEnd)
    {
        if (inputTokens[i] < foreignTokens[j])
            i++;
        else if (foreignTokens[j] < inputTokens[i])
            j++;
        else // intersect
        {
            count++;
            i++;
        }
    }
    return count;
}

__global__ void verifyPairs(Block leftBlock,
                            Block rightBlock,
                            DeviceCollection<unsigned int> input,
                            DeviceCollection<unsigned int> foreign,
                            DeviceArray<Pair> pairsArray,
                            DeviceArray<unsigned int> filter,
                            unsigned int blockSize,
                            double threshold,
                            unsigned int* count) {

    unsigned int pairsPerBlock = filter.length / gridDim.x + (filter.length % gridDim.x == 0 ? 0 : 1);
    unsigned int pairsStart = pairsPerBlock * (blockIdx.x);
    unsigned int pairsEnd = pairsStart + pairsPerBlock;
    if (pairsEnd >= filter.length) pairsEnd = filter.length - 1;

    for (unsigned int i = threadIdx.x; i < pairsEnd - pairsStart; i += blockDim.x) {
        unsigned int index = pairsStart + i;
        if (filter[index] > 0) {

            // extract right, left ids from 1D index
            unsigned int firstID = (index / blockSize) + (rightBlock.id * blockSize);
            unsigned int secondID = (index % blockSize) + (leftBlock.id * blockSize);

            // get set offsets and cardinalities
            unsigned int firstSetStart = foreign.offsets[secondID];
            unsigned int firstSetSize = foreign.cardinalities[secondID];
            unsigned int secondSetStart = input.offsets[firstID];
            unsigned int secondSetSize = input.cardinalities[firstID];

            unsigned int jaccardEqovelarp = jaccard_minoverlap(firstSetSize, secondSetSize, threshold);

            unsigned int inter = intersection(foreign.tokens.array,
                                              input.tokens.array,
                                              firstSetStart, firstSetStart + firstSetSize,
                                              secondSetStart, secondSetStart + secondSetSize);

            if (inter >= jaccardEqovelarp) {
                unsigned int position = atomicAdd(count, 1);
                pairsArray[position] = Pair(secondID, firstID);
            }

            filter[index] = 0;

        }
    }
}
#endif //BITMAP_GPUSSJOIN_VERIFICATION_HXX