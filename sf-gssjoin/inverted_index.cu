/*********************************************************************
11
12	 Copyright (C) 2017 by Sidney Ribeiro Junior
13
14	 This program is free software; you can redistribute it and/or modify
15	 it under the terms of the GNU General Public License as published by
16	 the Free Software Foundation; either version 2 of the License, or
17	 (at your option) any later version.
18
19	 This program is distributed in the hope that it will be useful,
20	 but WITHOUT ANY WARRANTY; without even the implied warranty of
21	 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
22	 GNU General Public License for more details.
23
24	 You should have received a copy of the GNU General Public License
25	 along with this program; if not, write to the Free Software
26	 Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
27
28	 ********************************************************************/

#include "inverted_index.hxx"

__global__ void countOccurences(InvertedIndex invertedIndex, unsigned int globalOffset, unsigned int n) {
	int block_size = n / gridDim.x + (n % gridDim.x == 0 ? 0 : 1);		//Number of items for each block
	int offset = block_size * (blockIdx.x); 				//Beginning of the block
	int lim = offset + block_size; 						//End of block the
	if (lim >= n) lim = n;
	int size = lim - offset;						//Block size

	invertedIndex.entries.array += (globalOffset + offset);

	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		int token = invertedIndex.entries[i].token;
		atomicAdd(&invertedIndex.count[token], 1);
	}
}

__global__ void createIndex(InvertedIndex invertedIndex, unsigned int globalOffset, unsigned int n) {
	int block_size = n / gridDim.x + (n % gridDim.x == 0 ? 0 : 1);		//Number of items used by each block
	int offset = block_size * (blockIdx.x); 				//Beginning of the block
	int lim = offset + block_size; 						//End of the block
	if (lim >= n) lim = n;
	int size = lim - offset;						//Block size

    invertedIndex.entries.array += (globalOffset + offset);

	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		Entry<unsigned int> entry = invertedIndex.entries[i];
		int pos = atomicAdd(&invertedIndex.index[entry.token], 1);
		invertedIndex.invertedIndex[pos] = entry;
	}
}
