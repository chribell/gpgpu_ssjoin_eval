/*********************************************************************
11
12	 Copyright (C) 2017 Sidney Ribeiro Junior
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

#include <cstdio>
#include <omp.h>

#include "inverted_index.cuh"
#include "utils.cuh"
#include "structs.cuh"


__host__ InvertedIndex make_inverted_index(int num_docs, int num_terms, int size_entries, int entries_offset, vector<Entry> &entries, struct DeviceVariables *dev_vars) {
	Entry *d_entries = dev_vars->d_entries + entries_offset, *d_inverted_index = dev_vars->d_inverted_index;
	int *d_count = dev_vars->d_count, *d_index = dev_vars->d_index;

	gpuAssert(cudaMemset(d_count, 0, num_terms * sizeof(int)));

	dim3 grid, threads;
	get_grid_config(grid, threads);

	double start = gettime();

	count_occurrences<<<grid, threads>>>(d_entries, d_count, size_entries);

	thrust::device_ptr<int> thrust_d_count(d_count);
	thrust::device_ptr<int> thrust_d_index(d_index);
	thrust::exclusive_scan(thrust_d_count, thrust_d_count + num_terms, thrust_d_index);

	mount_inverted_index_and_compute_tf_idf<<<grid, threads >>>(d_entries, d_inverted_index, d_count,
		d_index, size_entries, num_docs);

	gpuAssert(cudaDeviceSynchronize());

	double end = gettime();

	return InvertedIndex(d_inverted_index, d_index, d_count, d_entries, num_docs, size_entries, num_terms);
}

__global__ void count_occurrences(Entry *entries, int *count, int n) {
	int block_size = n / gridDim.x + (n % gridDim.x == 0 ? 0 : 1);		//Number of items for each block
	int offset = block_size * (blockIdx.x); 				//Beginning of the block
	int lim = offset + block_size; 						//End of block the
	if (lim >= n) lim = n;
	int size = lim - offset;						//Block size

	entries += offset;

	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		int term_id = entries[i].term_id;
		atomicAdd(count + term_id, 1);
	}
}

__global__ void mount_inverted_index_and_compute_tf_idf(Entry *entries, Entry *inverted_index, int *count, int *index, int n, int num_docs) {
	int block_size = n / gridDim.x + (n % gridDim.x == 0 ? 0 : 1);		//Number of items used by each block
	int offset = block_size * (blockIdx.x); 				//Beginning of the block
	int lim = offset + block_size; 						//End of the block
	if (lim >= n) lim = n;
	int size = lim - offset;						//Block size

	entries += offset;

	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		Entry entry = entries[i];
		int pos = atomicAdd(index + entry.term_id, 1);
		inverted_index[pos] = entry;

	}
}
