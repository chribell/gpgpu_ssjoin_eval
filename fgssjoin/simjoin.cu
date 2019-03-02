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

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>
#include <set>
#include <functional>

#include "simjoin.cuh"
#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"

__host__ int findSimilars(InvertedIndex index, float threshold, struct DeviceVariables *dev_vars, Pair *similar_pairs,
		int probes_start, int probe_block_size, int probes_offset,
		int indexed_start, int indexed_block_size, int indexed_offset, int block_size, bool aggregate, DeviceTiming& deviceTiming) {
	dim3 grid, threads;
	get_grid_config(grid, threads);
	int *intersection = dev_vars->d_intersection;
	int *starts = dev_vars->d_starts;
	int *sizes = dev_vars->d_sizes;
	Entry *probes = dev_vars->d_entries;
	Entry *indexed = dev_vars->d_entries;
	Pair *pairs = dev_vars->d_pairs;
	int intersection_size = block_size*block_size; // TODO verificar tamanho quando blocos são menores q os blocos normais
	int *totalSimilars = (int *)malloc(sizeof(int));

	DeviceTiming::EventPair* clearspace = deviceTiming.add("Clear intersection space", 0);
	// the last position of intersection is used to store the number of similar pairs
	cudaMemset(intersection, 0, sizeof(int) * (intersection_size + 1));
	deviceTiming.finish(clearspace);

	DeviceTiming::EventPair* genCans = deviceTiming.add("Generate candidates", 0);
	generateCandidates<<<grid, threads>>>(index, intersection, probes, starts, sizes, probes_start, probe_block_size,
			probes_offset, indexed_start, threshold, block_size);
	deviceTiming.finish(genCans);

	DeviceTiming::EventPair* verifyPairs = deviceTiming.add("Verify pairs", 0);
	verifyCandidates<<<grid, threads>>>(intersection, pairs, probes, indexed,
			 sizes, starts, probes_offset, indexed_offset, probes_start, indexed_start,
			 probe_block_size, indexed_block_size, intersection_size, intersection + intersection_size, threshold, block_size);
	deviceTiming.finish(verifyPairs);

	DeviceTiming::EventPair* transferPairs = deviceTiming.add("Transfer pairs", 0);
	gpuAssert(cudaMemcpy(totalSimilars, intersection + intersection_size, sizeof(int), cudaMemcpyDeviceToHost));
	if(!aggregate)
		gpuAssert(cudaMemcpy(similar_pairs, pairs, sizeof(Pair)*totalSimilars[0], cudaMemcpyDeviceToHost));
	deviceTiming.finish(transferPairs);

	return totalSimilars[0];
}

__global__ void generateCandidates(InvertedIndex index, int *intersection, Entry *probes, int *set_starts,
		int *set_sizes, int probes_start, int probe_block_size, int probes_offset, int indexed_start, float threshold, int block_size) {
	for (int i = blockIdx.x; i < probe_block_size; i += gridDim.x) { // percorre os probe sets
		int probe_id = i + probes_start; // setid_offset
		int probe_start = set_starts[probe_id]; // offset da entrada
		int probe_size = set_sizes[probe_id];

		int maxsize = ceil(((float) probe_size)/threshold) + 1;
		int maxprefix = probe_size - ceil(threshold * ((float) probe_size)) + 1;

		for (int j = 0; j < maxprefix; j++) { // percorre os termos de cada set
			int probe_entry = probes[probe_start + j].term_id;
			int list_size = index.d_count[probe_entry];
			int list_end = index.d_index[probe_entry];
			int list_start = list_end - list_size;

			for (int k = list_start + threadIdx.x; k < list_end; k += blockDim.x) { // percorre a lista invertida
				int idx_entry = index.d_inverted_index[k].set_id;

				if (idx_entry > probe_id && set_sizes[idx_entry] < maxsize) {
					atomicAdd(&intersection[i*block_size + idx_entry - indexed_start], 1);
				}
			}
		}
	}
}

__global__ void verifyCandidates(int *intersection, Pair *pairs, Entry *probes, Entry *indexed_sets,
		int *sizes, int *starts, int probes_offset, int indexed_offset, int probes_start,
		int indexed_start, int probe_block_size, int indexed_block_size, int intersection_size, int *totalSimilars,
		float threshold, int block_size) {

	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for (; i < intersection_size; i += gridDim.x*blockDim.x) {
		if (intersection[i]) {
		int x = i/block_size + probes_start;
		int y = i%block_size + indexed_start;

		Entry *set_x = probes + starts[x];
		Entry *set_y = indexed_sets + starts[y];

		
			float minoverlap = (threshold * ((float) (sizes[x] + sizes[y])) / (1 + threshold));
			int overlap = intersection[i], m = 0, n = 0;

			int maxprefX = sizes[x] - ceil(threshold * ((float) sizes[x])) + 1;
			int midprefY = sizes[y] - ceil(threshold * ((float) sizes[y] + sizes[y]) / (1.0 + threshold)) + 1;

			if (set_x[maxprefX].term_id < set_y[midprefY].term_id) {
				m = maxprefX;
			} else {
				n = midprefY;
			}

			while(m < sizes[x] && n < sizes[y] && sizes[x] + overlap - m >= minoverlap && sizes[y] + overlap - n >= minoverlap)	{
				if (set_x[m].term_id == set_y[n].term_id) {
					overlap++;
					n++;
					m++;
				} else if (set_x[m].term_id > set_y[n].term_id) {
					n++;
				} else {
					m++;
				}
				if (overlap >= minoverlap)
					break;
			}

			if (overlap >= minoverlap) {
				int pos = atomicAdd(totalSimilars, 1);

				pairs[pos].set_x = x;
				pairs[pos].set_y = y;
				pairs[pos].similarity = (float) overlap/(sizes[x] + sizes[y] - overlap);
			}
		}
	}
}
