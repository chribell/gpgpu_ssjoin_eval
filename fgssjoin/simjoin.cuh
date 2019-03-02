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


#ifndef KNN_CUH_
#define SIMJOIN_CUH_

#include "inverted_index.cuh"
#include "device_timing.hxx"

__host__ int findSimilars(InvertedIndex index, float threshold, struct DeviceVariables *dev_vars, Pair *similar_pairs,
		int probes_start, int probe_block_size, int probes_offset,
		int indexed_start, int indexed_block_size, int indexed_offset, int block_size, bool aggregate, DeviceTiming& deviceTiming);

__global__ void generateCandidates(InvertedIndex index, int *intersection, Entry *probes, int *set_starts,
		int *set_sizes, int probes_start, int probe_block_size, int probes_offset, int indexed_start, float threshold, int block_size);

__global__ void verifyCandidates(int *intersection, Pair *pairs, Entry *probes, Entry *indexed_sets,
		int *sizes, int *starts, int probes_offset, int indexed_offset, int probes_start,
		int indexed_start, int probe_block_size, int indexed_block_size, int intersection_size, int *totalSimilars,
		float threshold, int block_size);

#endif /* KNN_CUH_ */
