/*********************************************************************
11
12	 Copyright (C) 2015 by Wisllay Vitrio
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

/*
 * knn.cuh
 *
 *  Created on: Dec 4, 2013
 *      Author: silvereagle
 */

#ifndef KNN_CUH_
#define SIMJOIN_CUH_

#include "inverted_index.cuh"
#include "device_timing.hxx"

__host__ int findSimilars(InvertedIndex inverted_index, float threshold, struct DeviceVariables *dev_vars, Similarity* distances,
		int docid, int querystart, int querysize, bool aggregate, DeviceTiming& deviceTiming);

__global__ void calculateJaccardSimilarity(InvertedIndex inverted_index, Entry *d_query, int *index, int *dist, int D, int docid);

__global__ void get_term_count_and_tf_idf(InvertedIndex inverted_index, Entry *query, int *count, int N);

__global__ void filter_registers(int *sim, float threshold, int querysize, int docid, int N, int *doc_size, Similarity *similars);

#endif /* KNN_CUH_ */
