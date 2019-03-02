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

/*
 * inverted_index.cuh
 *
 *  Created on: Dec 4, 2013
 *      Author: silvereagle
 */
#include <vector>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#include "structs.cuh"

#ifndef INVERTED_INDEX_CUH_
#define INVERTED_INDEX_CUH_

struct InvertedIndex {
    int *d_index;				//Index that indicates where each list ends in the inverted index (position after the end)
    int *d_count;				//Number of entries for a given term in the inverted index
    Entry *d_inverted_index;	//Inverted index
    Entry *d_entries;

    int num_docs;				//Number of documents
    int num_entries;			//Number of entries
    int num_terms;				//Number of terms

    __host__ __device__ InvertedIndex(Entry *d_inverted_index = NULL, int *d_index = NULL, int *d_count = NULL,
        Entry *d_entries = NULL, int num_docs = 0, int num_entries = 0, int num_terms = 0) :
        d_inverted_index(d_inverted_index),
        d_index(d_index),
        d_count(d_count),
        d_entries(d_entries),
        num_docs(num_docs),
        num_entries(num_entries),
        num_terms(num_terms)
    {}
};

__host__ InvertedIndex make_inverted_index(int num_docs, int num_terms, int size_entries, int entries_offset, std::vector<Entry> &entries, struct DeviceVariables *dev_vars);

__global__ void count_occurrences(Entry *entries, int *count, int n);

__global__ void mount_inverted_index_and_compute_tf_idf(Entry *entries, Entry *inverted_index, int *count,
    int *index, int n, int num_docs);

#endif /* INVERTED_INDEX_CUH_ */
