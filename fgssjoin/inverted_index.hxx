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

#include "classes.hxx"
#include "structs.hxx"

#ifndef INVERTED_INDEX_CUH_
#define INVERTED_INDEX_CUH_


struct InvertedIndex {
    DeviceArray<unsigned int> index; //Index that indicates where each list ends in the inverted index (position after the end) d_index
    DeviceArray<unsigned int> count; //Number of entries for a given term in the inverted index d_count
    DeviceArray<Entry<unsigned int>> invertedIndex; // d_inverted_index
    DeviceArray<Entry<unsigned int>> entries; // d_entries

    unsigned int numberOfSets = 0;				//Number of documents
    unsigned int numberOfEntries = 0;			//Number of entries indexed
    unsigned int universeSize = 0;				//Number of terms

    inline void init(Collection<unsigned int>& collection, unsigned int universe) {
        numberOfSets = collection.sizes.size();
        universeSize  = universe;
        numberOfEntries = collection.prefixEntries.size();

        index.init(universeSize);
        count.init(universeSize);

        invertedIndex.init(numberOfEntries);
        entries.init(collection.prefixEntries);

    }

    inline size_t requiredMemory()
    {
        size_t memory = 0;
        memory += index.length * sizeof(unsigned int);
        memory += count.length * sizeof(unsigned int);
        memory += invertedIndex.length * sizeof(Entry<unsigned int>);
        memory += entries.length * sizeof(Entry<unsigned int>);
        return memory;
    }
};

__global__ void countOccurences(InvertedIndex invertedIndex, unsigned int globalOffset, unsigned int n);
__global__ void createIndex(InvertedIndex invertedIndex, unsigned int globalOffset, unsigned int n);

#endif /* INVERTED_INDEX_CUH_ */
