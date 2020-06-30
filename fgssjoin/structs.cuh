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

#ifndef STRUCTS_CUH_
#define STRUCTS_CUH_

using namespace std;

struct Entry {
    int set_id;
    int term_id;
    int pos;

    __host__ __device__ Entry(int doc_id, int term_id, int pos = 0) : set_id(doc_id), term_id(term_id), pos(pos) {}
};

struct Pair {
	int set_x;
	int set_y;
};

struct DeviceVariables{
    unsigned int* d_similars;
	int *d_sizes, *d_starts, *d_intersection, *d_index, *d_count, *d_positionX, *d_positionY;
    Entry *d_inverted_index, *d_entriesmid, *d_entries;
    Pair *d_pairs;
};

struct FileStats {
	int num_sets;
	int num_terms;

	vector<int> sizes; // set sizes
	vector<int> start; // beginning of each sets
	vector<int> startmid;
};

#endif /* STRUCTS_CUH_ */
