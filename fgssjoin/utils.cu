/*********************************************************************
11
12	 Copyright (C) 2015 by Wisllay Vitrio
42	 Copyright (C) 2017 by Sidney Ribeiro Junior
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

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __linux
#include <sys/time.h>

#else 
#include <sys/timeb.h>
#include <time.h>

#include <windows.h>


#endif

#ifdef __linux

double gettime() {
	timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	return double(ts.tv_sec) + double(ts.tv_nsec) / 1e9;
}
#else
#include <windows.h>
double gettime() { // granularity about 50 microsecs on my machine
	static LARGE_INTEGER freq, start;
	LARGE_INTEGER count;
	if (!QueryPerformanceCounter(&count))
		// FatalError("QueryPerformanceCounter");
		fprintf(stderr, "QueryPerformanceCounter");
	if (!freq.QuadPart) { // one time initialization
		if (!QueryPerformanceFrequency(&freq))
			//FatalError("QueryPerformanceFrequency");
			fprintf(stderr, "QueryPerformanceCounter");
		start = count;
	}
	return (double)(count.QuadPart - start.QuadPart) / freq.QuadPart;
}
#endif

#include "utils.cuh"

int WARP_SIZE = 32;

vector<string> &split(const string &s, char delim, vector<string> &elems) {
	stringstream ss(s);
	string item;
	while (getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

vector<string> split(const string &s, char delim) {
	vector<string> elems;
	split(s, delim, elems);
	return elems;
}

void get_grid_config(dim3 &grid, dim3 &threads) {
	//get the device properties
	static bool flag = 0;
	static dim3 lgrid, lthreads;
	if (!flag){
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, omp_get_thread_num() / NUM_STREAMS);

		//Adjust the grid dimensions based on the device properties
		int num_blocks = devProp.multiProcessorCount;
		lgrid = dim3(num_blocks*16);
		lthreads = dim3(devProp.maxThreadsPerBlock/2);
		//lgrid = dim3(8);
		//lthreads = dim3(512);
		flag = 1;
	}
	grid = lgrid;
	threads = lthreads;
}

void __gpuAssert(cudaError_t stat, int line, string file) {
	if (stat != cudaSuccess) {
		fprintf(stderr, "Error %s at line %d in file %s\n",
			cudaGetErrorString(stat), line, file.c_str());
		exit(1);
	}
}

float get_minoverlap_jaccard(int size_x1, int size_x2, double threshold) {
	return (threshold*((float) size_x1 + size_x2)) / (1.0 + threshold);
}

float get_minsize_jaccard(int size, double threshold) {
	return threshold * ((float) size);
}

int get_midprefix(int size, double threshold) {
	return size - ceil(get_minoverlap_jaccard(size, size, threshold)) + 1;
}

int get_maxprefix(int size, double threshold) {
	return size - ceil(get_minsize_jaccard(size, threshold)) + 1;
}

int max_size_jaccard(int size, double threshold) {
	return ceil(((float) size)/threshold);
}
