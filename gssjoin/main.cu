/*********************************************************************
11
12	 Copyright (C) 2016 by Sidney Ribeiro Junior
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

#include "gpu.h"

#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <string>
#include <sstream>
#include <cuda.h>
#include <map>

#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "simjoin.cuh"
#include "device_timing.hxx"

#define NUM_STREAMS 1

using namespace std;

struct FileStats {
	int num_sets;
	int num_terms;

	vector<int> sizes; // set sizes
	vector<int> start; // beginning of each entrie

	FileStats() : num_sets(0), num_terms(0) {}
};

FileStats readInputFile(string &file, vector<Entry> &entries);
void processTestFile(InvertedIndex &index, FileStats &stats, string &file, float threshold, stringstream &fileout, bool aggregate, DeviceTiming& deviceTiming);


int biggestQuerySize = -1;

int gpu(int argc, char **argv) {

    if (argc != 6) {
        cerr
                << "Wrong parameters. Correct usage: <executable> <input_file> <threshold> <output_file> <number_of_gpus> <aggregate>"
                << endl;
        exit(1);
    }

    int gpuNum;
    cudaGetDeviceCount(&gpuNum);

    if (gpuNum > atoi(argv[4])) {
        gpuNum = atoi(argv[4]);
        if (gpuNum < 1)
            gpuNum = 1;
    }
    bool aggregate = atoi(argv[5]) == 1;
    //cerr << "Using " << gpuNum << "GPUs" << endl;

    // we use 2 streams per GPU
    int numThreads = gpuNum * NUM_STREAMS;

    omp_set_num_threads(numThreads);

    //truncate output files
    ofstream ofsf(argv[3], ofstream::trunc);
    ofsf.close();

    ofstream ofsfileoutput(argv[3], ofstream::out | ofstream::app);

	vector<string> inputs;// to read the whole test file in memory
	vector<InvertedIndex> indexes;
	indexes.resize(gpuNum);

	double starts, ends;

	string inputFileName(argv[1]);

	printf("Reading file...\n");
	vector<Entry> entries;

	starts = gettime();
	FileStats stats = readInputFile(inputFileName, entries);
	ends = gettime();

	printf("Time taken: %lf seconds\n", ends - starts);
	DeviceTiming deviceTiming;
	vector<stringstream*> outputString;
	//Each thread builds an output string, so it can be flushed at once at the end of the program
	for (int i = 0; i < numThreads; i++){
		outputString.push_back(new stringstream);
	}

	//create an inverted index for all streams in each GPU
	#pragma omp parallel num_threads(gpuNum)
	{
		int cpuid = omp_get_thread_num();
		cudaSetDevice(cpuid);
		double start, end;

		start = gettime();
		indexes[cpuid] = make_inverted_index(stats.num_sets, stats.num_terms, entries);
		end = gettime();

		#pragma omp single nowait
		printf("Total time taken for insertion: %lf seconds\n", end - start);
	}


	#pragma omp parallel 
	{
		int cpuid = omp_get_thread_num();
		cudaSetDevice(cpuid / NUM_STREAMS);

		float threshold = atof(argv[2]);

		FileStats lstats = stats;

		processTestFile(indexes[cpuid / NUM_STREAMS], lstats, inputFileName, threshold, *outputString[cpuid], aggregate, deviceTiming);
		if (cpuid %  NUM_STREAMS == 0)
			gpuAssert(cudaDeviceReset());

	}

    if (!aggregate) {
        starts = gettime();
        for (int i = 0; i < numThreads; i++){
            ofsfileoutput << outputString[i]->str();
        }
        ends = gettime();
        printf("Time taken to write output: %lf seconds\n", ends - starts);
    }

    ofsfileoutput.close();

    return 0;
}

FileStats readInputFile(string &filename, vector<Entry> &entries) {
	ifstream input(filename.c_str());
	string line;

	FileStats stats;
	int accumulatedsize = 0;
	int set_id = 0;

	while (!input.eof()) {
		getline(input, line);
		if (line == "") continue;

		vector<string> tokens = split(line, ' ');
		biggestQuerySize = max((int)tokens.size(), biggestQuerySize);

		int size = tokens.size();
		stats.sizes.push_back(size);
		stats.start.push_back(accumulatedsize);
		accumulatedsize += size;

		for (int i = 0; i < size; i++) {
			int term_id = atoi(tokens[i].c_str());
			stats.num_terms = max(stats.num_terms, term_id + 1);
			entries.push_back(Entry(set_id, term_id));
		}
		set_id++;
	}

	stats.num_sets = stats.start.size();

	input.close();

	return stats;
}

void allocVariables(DeviceVariables *dev_vars, float threshold, int num_sets, Similarity** distances){
	dim3 grid, threads;

	get_grid_config(grid, threads);

	gpuAssert(cudaMalloc(&dev_vars->d_dist, num_sets * sizeof(Similarity))); // distance between all the sets and the query doc
	gpuAssert(cudaMalloc(&dev_vars->d_result, num_sets * sizeof(Similarity))); // compacted similarities between all the sets and the query doc
	gpuAssert(cudaMalloc(&dev_vars->d_sim, num_sets * sizeof(int))); // count of elements in common
	gpuAssert(cudaMalloc(&dev_vars->d_sizes, num_sets * sizeof(int))); // size of all sets
	gpuAssert(cudaMalloc(&dev_vars->d_query, biggestQuerySize * sizeof(Entry))); // query
	gpuAssert(cudaMalloc(&dev_vars->d_index, biggestQuerySize * sizeof(int)));
	gpuAssert(cudaMalloc(&dev_vars->d_count, biggestQuerySize * sizeof(int)));

	*distances = (Similarity*)malloc(num_sets * sizeof(Similarity));

	int blocksize = 1024;
	int numBlocks = num_sets / blocksize + (num_sets % blocksize ? 1 : 0);

	gpuAssert(cudaMalloc(&dev_vars->d_bC,sizeof(int)*(numBlocks + 1)));
	gpuAssert(cudaMalloc(&dev_vars->d_bO,sizeof(int)*numBlocks));

}

void freeVariables(DeviceVariables *dev_vars, InvertedIndex &index, Similarity** distances){
	cudaFree(dev_vars->d_dist);
	cudaFree(dev_vars->d_result);
	cudaFree(dev_vars->d_sim);
	cudaFree(dev_vars->d_sizes);
	cudaFree(dev_vars->d_query);
	cudaFree(dev_vars->d_index);
	cudaFree(dev_vars->d_count);
	cudaFree(dev_vars->d_bC);
	cudaFree(dev_vars->d_bO);

	free(*distances);

	if (omp_get_thread_num() % NUM_STREAMS == 0){
		cudaFree(index.d_count);
		cudaFree(index.d_index);
		cudaFree(index.d_inverted_index);
	}
}

void processTestFile(InvertedIndex &index, FileStats &stats, string &filename, float threshold, stringstream &outputfile, bool aggregate, DeviceTiming& deviceTiming) {

	int num_test_local = 0, setid;

	//#pragma omp single nowait
	size_t finalResult = 0;
	DeviceVariables dev_vars;
	Similarity* distances;

	allocVariables(&dev_vars, threshold, index.num_sets, &distances);

	cudaMemcpyAsync(dev_vars.d_sizes, &stats.sizes[0], index.num_sets * sizeof(int), cudaMemcpyHostToDevice);

	double start = gettime();

#pragma omp for
	for (setid = 0; setid < index.num_sets - 1; setid++){

		num_test_local++;

		int totalSimilars = findSimilars(index, threshold, &dev_vars, distances, setid, stats.start[setid], stats.sizes[setid], aggregate, deviceTiming);
        finalResult += totalSimilars;
        if (!aggregate) {
            for (int i = 0; i < totalSimilars; i++) {
                outputfile << "(" << setid << ", " << distances[i].set_id << "): "
                    << distances[i].similarity << endl;
            }
        }
    }

	freeVariables(&dev_vars, index, &distances);
	int threadid = omp_get_thread_num();

	#pragma omp barrier

	double end = gettime();

    std::cout
        << "Result: " << finalResult << std::endl
        << "Runtime: " << end - start << " secs" << std::endl;

    std::cout << deviceTiming;
}
