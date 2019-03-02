#include <stdio.h>
#include <stdlib.h>

#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"

void print_sets(vector<Entry> &entries, vector<int> &sizes, vector<int> &start) {
	printf("\nSets:\n");
	for (int i = 0; i < sizes.size(); i++) {
		printf("[%d]: ", i);
		for (int j = 0; j < sizes[i]; j++) {
			printf(" %d ", entries[start[i] + j].term_id);
		}
		printf("\n");
	}

}

void print_invertedIndex(InvertedIndex index) {
	printf("Docs: %d\nEntries: %d\nTerms: %d\n", index.num_docs, index.num_entries, index.num_terms);

	Entry *inverted_index = (Entry*)malloc(sizeof(Entry)*index.num_entries);
	cudaMemcpyAsync(inverted_index, index.d_inverted_index, sizeof(Entry)*index.num_entries, cudaMemcpyDeviceToHost);

	int *count = (int *)malloc(sizeof(int)*index.num_terms);
	cudaMemcpyAsync(count, index.d_count, sizeof(int)*index.num_terms, cudaMemcpyDeviceToHost);

	int *h_index = (int *)malloc(sizeof(int)*index.num_terms);
	cudaMemcpyAsync(h_index, index.d_index, sizeof(int)*index.num_terms, cudaMemcpyDeviceToHost);

	printf("Count: ");
	for (int i = 0; i < index.num_terms; i++) {
		printf("%d ", count[i]);
	}

	printf("\nList's ends: ");
		for (int i = 0; i < index.num_terms; i++) {
			printf("%d ", h_index[i]);
	}

	printf("\nIndex:");
	int term = -1;
	for (int i = 0; i < index.num_entries; i++) {
		if (term != inverted_index[i].term_id) {
			printf("\n[%d]: ", inverted_index[i].term_id);
			term = inverted_index[i].term_id;
		}
		printf("%d ", inverted_index[i].set_id);
	}
	printf("\n");
}

void print_intersection(int *intersection, int block_size, int indexed, int probe) {
	int *h_intersection = (int *)malloc(sizeof(int)*block_size*block_size);
	cudaMemcpyAsync(h_intersection, intersection, sizeof(int)*block_size*block_size, cudaMemcpyDeviceToHost);


	printf("\n===Intersection (%d, %d):===\n   ", probe, indexed);
	for (int i = 0; i < block_size; i++) {
		printf("[%d]", indexed*block_size + i);
	}
	printf("\n");
	for (int i = 0; i < block_size; i++) {
		printf("[%d]", i + probe*block_size);
		for (int j = 0; j < block_size; j++) {
			printf(" %d ", h_intersection[i*block_size + j]);// > 0? 1: 0);
		}
		printf("\n");
	}
	printf("==========================\n");

}

void print_result(Pair *pairs, int size) {
	printf("\n============ Similarity Join Result ============\n");
	for (int i = 0; i < size; i++) {
		printf("[%d, %d]:%.3f  ", pairs[i].set_x, pairs[i].set_y, pairs[i].similarity);
	}
	printf("\n================================================\n");
}
