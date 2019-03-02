/*
 * cuCompactor.h
 *
 *  Created on: 21/mag/2015
 *      Author: knotman
 */

#ifndef CUCOMPACTOR_H_
#define CUCOMPACTOR_H_

#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include "cuda_error_check.cu"

namespace cuCompactor {

#define warpSize (32)

__host__ __device__ int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

__device__ __inline__ int pow2i (int e){
	return 1<<e;
}


template <typename T,typename Predicate>
__global__ void computeBlockCounts(T* d_input,int length,int*d_BlockCounts,Predicate predicate){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < length){
		int pred = predicate(d_input[idx]);
		int BC=__syncthreads_count(pred);

		if(threadIdx.x==0){
			d_BlockCounts[blockIdx.x]=BC;
		}
	}
}



template <typename T,typename Predicate>
__global__ void compactK(T* d_input,int length, T* d_output,int* d_BlocksOffset,Predicate predicate ){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	extern __shared__ int warpTotals[];
	if(idx < length){
		int pred = predicate(d_input[idx]);
		int w_i = threadIdx.x/warpSize; //warp index
		int w_l = idx % warpSize;//thread index within a warp
		int t_m = INT_MAX >> (warpSize-w_l-1); //thread mask (ERROR IN THE PAPERminus one is required)

		int b	= __ballot(pred) & t_m; //balres = number whose ith bit isone if the ith's thread pred is true masked up to the current index in warp
		int t_u	= __popc(b); // popc count the number of bit one. simply count the number predicated true BEFORE MY INDEX



		if(w_l==warpSize-1){
			warpTotals[w_i]=t_u+pred;
		}


		__syncthreads();


		if(w_i==0 && w_l<blockDim.x/warpSize){
			int w_i_u=0;
			for(int j=0;j<=5;j++){
				int b_j =__ballot( warpTotals[w_l] & pow2i(j) ); //# of the ones in the j'th digit of the warp offsets
				w_i_u += (__popc(b_j & t_m)  ) << j;
				//printf("indice %i t_m=%i,j=%i,b_j=%i,w_i_u=%i\n",w_l,t_m,j,b_j,w_i_u);
			}
			warpTotals[w_l]=w_i_u;
		}

		__syncthreads();


		if(pred){
			d_output[t_u+warpTotals[w_i]+d_BlocksOffset[blockIdx.x]]= d_input[idx];

		}


	}
}

template <class T>
__global__  void printArray_GPU(T* hd_data, int size,int newline){
	int w=0;
	for(int i=0;i<size;i++){
		if(i%newline==0) {
			printf("\n%i -> ",w);
			w++;
		}
		printf("%i ",hd_data[i]);
	}
	printf("\n");
}

template <typename T,typename Predicate>
void compact(T* d_input,T* d_output,int length, Predicate predicate, int blockSize){
	int numBlocks = divup(length,blockSize);
	int* d_BlocksCount;
	int* d_BlocksOffset;
	CUDASAFECALL (cudaMalloc(&d_BlocksCount,sizeof(int)*numBlocks));
	CUDASAFECALL (cudaMalloc(&d_BlocksOffset,sizeof(int)*numBlocks));
	thrust::device_ptr<int> thrustPrt_bCount(d_BlocksCount);
	thrust::device_ptr<int> thrustPrt_bOffset(d_BlocksOffset);

	//phase1
	computeBlockCounts<<<numBlocks,blockSize>>>(d_input,length,d_BlocksCount,predicate);
	//phase2
	cudaDeviceSynchronize();
	thrust::exclusive_scan(thrustPrt_bCount, thrustPrt_bCount + numBlocks, thrustPrt_bOffset);
	//phase3
	cudaDeviceSynchronize();
	compactK<<<numBlocks,blockSize,sizeof(int)*(blockSize/warpSize)>>>(d_input,length,d_output,d_BlocksOffset,predicate);


	cudaFree(d_BlocksCount);
	cudaFree(d_BlocksOffset);

}

template <typename T,typename Predicate>
int compact2(T* d_input, T* d_output, int length, Predicate predicate, int blockSize, int numBlocks, int* d_BlocksCount, int* d_BlocksOffset){
	thrust::device_ptr<int> thrustPrt_bCount(d_BlocksCount);
	thrust::device_ptr<int> thrustPrt_bOffset(d_BlocksOffset);
	int lastSum[2];

	//phase1
	computeBlockCounts<<<numBlocks,blockSize>>>(d_input,length,d_BlocksCount,predicate);
	//phase2
	//cudaDeviceSynchronize();
	thrust::exclusive_scan(thrustPrt_bCount, thrustPrt_bCount + numBlocks, thrustPrt_bOffset);
	//phase3
	//cudaDeviceSynchronize();
	compactK<<<numBlocks,blockSize,sizeof(int)*(blockSize/warpSize)>>>(d_input,length,d_output,d_BlocksOffset,predicate);

	gpuAssert(cudaMemcpyAsync(d_BlocksCount + numBlocks, d_BlocksOffset + numBlocks - 1, sizeof(int), cudaMemcpyDeviceToDevice));
	gpuAssert(cudaMemcpyAsync(lastSum, d_BlocksCount + numBlocks - 1, 2*sizeof(int), cudaMemcpyDeviceToHost)); // copies only the last position

	return lastSum[0] + lastSum[1];
}

} /* namespace cuCompactor */
#endif /* CUCOMPACTOR_H_ */
