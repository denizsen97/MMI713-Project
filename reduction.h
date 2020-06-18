#include <thrust/device_vector.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_atomic_functions.h>

struct add_sums {

	float sum;
	float* oldranks;
	add_sums(thrust::device_vector<float>& o) {
		sum = 0;
		oldranks = thrust::raw_pointer_cast(o.data());
	}

	__device__
		void operator()(int index) {
		sum += oldranks[index];
	}
};




template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

__device__
float abs(float first, float second) {
	float a = first - second;
	return  a >= 0 ? a : -a;

}

#define BLOCK_SIZE 1024


template <unsigned int blockSize, bool takeDifference>
__global__ void reduce_kernel(float* g_newranks, float* g_oldranks, unsigned int offset, unsigned int n, int size) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;

	if (takeDifference) {
		while (i < size) {
			float a = g_newranks[i] - g_oldranks[i];
			sdata[tid] += a >= 0 ? a : -a;
			if (i + blockSize < size) {
				a = g_newranks[i + blockSize] - g_oldranks[i + blockSize];
				sdata[tid] += a >= 0 ? a : -a;
			}
			else break;
			i += gridSize;
		}
	}

	if (!takeDifference) {
		while (i < n) { sdata[tid] += g_oldranks[i] + g_oldranks[i + blockSize]; i += gridSize; }
	}

	__syncthreads();
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) warpReduce<blockSize>(sdata, tid);

	if (tid == 0) g_oldranks[blockIdx.x] = sdata[0];

}


__device__ void warpReduce(volatile float* sdata, unsigned int tid, unsigned int blockSize) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <bool takeDifference>
__global__ void next_reduce_kernel(float* g_newranks, float* g_oldranks, unsigned int offset, unsigned int n, int size) {
	extern __shared__ float sdata[];
	unsigned int blockSize = blockDim.x;
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;

	if (takeDifference) {
		while (i < n) {
			float a = g_newranks[i] - g_oldranks[i];
			sdata[tid] += a >= 0 ? a : -a;

			if (i + gridSize < n) {
				a = g_newranks[i + blockSize] - g_oldranks[i + blockSize];
				sdata[tid] += a >= 0 ? a : -a;
			}
			else break;
			i += gridSize;
		}
	}

	if (!takeDifference) {
		while (i < n) { 
			sdata[tid] += g_oldranks[i];
			if (i + gridSize < n) {
				sdata[tid] += g_oldranks[i + blockSize];
			}
			else break;
			i += gridSize; 
		}
	}

	__syncthreads();
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
	else if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	else if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	else if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	else if (tid < 32) warpReduce(sdata, tid, blockSize);

	if (tid == 0) g_oldranks[blockIdx.x] = sdata[0];

}




//should only be called with 1 thread
__global__
void addAll(float* sum, int* indices, float* newrankVector, float* oldrankVector, int sizeIndices, int offset, int rankSize) {

	extern __shared__ float sdata[1];

	sdata[0] = 0;

	for (int i = 0; i < sizeIndices; i++)
	{
		sdata[0] += oldrankVector[indices[i]];
	}

	for (int i = offset; i < rankSize; i++)
	{
		sdata[0] += abs(oldrankVector[i], newrankVector[i]);
	}
	sum[0] = sdata[0];

}



float reduce(thrust::device_vector<float>& newranks, thrust::device_vector<float>& oldranks) {
	long size = newranks.size();
	int blockSize = BLOCK_SIZE;

	//std::cout << "Size :" << size << std::endl;

	thrust::device_vector<int> sumIndices(0);
	//find the biggest power of two


	thrust::host_vector<cudaStream_t> streams(10);

	int offset = 0;
	int N = 1;
	int counter = 0;

	//	while (size - offset >= BLOCK_SIZE) {
	N = 1;
	while (offset + (N * 2) <= size) N *= 2;
	N *= 2;
	int gridSize = (N / (2 * blockSize)) > 0 ? (N / (2 * blockSize)) : 1;

	float* newRankRawPtr = thrust::raw_pointer_cast(&newranks[offset]);
	float* oldRankRawPtr = thrust::raw_pointer_cast(&oldranks[offset]);
	//std::cout << "N: " << N << " Offset: " << offset << " NewRankPtr: " << newRankRawPtr << " OldRankPtr: " << oldRankRawPtr << std::endl;
	sumIndices.push_back(offset);

	//cudaStreamCreate(&streams[counter]);

	reduce_kernel<BLOCK_SIZE, true> << <gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float) >> > (newRankRawPtr, oldRankRawPtr, offset, N, size);
	//std::cout << "Grid Size: " << gridSize << std::endl;
	auto cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "reduction 1 failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	while (1) {
		if (gridSize == 1) {
			offset += N;
			++counter;
			break;
		}
		else {
			int g = (int)(gridSize / (blockSize * 2)) > 0 ? (int)(gridSize / (blockSize * 2)) : 1;
			next_reduce_kernel<false> << <g, gridSize, gridSize * sizeof(float) >> > (newRankRawPtr, oldRankRawPtr, offset, gridSize, size);


			gridSize = g;
			offset += N;
			++counter;
		}
	}
	/*
	
	if (gridSize == 1) {
		offset += N;
		++counter;
		//break;
	}
	else {
		int g = (int)(gridSize / (blockSize * 2)) > 0 ? (int)(gridSize / (blockSize * 2)) : 1;
		next_reduce_kernel<false> << <g, gridSize, gridSize * sizeof(float) >> > (newRankRawPtr, oldRankRawPtr, offset, gridSize, size);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "reduction 2 failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		gridSize = g;
		offset += N;
		++counter;
	}
	*/
	
	//}

	
	thrust::host_vector<float> h_rank(oldranks.begin(), oldranks.begin() + 1);
	
	return h_rank[0];
}
