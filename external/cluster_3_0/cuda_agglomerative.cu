/*
 =====================================================================================================================
 Name        : cuda_agglomerative.cu
 Author      : Vinay B Gavirangaswamy
 Version     : 1.0
 Copyright   :  This file is part of application to do "Ensemble Clustering Analysis on CUDA".

    			"Ensemble Clustering Analysis on CUDA" is free software: you can redistribute it and/or modify
    			it under the terms of the GNU General Public License as published by the Free Software Foundation,
    			either version 3 of the License, or (at your option) any later version.

    			"Ensemble Clustering Analysis on CUDA" is distributed in the hope that it will be useful, but
    			WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    			PARTICULAR PURPOSE.  See the GNU General Public License for more details.

    			You should have received a copy of the GNU General Public License along with
    			"Ensemble Cluster Analysis on CUDA".  If not, see <http://www.gnu.org/licenses/>.

 Description :
 =====================================================================================================================
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cassert>
#include <math_functions.h>

#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <helper_string.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include "common/clustLib.h"
#include "common/wrapper.h"
#include "common/wrapperFuncs.h"
#include "common/distCalcMthds.h"
#include "common/fibo.h"
#include "mat.h"


struct mask_functor
{


	mask_functor() {}

  __host__ __device__
  int operator()(const float& x) const  {
	  return (x != 0)? 1 : 0;
  }
};



//texture<float, 1, cudaReadModeElementType> texData;
//texture<float, 1, cudaReadModeElementType> texWeight;
//texture<int, 1, cudaReadModeElementType> texMask;

#ifndef MIN
#define MIN(x,y) ((x <= y) ? x : y)
#endif

#ifndef MIN_IDX
#define MIN_IDX(x,y, idx_x, idx_y) ((x <= y) ? idx_x : idx_y)
#endif

#ifndef MAX
#define MAX(x,y) ((x >= y) ? x : y)
#endif

#if (__CUDA_ARCH__ < 200)
#define int_mult(x,y)	__mul24(x,y)
#else
#define int_mult(x,y)	x*y
#endif



/* ********************************************************************** */

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

static __device__ void getNumBlocksAndThreadsDeviceNN(/*int whichKernel,*/ int n, int maxBlocks,
	int maxThreads, int &blocks, int &threads, int *maxGridSize, int *maxThreadsPerBlock) {

	threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
	blocks = (n + (threads * 2 - 1)) / (threads * 2);

	blocks = MIN(maxBlocks, blocks);

	if ((float) threads * blocks
			> (float) *maxGridSize * *maxThreadsPerBlock) {
		printf("n is too large, please choose a smaller number!\n");
	}

	if (blocks > *maxGridSize) {
		printf(
				"Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
				blocks, maxGridSize, threads * 2, threads);

		blocks /= 2;
		threads *= 2;
	}

//	threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
//	blocks = (n + threads - 1) / threads;
}
/* ********************************************************************* */

//static __device__ bool isPow2(unsigned int x) {
//	return ((x & (x - 1)) == 0);
//}
//
//static __device__ unsigned int nextPow2(unsigned int x) {
//	--x;
//	x |= x >> 1;
//	x |= x >> 2;
//	x |= x >> 4;
//	x |= x >> 8;
//	x |= x >> 16;
//	return ++x;
//}
//
//static bool isPow2Host(unsigned int x) {
//	return ((x & (x - 1)) == 0);
//}
//
//static unsigned int nextPow2Host(unsigned int x) {
//	--x;
//	x |= x >> 1;
//	x |= x >> 2;
//	x |= x >> 4;
//	x |= x >> 8;
//	x |= x >> 16;
//	return ++x;
//}

/* ******************************************************************** */

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory {
	__device__ inline operator T *() {
		extern __shared__ int __smem[];
		return (T *) __smem;
	}

	__device__ inline operator const T *() const {
		extern __shared__ int __smem[];
		return (T *) __smem;
	}
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double> {
	__device__ inline operator double *() {
		extern __shared__ double __smem_d[];
		return (double *) __smem_d;
	}

	__device__ inline operator const double *() const {
		extern __shared__ double __smem_d[];
		return (double *) __smem_d;
	}
};

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// We set threads / block to the minimum of maxThreads and
////////////////////////////////////////////////////////////////////////////////
//static void getNumBlocksAndThreadsHost(/*int whichKernel,*/int n, int maxBlocks,
//		int maxThreads, int &blocks, int &threads, int *maxGridSize,
//		int *maxThreadsPerBlock) {
//
//	threads = (n < maxThreads) ? nextPow2Host(n) : maxThreads;
//	blocks = (n + threads - 1) / threads;
//}

//static void getNumBlocksAndThreadsReduceMin6Host(int whichKernel, int n,
//		int maxBlocks, int maxThreads, int &blocks, int &threads,
//		int *maxGridSize, int *maxThreadsPerBlock) {
//
////	get device capability, to avoid block/grid size exceed the upper bound
////	cudaDeviceProp prop;
////	int device;
////	cudaGetDevice(&device);
////	cudaGetDeviceProperties(&prop, device);
//
//	if (whichKernel < 3) {
//		threads = (n < maxThreads) ? nextPow2Host(n) : maxThreads;
//		blocks = (n + threads - 1) / threads;
//	} else {
//		threads = (n < maxThreads * 2) ? nextPow2Host((n + 1) / 2) : maxThreads;
//		blocks = (n + (threads * 2 - 1)) / (threads * 2);
//	}
//
//	if ((float) threads * blocks
//			> (float) (*maxGridSize) * (*maxThreadsPerBlock)) {
//		DEBUG_PRINT("n (%d) is too large, please choose a smaller number!\n", n);
//	}
//
//	if (blocks > (*maxGridSize)) {
//		DEBUG_PRINT(
//				"Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
//				blocks, (*maxGridSize), threads * 2, threads);
//
//		blocks /= 2;
//		threads *= 2;
//	}
//
//	if (whichKernel == 6) {
//		blocks = MIN(maxBlocks, blocks);
//	}
//}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
//static __device__ void getNumBlocksAndThreadsReduceMin6(int whichKernel, int n,
//		int maxBlocks, int maxThreads, int &blocks, int &threads,
//		int *maxGridSize, int *maxThreadsPerBlock) {
//
////	get device capability, to avoid block/grid size exceed the upper bound
////	cudaDeviceProp prop;
////	int device;
////	cudaGetDevice(&device);
////	cudaGetDeviceProperties(&prop, device);
//
//	if (whichKernel < 3) {
//		threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
//		blocks = (n + threads - 1) / threads;
//	} else {
//		threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
//		blocks = (n + (threads * 2 - 1)) / (threads * 2);
//	}
//
//	if ((float) threads * blocks
//			> (float) (*maxGridSize) * (*maxThreadsPerBlock)) {
//		DEBUG_PRINT("n (%d) is too large, please choose a smaller number!\n", n);
//	}
//
//	if (blocks > (*maxGridSize)) {
//		DEBUG_PRINT(
//				"Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
//				blocks, (*maxGridSize), threads * 2, threads);
//
//		blocks /= 2;
//		threads *= 2;
//	}
//
//	if (whichKernel == 6) {
//		blocks = MIN(maxBlocks, blocks);
//	}
//}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// We set threads / block to the minimum of maxThreads and
////////////////////////////////////////////////////////////////////////////////
//static __device__ void getNumBlocksAndThreadsDevice(/*int whichKernel,*/int n,
//		int maxBlocks, int maxThreads, int &blocks, int &threads,
//		int *maxGridSize, int *maxThreadsPerBlock) {
//
//	threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
//	blocks = (n + threads - 1) / threads;
//}
/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		DEBUG_PRINT("Error %s \n",											\
				cudaGetErrorString(_m_cudaStat));							\
		exit(1);															\
	} }

/* ********************************************************************** */
static __device__ void getNumBlocksAndThreadsDevice(int whichKernel, int n, int maxBlocks,
	int maxThreads, int &blocks, int &threads, int *maxGridSize, int *maxThreadsPerBlock) {


//	get device capability, to avoid block/grid size exceed the upper bound
//	cudaDeviceProp prop;
//	int device;
//	cudaGetDevice(&device);
//	cudaGetDeviceProperties(&prop, device);

	if (whichKernel < 3) {
		threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
		blocks = (n + threads - 1) / threads;
	} else {
		threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
		blocks = (n + (threads * 2 - 1)) / (threads * 2);
	}

	if ((float) threads * blocks
			> (float) (*maxGridSize) * (*maxThreadsPerBlock)) {
		printf("n (%d) is too large, please choose a smaller number!\n", n);
	}

	if (blocks > (*maxGridSize)) {
		printf(
				"Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
				blocks, (*maxGridSize), threads * 2, threads);

		blocks /= 2;
		threads *= 2;
	}

	if (whichKernel == 6) {
		blocks = MIN(maxBlocks, blocks);
	}
}


/* ********************************************************************** */

static const float* sortdata = NULL; /* used in the quicksort algorithm */

/* ---------------------------------------------------------------------- */

static
int compare(const void* a, const void* b)
/* Helper function for sort. Previously, this was a nested function under
 * sort, which is not allowed under ANSI C.
 */
{ const int i1 = *(const int*)a;
  const int i2 = *(const int*)b;
  const float term1 = sortdata[i1];
  const float term2 = sortdata[i2];
  if (term1 < term2) return -1;
  if (term1 > term2) return +1;
  return 0;
}

/* ---------------------------------------------------------------------- */

void sort(int n, const float data[], int index[])
/* Sets up an index table given the data, such that data[index[]] is in
 * increasing order. Sorting is done on the indices; the array data
 * is unchanged.
 */
{ int i;
  sortdata = data;
  for (i = 0; i < n; i++) index[i] = i;
  qsort(index, n, sizeof(int), compare);
}

/* ********************************************************************* */
static float* getrank (int n, float data[])
/* Calculates the ranks of the elements in the array data. Two elements with
 * the same value get the same rank, equal to the average of the ranks had the
 * elements different values. The ranks are returned as a newly allocated
 * array that should be freed by the calling routine. If getrank fails due to
 * a memory allocation error, it returns NULL.
 */
{ int i;
  float* rank;
  int* index;
  rank = (float*) malloc(n*sizeof(float));
  if (!rank) return NULL;
  index = (int*)malloc(n*sizeof(int));
  if (!index)
  { free(rank);
    return NULL;
  }
  /* Call sort to get an index table */
  sort (n, data, index);
  /* Build a rank table */
  for (i = 0; i < n; i++) rank[index[i]] = i;
  /* Fix for equal ranks */
  i = 0;
  while (i < n)
  { int m;
    float value = data[index[i]];
    int j = i + 1;
    while (j < n && data[index[j]] == value) j++;
    m = j - i; /* number of equal ranks found */
    value = rank[index[i]] + (m-1)/2.;
    for (j = i; j < i + m; j++) rank[index[j]] = value;
    i += m;
  }
  free (index);
  return rank;
}

/* ******************************************************************** */

static
int nodecompare(const void* a, const void* b)
/* Helper function for qsort. */
{ const Node* node1 = (const Node*)a;
  const Node* node2 = (const Node*)b;
  const float term1 = node1->distance;
  const float term2 = node2->distance;
  if (term1 < term2) return -1;
  if (term1 > term2) return +1;
  return 0;
}
/* ******************************************************************** */

/*
 This version finds Nearest Neighbor.  This reduces the overall
 cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
 (Brent's Theorem optimization)

 Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
 In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
 If blockSize > 32, allocate blockSize*sizeof(T) bytes.
 */
template<class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduceMin6(T *g_idata, cudaTextureObject_t  texIdx, T *g_odata, int *g_oIdxs, unsigned int n) {

	T *sdata = SharedMemory<T>();
	int *sdataIdx = ((int *)sdata) + blockSize;

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;


	T myMin = FLT_MAX;
	int myMinIdx = -1;
	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n) {
//		printf("\n[%d] %d %f",i, g_idxs[i], g_idata[g_idxs[i]]);
		myMinIdx  = MIN_IDX(g_idata[tex1Dfetch<int>(texIdx, i)], myMin, tex1Dfetch<int>(texIdx, i), myMinIdx);
		myMin = MIN(g_idata[tex1Dfetch<int>(texIdx, i)], myMin);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n){
			//myMin += g_idata[i + blockSize];
			myMinIdx  = MIN_IDX(g_idata[tex1Dfetch<int>(texIdx, i + blockSize)], myMin, tex1Dfetch<int>(texIdx, i + blockSize), myMinIdx);
			myMin = MIN(g_idata[tex1Dfetch<int>(texIdx, i + blockSize)], myMin);
		}


//		printf("\n%d %f", myMinIdx, myMin);
		i += gridSize;
	}


	// each thread puts its local sum into shared memory
	sdata[tid] = myMin;
	sdataIdx[tid] = myMinIdx;

	__syncthreads();

	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256)) {

		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 256], myMin, sdataIdx[tid + 256], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 256], myMin);

	}

	__syncthreads();

	if ((blockSize >= 256) && (tid < 128)) {

		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 128], myMin, sdataIdx[tid + 128], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 128], myMin);


	}

	__syncthreads();

	if ((blockSize >= 128) && (tid < 64)) {

		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 64], myMin, sdataIdx[tid + 64], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 64], myMin);
	}

	__syncthreads();

#if (__CUDA_ARCH__ >= 300 )
	if (tid < 32) {
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64){

			myMinIdx = MIN_IDX(sdata[tid + 32], myMin, sdataIdx[tid + 32], myMinIdx);
			myMin = MIN(sdata[tid + 32], myMin);
		}
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2) {

			int tempMyMinIdx = __shfl_down(myMinIdx, offset);
			float tempMyMin = __shfl_down(myMin, offset);

			myMinIdx = MIN_IDX(tempMyMin, myMin, tempMyMinIdx , myMinIdx);
			myMin = MIN(tempMyMin, myMin);

		}

	}
#else
	// fully unroll reduction within a single warp
	if ((blockSize >= 64) && (tid < 32))
	{
		//sdata[tid] = myMin = myMin + sdata[tid + 32];
		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 32], myMin, sdataIdx[tid + 32], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 32], myMin);
	}

	__syncthreads();

	if ((blockSize >= 32) && (tid < 16))
	{

		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 16], myMin, sdataIdx[tid + 16], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 16], myMin);
	}

	__syncthreads();

	if ((blockSize >= 16) && (tid < 8))
	{

		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 8], myMin, sdataIdx[tid + 8], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 8], myMin);
	}

	__syncthreads();

	if ((blockSize >= 8) && (tid < 4))
	{

		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 4], myMin, sdataIdx[tid + 4], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 4], myMin);
	}

	__syncthreads();

	if ((blockSize >= 4) && (tid < 2))
	{

		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 2], myMin, sdataIdx[tid + 2], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 2], myMin);
	}

	__syncthreads();

	if ((blockSize >= 2) && ( tid < 1))
	{

		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 1], myMin, sdataIdx[tid + 1], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 1], myMin);
	}

	__syncthreads();
#endif

	__syncthreads();
	// write result for this block to global mem
	if (tid == 0){
		g_odata[blockIdx.x] = myMin;
		g_oIdxs[blockIdx.x] = myMinIdx;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template<class T>
__device__ void reduceMin(int size, int threads, int blocks, int whichKernel, T *d_idata,
		T *d_odata, cudaTextureObject_t  texIdx, int *oIdxs) {
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize =
	        (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	smemSize += threads*sizeof(int);

	if (isPow2(size)) {
		switch (threads) {
		case 512:
			reduceMin6<T, 512, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 256:
			reduceMin6<T, 256, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 128:
			reduceMin6<T, 128, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 64:
			reduceMin6<T, 64, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 32:
			reduceMin6<T, 32, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 16:
			reduceMin6<T, 16, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 8:
			reduceMin6<T, 8, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 4:
			reduceMin6<T, 4, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 2:
			reduceMin6<T, 2, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 1:
			reduceMin6<T, 1, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;
		}
	} else {
		switch (threads) {
		case 512:
			reduceMin6<T, 512, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 256:
			reduceMin6<T, 256, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 128:
			reduceMin6<T, 128, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 64:
			reduceMin6<T, 64, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 32:
			reduceMin6<T, 32, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 16:
			reduceMin6<T, 16, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 8:
			reduceMin6<T, 8, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 4:
			reduceMin6<T, 4, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 2:
			reduceMin6<T, 2, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;

		case 1:
			reduceMin6<T, 1, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, texIdx,
					d_odata, oIdxs, size);
			break;
		}
	}

}


/* ******************************************************************* */
// Instantiate the reduction function for 3 types
template void
reduceMin<float>(int size, int threads, int blocks, int whichKernel, float *d_idata,
		float *d_odata, cudaTextureObject_t  texIdx, int *oIdxs);
/* ******************************************************************* */

static __device__ void find_nearest_neighbor(int *maxGridSize,int *maxThreadsPerBlock, int n, float *distmatrix, cudaTextureObject_t  texIdx, int *row, int* col, float *min) {


	cudaError_t err;

	int maxThreads = THREADS_PER_BLOCK_256;  // number of threads per block

	int maxBlocks = 64;
	int numBlocks = 0;
	int numThreads = 0;

	int threads;
	int gridSize;

	float* d_out = NULL;
	int *d_oIdxs = NULL;
	if(n >= 64){
		getNumBlocksAndThreadsDeviceNN(TRI_COUNT(n-2), maxBlocks, maxThreads, numBlocks,numThreads, maxGridSize, maxThreadsPerBlock);

		cudaMalloc((void **) &d_out, numBlocks * sizeof(float));

		if(!d_out && cudaSuccess != cudaGetLastError()){
			printf("\n cudaMalloc failed for d_out in thread id %d \n", blockIdx.x * blockDim.x + threadIdx.x);
			return;
		}

		cudaMalloc((void **) &d_oIdxs, numBlocks * sizeof(int));

		if(!d_oIdxs && cudaSuccess != cudaGetLastError()){

			printf("\n cudaMalloc failed for d_out in thread id %d \n", blockIdx.x * blockDim.x + threadIdx.x);
			return;
		}

		//		for(int count=TRI_COUNT(n); count>=0;count--) printf("\n%d %f", count, distmatrix[count]);


		reduceMin<float>(TRI_COUNT(n-2), numThreads, numBlocks, 6, distmatrix, d_out, texIdx, d_oIdxs);

		if (cudaSuccess != cudaGetLastError()) {
			return;
		}

		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

		int min_idx = -1;
		float distance = FLT_MAX;

		for (int i = 0; i < numBlocks; i++) {
			min_idx = MIN_IDX(d_out[i], distance, d_oIdxs[i], min_idx);
			distance = MIN(d_out[i], distance);
//			printf("\n%d %f\n", min_idx, distance);
		}
		*row = floorf((-1 + sqrtf(1 - (4 * 1 * (-min_idx * 2)))) / 2);
		*col = min_idx - ((*row*(*row + 1)) / 2);
		*min = distance;
		cudaFree(d_out);
		cudaFree(d_oIdxs);
	}
	else{
		int i, j;
		float temp;
		float distance = distmatrix[TRI_COUNT(1)+0];
		int ip = 1;
		int jp = 0;

//		  printf("\n----------------------------------------------------------------------------------------\n ");
		for (i = 1; i < n; i++)
		{ for (j = 0; j < i; j++)
		{ temp = distmatrix[TRI_COUNT(i)+j];
//		     printf("%f[%d], ",temp, TRI_COUNT(i)+j);
		if (temp<distance)
		{ distance = temp;
		ip = i;
		jp = j;
		}
		}
		}

		*row = ip;
		*col = jp;
		*min = distance;
	}

}

/* ******************************************************************** */

void cuttree (int nelements, Node* tree, int nclusters, int *clusterid)

/*
Purpose
=======

The cuttree routine takes the output of a hierarchical clustering routine, and
divides the elements in the tree structure into clusters based on the
hierarchical clustering result. The number of clusters is specified by the user.

Arguments
=========

nelements      (input) int
The number of elements that were clustered.

tree           (input) Node[nelements-1]
The clustering solution. Each node in the array describes one linking event,
with tree[i].left and tree[i].right representig the elements that were joined.
The original elements are numbered 0..nelements-1, nodes are numbered
-1..-(nelements-1).

nclusters      (input) int
The number of clusters to be formed.

clusterid      (output) int[nelements]
The number of the cluster to which each element was assigned. Space for this
array should be allocated before calling the cuttree routine. If a memory
error occured, all elements in clusterid are set to -1.

========================================================================
*/
{ int i, j, k;
  int icluster = 0;
  const int n = nelements-nclusters; /* number of nodes to join */
  int* nodeid;
  for (i = nelements-2; i >= n; i--)
  { k = tree[i].left;
    if (k>=0)
    { clusterid[k] = icluster;
      icluster++;
    }
    k = tree[i].right;
    if (k>=0)
    { clusterid[k] = icluster;
      icluster++;
    }
  }
  nodeid = (int*) malloc(n*sizeof(int));
  if(!nodeid)
  { for (i = 0; i < nelements; i++) clusterid[i] = -1;
    return;
  }
  for (i = 0; i < n; i++) nodeid[i] = -1;
  for (i = n-1; i >= 0; i--)
  { if(nodeid[i]<0)
    { j = icluster;
      nodeid[i] = j;
      icluster++;
    }
    else j = nodeid[i];
    k = tree[i].left;
    if (k<0) nodeid[-k-1] = j; else clusterid[k] = j;
    k = tree[i].right;
    if (k<0) nodeid[-k-1] = j; else clusterid[k] = j;
  }
//  free(nodeid);
  return;
}

/* ******************************************************************* */
__global__ void mlFixDistRow(int col, int row, float* distmatrix) {
	/* Fix the distances */
//	for (int j = 0; j < col; j++)
	for(int j = blockIdx.x * blockDim.x + threadIdx.x;j < col;j += blockDim.x * gridDim.x)
		distmatrix[TRI_COUNT(col) + j] = MAX(distmatrix[TRI_COUNT(row) + j], distmatrix[TRI_COUNT(col) + j]);

}

__global__ void mlFixDistCol(int col, int row, float* distmatrix) {
//	for (int j = col + 1; j < row; j++)
	for(int j = blockIdx.x * blockDim.x + threadIdx.x+col+1;
					j < row;
					j += blockDim.x * gridDim.x)
		distmatrix[TRI_COUNT(j) + col] = MAX(distmatrix[TRI_COUNT(row) + j],
				distmatrix[TRI_COUNT(j) + col]);
}

__global__ void mlFixDistRest(int row, int n, int col, float* distmatrix) {
//	for (int j = row + 1; j < n; j++)
	for(int j = blockIdx.x * blockDim.x + threadIdx.x+row+1;
						j < n;
						j += blockDim.x * gridDim.x)
		distmatrix[TRI_COUNT(j) + col] = MAX(distmatrix[TRI_COUNT(j) + row],
				distmatrix[TRI_COUNT(j) + col]);
}

__global__ void mlDelObjDistRow(int row, int n, float* distmatrix) {
//	for (int j = 0; j < row; j++)
	for(int j = blockIdx.x * blockDim.x + threadIdx.x;
							j < row;
							j += blockDim.x * gridDim.x)
		distmatrix[TRI_COUNT(row) + j] = distmatrix[TRI_COUNT(n-1) + j];
}

__global__ void mlDelObjDistCol(int row, int n, float* distmatrix) {
//	for (int j = row + 1; j < n - 1; j++)
		for(int j = blockIdx.x * blockDim.x + threadIdx.x+row+1;
									j < n-1;
									j += blockDim.x * gridDim.x)
		distmatrix[TRI_COUNT(j) + row] = distmatrix[TRI_COUNT(n-1) + j];
}



static __global__ void pmlcluster(int nelements, int ncolumns,
		  float* distmatrix, char dist, int transpose, Node* result, int *clusterId, cudaTextureObject_t  texIdx, int *maxGridSize,int *maxThreadsPerBlock){


	cudaError_t err;

	int maxThreads = THREADS_PER_BLOCK_256;  // number of threads per block

	int maxBlocks = 64;
	int numBlocks = 0;
	int numThreads = 0;

	int threads;
	int gridSize;

	int row;
	int col;
	float min;

	for (int n = nelements; n > 1; n--){

		find_nearest_neighbor(maxGridSize, maxThreadsPerBlock, n,distmatrix,texIdx, &row, &col,&min);

		result[nelements-n].distance = min;

//		printf("\n[%d] %f -------->[%dx%d]\n",n, result[nelements-n].distance, row, col);

//	    /* Fix the distances */
//	    for (int j = 0; j < col; j++)
//	      distmatrix[TRI_COUNT(col)+ j] = MAX(distmatrix[TRI_COUNT(row) + j],distmatrix[TRI_COUNT(col)+j]);
//	    for (int j = col+1; j < row; j++)
//	      distmatrix[TRI_COUNT(j)+ col] = MAX(distmatrix[TRI_COUNT(row)+j],distmatrix[TRI_COUNT(j)+col]);
//	    for (int j = row+1; j < n; j++)
//	      distmatrix[TRI_COUNT(j)+col] = MAX(distmatrix[TRI_COUNT(j)+ row],distmatrix[TRI_COUNT(j)+ col]);
//
//	    for (int j = 0; j < row; j++) distmatrix[TRI_COUNT(row)+j] = distmatrix[TRI_COUNT(n-1)+j];
//	    for (int j = row+1; j < n-1; j++) distmatrix[TRI_COUNT(j)+row] = distmatrix[TRI_COUNT(n-1)+j];


		/* Fix the distances */
		threads = (col< maxThreads) ? nextPow2(col) : maxThreads;
		gridSize = ( col + threads - 1) / threads;

		//		printf("\n col = %d threads = %d gridSize = %d ", col, threads, gridSize);

		if(threads > 0 && gridSize > 0)mlFixDistRow<<<gridSize, threads>>>(col, row, distmatrix);
		if (cudaSuccess != (err=cudaGetLastError())) {

			printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
			return;
		}

		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

		threads = (row- (col+1) < maxThreads) ? nextPow2(row- col+1) : maxThreads;
		gridSize = (row- (col+1) + threads - 1) / threads;

		//		printf("\n row-col+1 = %d threads = %d gridSize = %d ", row-col+1, threads, gridSize);

		if(threads > 0 && gridSize > 0) mlFixDistCol<<<gridSize, threads>>>(col, row, distmatrix);
		if (cudaSuccess != (err=cudaGetLastError())) {

			printf("\n@fixDistCol kernel error: %s\n", cudaGetErrorString(err));
			return;
		}

		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

		threads = (n-row+1 < maxThreads) ? nextPow2(n-row+1) : maxThreads;
		gridSize = (n-row+1 + threads - 1) / threads;

		//		 printf("\n n-row+1 = %d threads = %d gridSize = %d ", n-row+1, threads, gridSize);

		if(threads > 0 && gridSize > 0) mlFixDistRest<<<gridSize, threads>>>(row, n, col, distmatrix);

		if (cudaSuccess != (err=cudaGetLastError())) {

			printf("\n@fixDistCol kernel error: %s\n", cudaGetErrorString(err));
			return;
		}

		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

		threads = (row < maxThreads) ? nextPow2(row) : maxThreads;
		gridSize = (row + threads - 1) / threads;

		if(threads > 0 && gridSize > 0)
			mlDelObjDistRow<<<gridSize, threads>>>(row, n, distmatrix);
		if (cudaSuccess != (err=cudaGetLastError())) {

			printf("\n@fixDistCol kernel error: %s\n", cudaGetErrorString(err));
			return;
		}

		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}
		threads = (n -row < maxThreads) ? nextPow2(n-row) : maxThreads;
		gridSize = (n- row + threads - 1) / threads;

		if(threads > 0 && gridSize > 0)

		mlDelObjDistCol<<<gridSize, threads>>>(row, n, distmatrix);
		if (cudaSuccess != (err=cudaGetLastError())) {

			printf("\n@fixDistCol kernel error: %s\n", cudaGetErrorString(err));
			return;
		}

		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

//		if(row >= 0 && col>= 0){
			/* Update clusterids */
			result[nelements-n].left = clusterId[row];
			result[nelements-n].right = clusterId[col];
			clusterId[col] = n-nelements-1;
			clusterId[row] = clusterId[n-1];
//		}

	}

}


/* ******************************************************************* */
__global__ void clFixDistRow1(int row, const int nnodes, int inode, float* distmatrix) {
	//		data[is] = data[nnodes-inode];
	//		mask[is] = mask[nnodes-inode];
	/* Fix the distances */
//	for (int i = 0; i < row; i++)
	for(int i = blockIdx.x * blockDim.x + threadIdx.x;
										i < row;
										i += blockDim.x * gridDim.x)
		distmatrix[TRI_COUNT(row) + i] =
				distmatrix[TRI_COUNT(nnodes-inode) + i];
}

__global__ void clFixDistRow2(int row, const int nnodes, int inode, float* distmatrix) {
//	for (int i = row + 1; i < nnodes - inode; i++)
	for(int i = blockIdx.x * blockDim.x + threadIdx.x+row+1;
										i < nnodes-inode;
										i += blockDim.x * gridDim.x)
		distmatrix[TRI_COUNT(i) + row] =
				distmatrix[TRI_COUNT(nnodes-inode) + i];
}

__global__ void clFixDistCol1(int col, char dist, const int ndata, const int nelements,
		int transpose, float* distmatrix, float* data, int* mask,
		cudaTextureObject_t texWt) {
//	for (int i = 0; i < col; i++)
	for(int i = blockIdx.x * blockDim.x + threadIdx.x;
										i < col;
										i += blockDim.x * gridDim.x)
		distmatrix[TRI_COUNT(col) + i] = calcDistMetricGPU(dist, ndata,
				nelements, nelements, data, data, texWt, col, i,
				transpose);
}

__global__ void clFixDistCol2(int col, const int nnodes, int inode, char dist,
		const int ndata, const int nelements, int transpose, float* distmatrix,
		float* data, int* mask, cudaTextureObject_t texWt) {
	//metric(ndata,data,data,mask,mask,weight,js,i,0);
//	for (int i = col + 1; i < nnodes - inode; i++)
	for(int i = blockIdx.x * blockDim.x + threadIdx.x+col+1;
										i < nnodes-inode;
										i += blockDim.x * gridDim.x)
		distmatrix[TRI_COUNT(i) + col] = calcDistMetricGPU(dist, ndata,
				nelements, nelements, data, data, texWt, col, i,
				transpose);
}

__global__ void clMakeNewNode(const int ndata, int col, int row, float* data, int* mask) {
	/* Make node js the new node */
//	for (int i = 0; i < ndata; i++)
		for(int i = blockIdx.x * blockDim.x + threadIdx.x;
												i < ndata;
												i += blockDim.x * gridDim.x){
		data[col * ndata + i] = data[col * ndata + i] * mask[col * ndata + i]
				+ data[row * ndata + i] * mask[row * ndata + i];
		mask[col * ndata + i] += mask[row * ndata + i];
		if (mask[col * ndata + i])
			data[col * ndata + i] /= mask[col * ndata + i];
	}
}

__global__  void myMemcpyDatanMask(const int ndata, int row, const int nnodes, int inode,
		float* data, int* mask) {
	//		memcpy(&data[row], &data[nnodes-inode], ndata*sizeof(float));
	//		memcpy(&mask[row], &mask[nnodes-inode], ndata*sizeof(int));
//	for (int i = 0; i < ndata; i++) {
	for(int i = blockIdx.x * blockDim.x + threadIdx.x;
													i < ndata;
													i += blockDim.x * gridDim.x){
		data[row * ndata + i] = data[(nnodes - inode) * ndata + i];
		mask[row * ndata + i] = mask[(nnodes - inode) * ndata + i];
	}
}

/* ******************************************************************* */

/*

Purpose
=======

The pslcluster2 routine performs single-linkage hierarchical clustering, using
either the distance matrix directly, if available, or by calculating the
distances from the data array. This implementation is based on the
algorithm, described in:
Peter Willett, (1989). Efficiency of Hierarchic Agglomerative clustering
using the ICL Distributed Array Processor. The Journal of Documentation,
Vol. 45, No. 1, March 1989, pp 1-24
and modified to suit cuda architecture with authors modifications incorporated.
TODO: <<Write other documentation>>

Arguments
=========

nrows     (input) int
The number of rows in the gene expression data matrix, equal to the number of
genes.

ncolumns  (input) int
The number of columns in the gene expression data matrix, equal to the number of
microarrays.

data       (input) double[nrows][ncolumns]
The array containing the gene expression data.

mask       (input) int[nrows][ncolumns]
This array shows which data values are missing. If
mask[i][j] == 0, then data[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance. The length of this vector
is ncolumns if genes are being clustered, and nrows if microarrays are being
clustered.

transpose  (input) int
If transpose==0, the rows of the matrix are clustered. Otherwise, columns
of the matrix are clustered.

dist       (input) char
Defines which distance measure is used, as given by the table:
dist=='e': Euclidean distance
dist=='b': City-block distance
dist=='c': correlation
dist=='a': absolute value of the correlation
dist=='u': uncentered correlation
dist=='x': absolute uncentered correlation
dist=='s': Spearman's rank correlation
dist=='k': Kendall's tau
For other values of dist, the default (Euclidean distance) is used.

distmatrix (input) float*
The distance matrix.

Return value
============
result

A pointer to a newly allocated array of Node structs, describing the
hierarchical clustering solution consisting of nelements-1 nodes. Depending on
whether object (rows) or microarrays (columns) were clustered, nelements is
equal to nrows or ncolumns. See src/clustLib.h for a description of the Node
structure.


========================================================================
*/

static __global__ void pclcluster(int nrows, int ncolumns, float* data, int* mask,
		cudaTextureObject_t texWt, float* distmatrix, char dist, int transpose, Node* result, int *clusterId, cudaTextureObject_t  texIdx, int *maxGridSize,int *maxThreadsPerBlock){

	const int nelements = (transpose==0) ? nrows : ncolumns;
	int inode;
	const int ndata = transpose ? nrows : ncolumns;
	const int nnodes = nelements - 1;

	cudaError_t err;

	int maxThreads = THREADS_PER_BLOCK_256;  // number of threads per block

	int maxBlocks = 64;
	int numBlocks = 0;
	int numThreads = 0;

	int row;
	int col;
	float min;

	int threads;
	int gridSize;

//	for (int inode = 0, iteration=0; inode < nnodes, iteration < 11; inode++, iteration++)
	for (inode = 0; inode < nnodes; inode++)
	{ /* Find the pair with the shortest distance */
		find_nearest_neighbor(maxGridSize, maxThreadsPerBlock, nelements-inode,distmatrix,texIdx, &row, &col,&min);

		result[inode].distance = min;

		/* Make node col the new node */
		threads = (ndata< maxThreads) ? nextPow2(ndata) : maxThreads;
		gridSize = ( ndata + threads - 1) / threads;
//		printf("\nndata = %d threads = %d gridSize = %d ", ndata, threads, gridSize);
		if(threads > 0 && gridSize > 0)
		clMakeNewNode<<<gridSize, threads>>>(ndata, col, row, data, mask);
		if (cudaSuccess != (err=cudaGetLastError())) {
			printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
			return;
		}
		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

//		memcpy(&data[row], &data[nnodes-inode], ndata*sizeof(float));
//		memcpy(&mask[row], &mask[nnodes-inode], ndata*sizeof(int));
		threads = (ncolumns < maxThreads) ? nextPow2(ncolumns) : maxThreads;
		gridSize = ( ncolumns + threads - 1) / threads;
		//		printf("\n col = %d threads = %d gridSize = %d ", col, threads, gridSize);
		if(threads > 0 && gridSize > 0)
		myMemcpyDatanMask<<<gridSize, threads>>>(ndata, row, nnodes, inode, data, mask);

		if (cudaSuccess != (err=cudaGetLastError())) {
			printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
			return;
		}
		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

		/* Fix the distances */
		threads = (row< maxThreads) ? nextPow2(row) : maxThreads;
		gridSize = ( row + threads - 1) / threads;
		//		printf("\n col = %d threads = %d gridSize = %d ", col, threads, gridSize);
		if(threads > 0 && gridSize > 0)clFixDistRow1<<<gridSize, threads>>>(row, nnodes, inode, distmatrix);
		if (cudaSuccess != (err=cudaGetLastError())) {
			printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
			return;
		}
		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

		threads = (nnodes-inode -row+1 < maxThreads) ? nextPow2(nnodes-inode -row+1) : maxThreads;
		gridSize = ( nnodes-inode -row+1 + threads - 1) / threads;
		//		printf("\n col = %d threads = %d gridSize = %d ", col, threads, gridSize);
		if(threads > 0 && gridSize > 0)clFixDistRow2<<<gridSize, threads>>>(row, nnodes, inode, distmatrix);
		if (cudaSuccess != (err=cudaGetLastError())) {
			printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
			return;
		}
		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

		threads = (col < maxThreads) ? nextPow2(col) : maxThreads;
		gridSize = ( col + threads - 1) / threads;
		//		printf("\n col = %d threads = %d gridSize = %d ", col, threads, gridSize);
		if(threads > 0 && gridSize > 0)
			clFixDistCol1<<<gridSize, threads>>>(col, dist, ndata, nelements, transpose, distmatrix, data,
					mask, texWt);
		if (cudaSuccess != (err=cudaGetLastError())) {
			printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
			return;
		}
		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

		threads = (nnodes-inode -col+1 < maxThreads) ? nextPow2(nnodes-inode -col+1) : maxThreads;
		gridSize = ( nnodes-inode -col+1 + threads - 1) / threads;
		//		printf("\n col = %d threads = %d gridSize = %d ", col, threads, gridSize);
		if(threads > 0 && gridSize > 0)
			clFixDistCol2<<<gridSize, threads>>>(col, nnodes, inode, dist, ndata, nelements, transpose,
					distmatrix, data, mask, texWt);
		if (cudaSuccess != (err=cudaGetLastError())) {
			printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
			return;
		}
		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

		/* Update clusterids */
		result[inode].left = clusterId[col];
		result[inode].right = clusterId[row];
		clusterId[row] = clusterId[nnodes-inode];
		clusterId[col] = -inode-1;
	}

}
/* ******************************************************************* */
__global__ void wlFixDistCol1(int row, int col, char dist, const int ndata, const int nelements,
		int transpose, float* distmatrix, float* data, int* mask,
		cudaTextureObject_t texWt, int *number) {
//	for (int i = 0; i < col; i++)
	for(int i = blockIdx.x * blockDim.x + threadIdx.x;
										i < col;
										i += blockDim.x * gridDim.x){
		distmatrix[TRI_COUNT(col) + i] = calcDistMetricGPU(dist, ndata,
				nelements, nelements, data, data, texWt, col, i,
				transpose);
	distmatrix[TRI_COUNT(col) + i] = ((float)(number[row]*number[col])/(float)(number[row]+number[col]))* fabsf(distmatrix[TRI_COUNT(col)+i]*distmatrix[TRI_COUNT(col)+i]);
	}
}

__global__ void wlFixDistCol2(int row, int col, const int nnodes, int inode, char dist,
		const int ndata, const int nelements, int transpose, float* distmatrix,
		float* data, int* mask, cudaTextureObject_t texWt, int *number) {
	//metric(ndata,data,data,mask,mask,weight,js,i,0);
//	for (int i = col + 1; i < nnodes - inode; i++)
	for(int i = blockIdx.x * blockDim.x + threadIdx.x+col+1;
										i < nnodes-inode;
										i += blockDim.x * gridDim.x){
		distmatrix[TRI_COUNT(i) + col] = calcDistMetricGPU(dist, ndata,
				nelements, nelements, data, data, texWt, col, i,
				transpose);
		distmatrix[TRI_COUNT(i) + col] = ((float)(number[row]*number[col])/(float)(number[row]+number[col]))* fabsf(distmatrix[TRI_COUNT(i) + col]*distmatrix[TRI_COUNT(i) + col]);
	}
}



/* ******************************************************************* */


static __global__ void pwlcluster(int nrows, int ncolumns, float* data, int* mask,
		cudaTextureObject_t texWt, float* distmatrix, char dist, int transpose, Node* result, int *clusterId,cudaTextureObject_t texIdx, int *number, int *maxGridSize,int *maxThreadsPerBlock){

	const int nelements = (transpose==0) ? nrows : ncolumns;
	int inode;
	const int ndata = transpose ? nrows : ncolumns;
	const int nnodes = nelements - 1;

	cudaError_t err;

	int maxThreads = THREADS_PER_BLOCK_256;  // number of threads per block

	int maxBlocks = 64;
	int numBlocks = 0;
	int numThreads = 0;
	int sum;
	int row;
	int col;
	float min;

	int threads;
	int gridSize;

//	for (int inode = 0, iteration=0; inode < nnodes, iteration < 11; inode++, iteration++)
	for (inode = 0; inode < nnodes; inode++)
	{ /* Find the pair with the shortest distance */
		find_nearest_neighbor(maxGridSize, maxThreadsPerBlock, nelements-inode,distmatrix,texIdx, &row, &col,&min);

		result[inode].distance = min;

		sum = number[row] + number[col];

		/* Make node col the new node */
		threads = (ndata< maxThreads) ? nextPow2(ndata) : maxThreads;
		gridSize = ( ndata + threads - 1) / threads;
//		printf("\nndata = %d threads = %d gridSize = %d ", ndata, threads, gridSize);
		if(threads > 0 && gridSize > 0)
		clMakeNewNode<<<gridSize, threads>>>(ndata, col, row, data, mask);
		if (cudaSuccess != (err=cudaGetLastError())) {
			printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
			return;
		}
		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

//		memcpy(&data[row], &data[nnodes-inode], ndata*sizeof(float));
//		memcpy(&mask[row], &mask[nnodes-inode], ndata*sizeof(int));
		threads = (ncolumns < maxThreads) ? nextPow2(ncolumns) : maxThreads;
		gridSize = ( ncolumns + threads - 1) / threads;
		//		printf("\n col = %d threads = %d gridSize = %d ", col, threads, gridSize);
		if(threads > 0 && gridSize > 0)
		myMemcpyDatanMask<<<gridSize, threads>>>(ndata, row, nnodes, inode, data, mask);

		if (cudaSuccess != (err=cudaGetLastError())) {
			printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
			return;
		}
		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

		/* Fix the distances */
		threads = (row< maxThreads) ? nextPow2(row) : maxThreads;
		gridSize = ( row + threads - 1) / threads;
		//		printf("\n col = %d threads = %d gridSize = %d ", col, threads, gridSize);
		if(threads > 0 && gridSize > 0)clFixDistRow1<<<gridSize, threads>>>(row, nnodes, inode, distmatrix);
		if (cudaSuccess != (err=cudaGetLastError())) {
			printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
			return;
		}
		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

		threads = (nnodes-inode -row+1 < maxThreads) ? nextPow2(nnodes-inode -row+1) : maxThreads;
		gridSize = ( nnodes-inode -row+1 + threads - 1) / threads;
		//		printf("\n col = %d threads = %d gridSize = %d ", col, threads, gridSize);
		if(threads > 0 && gridSize > 0)clFixDistRow2<<<gridSize, threads>>>(row, nnodes, inode, distmatrix);
		if (cudaSuccess != (err=cudaGetLastError())) {
			printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
			return;
		}
		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

		threads = (col < maxThreads) ? nextPow2(col) : maxThreads;
		gridSize = ( col + threads - 1) / threads;
		//		printf("\n col = %d threads = %d gridSize = %d ", col, threads, gridSize);
		if(threads > 0 && gridSize > 0)
			wlFixDistCol1<<<gridSize, threads>>>(row,col, dist, ndata, nelements, transpose, distmatrix, data,
					mask, texWt, number);
		if (cudaSuccess != (err=cudaGetLastError())) {
			printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
			return;
		}
		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

		threads = (nnodes-inode -col+1 < maxThreads) ? nextPow2(nnodes-inode -col+1) : maxThreads;
		gridSize = ( nnodes-inode -col+1 + threads - 1) / threads;
		//		printf("\n col = %d threads = %d gridSize = %d ", col, threads, gridSize);
		if(threads > 0 && gridSize > 0)
			wlFixDistCol2<<<gridSize, threads>>>(row, col, nnodes, inode, dist, ndata, nelements, transpose,
					distmatrix, data, mask, texWt, number);
		if (cudaSuccess != (err=cudaGetLastError())) {
			printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
			return;
		}
		// wait for child to complete
		if (cudaSuccess != cudaDeviceSynchronize()) {
			return;
		}

		/* Update number of elements in the clusters */
		number[col] = sum;
		number[row] = number[nnodes-inode];

		/* Update clusterids */
		result[inode].left = clusterId[col];
		result[inode].right = clusterId[row];
		clusterId[row] = clusterId[nnodes-inode];
		clusterId[col] = -inode-1;
	}

}
/* ******************************************************************* */

__global__ void alFixDistCol1(int col, int row, int sum, float* distmatrix, int* number) {
//	for (int j = 0; j < col; j++) {
		for(int j = blockIdx.x * blockDim.x + threadIdx.x;
														j < col;
														j += blockDim.x * gridDim.x){
		distmatrix[TRI_COUNT(col) + j] = distmatrix[TRI_COUNT(row) + j]
				* number[row] + distmatrix[TRI_COUNT(col) + j] * number[col];
		distmatrix[TRI_COUNT(col) + j] /= sum;
	}
}

__global__ void alFixDistCol2(int col, int row, int sum, float* distmatrix, int* number) {
//	for (int j = col + 1; j < row; j++) {
	for(int j = blockIdx.x * blockDim.x + threadIdx.x+col + 1;
			j < row;
			j += blockDim.x * gridDim.x){

		distmatrix[TRI_COUNT(j) + col] = distmatrix[TRI_COUNT(row) + j]
				* number[row] + distmatrix[TRI_COUNT(j) + col] * number[col];
		distmatrix[TRI_COUNT(j) + col] /= sum;
	}
}

__global__ void alFixDistRow(int row, int n, int col, int sum, float* distmatrix,
		int* number) {
//	for (int j = row + 1; j < n; j++) {
	for(int j = blockIdx.x * blockDim.x + threadIdx.x+row + 1;
				j < n;
				j += blockDim.x * gridDim.x){
		distmatrix[TRI_COUNT(j) + col] = distmatrix[TRI_COUNT(j) + row]
				* number[row] + distmatrix[TRI_COUNT(j) + col] * number[col];
		distmatrix[TRI_COUNT(j) + col] /= sum;
	}
}

__global__ void alDelDist1(int row, int n, float* distmatrix) {
//	for (int j = 0; j < row; j++)
		for(int j = blockIdx.x * blockDim.x + threadIdx.x;
						j < row;
						j += blockDim.x * gridDim.x)

		distmatrix[TRI_COUNT(row) + j] = distmatrix[TRI_COUNT(n-1) + j];
}

__global__ void alDelDist2(int row, int n, float* distmatrix) {
//	for (int j = row + 1; j < n - 1; j++)
		for(int j = blockIdx.x * blockDim.x + threadIdx.x+row + 1;
						j < n-1;
						j += blockDim.x * gridDim.x)
		distmatrix[TRI_COUNT(j) + row] = distmatrix[TRI_COUNT(n-1) + j];
}

static __global__ void palcluster(int nrows, int ncolumns,
		cudaTextureObject_t texWt, float* distmatrix, char dist, int transpose, Node* result, int *clusterId, cudaTextureObject_t texIdx, int* number, int *maxGridSize,int *maxThreadsPerBlock)

{

	const int nelements = (transpose==0) ? nrows : ncolumns;

	const int ndata = transpose ? nrows : ncolumns;

	cudaError_t err;

	int maxThreads = THREADS_PER_BLOCK_256;  // number of threads per block

	int maxBlocks = 64;
	int numBlocks = 0;
	int numThreads = 0;

	int row;
	int col;
	float min;

  for (int n = nelements; n > 1; n--)
  { int sum;

  find_nearest_neighbor(maxGridSize, maxThreadsPerBlock, n,distmatrix,texIdx, &row, &col,&min);

  result[nelements-n].distance = min;

    /* Save result */
    result[nelements-n].left = clusterId[row];
    result[nelements-n].right = clusterId[col];

    /* Fix the distances */
    sum = number[row] + number[col];

    getNumBlocksAndThreadsDevice(1,col, maxBlocks, maxThreads, numBlocks,numThreads, maxGridSize, maxThreadsPerBlock);
//
//    threads = (ndata< maxThreads) ? nextPow2(ndata) : maxThreads;
//    gridSize = ( ndata + threads - 1) / threads;
    //		printf("\nndata = %d threads = %d gridSize = %d ", ndata, threads, gridSize);
    if(numThreads > 0 && numBlocks > 0)
    	alFixDistCol1<<<numBlocks, numThreads>>>(col, row, sum, distmatrix, number);
    if (cudaSuccess != (err=cudaGetLastError())) {
    	printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
    	return;
    }
    // wait for child to complete
    if (cudaSuccess != cudaDeviceSynchronize()) {
    	return;
    }

    getNumBlocksAndThreadsDevice(1,row-col+1, maxBlocks, maxThreads, numBlocks,numThreads, maxGridSize, maxThreadsPerBlock);
//
//    threads = (ndata< maxThreads) ? nextPow2(ndata) : maxThreads;
//    gridSize = ( ndata + threads - 1) / threads;
    //		printf("\nndata = %d threads = %d gridSize = %d ", ndata, threads, gridSize);
    if(numThreads > 0 && numBlocks > 0)
    	alFixDistCol2<<<numBlocks, numThreads>>>(col, row, sum, distmatrix, number);

    if (cudaSuccess != (err=cudaGetLastError())) {
    	printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
    	return;
    }
    // wait for child to complete
    if (cudaSuccess != cudaDeviceSynchronize()) {
    	return;
    }


    getNumBlocksAndThreadsDevice(1,n-row+1, maxBlocks, maxThreads, numBlocks,numThreads, maxGridSize, maxThreadsPerBlock);

    if(numThreads > 0 && numBlocks > 0) alFixDistRow<<<numBlocks, numThreads>>>(row, n, col, sum, distmatrix, number);

    if (cudaSuccess != (err=cudaGetLastError())) {
    	printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
    	return;
    }
    // wait for child to complete
    if (cudaSuccess != cudaDeviceSynchronize()) {
    	return;
    }

    getNumBlocksAndThreadsDevice(1,row, maxBlocks, maxThreads, numBlocks,numThreads, maxGridSize, maxThreadsPerBlock);

    if(numThreads > 0 && numBlocks > 0) alDelDist1<<<numBlocks, numThreads>>>(row, n, distmatrix);

    if (cudaSuccess != (err=cudaGetLastError())) {
    	printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
    	return;
    }
    // wait for child to complete
    if (cudaSuccess != cudaDeviceSynchronize()) {
    	return;
    }

    getNumBlocksAndThreadsDevice(1,n-row, maxBlocks, maxThreads, numBlocks,numThreads, maxGridSize, maxThreadsPerBlock);

    if(numThreads > 0 && numBlocks > 0)alDelDist2<<<numBlocks, numThreads>>>(row, n, distmatrix);

    if (cudaSuccess != (err=cudaGetLastError())) {
    	printf("\n@fixDistRow kernel error: %s\n", cudaGetErrorString(err));
    	return;
    }
    // wait for child to complete
    if (cudaSuccess != cudaDeviceSynchronize()) {
    	return;
    }

    /* Update number of elements in the clusters */
    number[col] = sum;
    number[row] = number[n-1];

    /* Update clusterids */
    clusterId[col] = n-nelements-1;
    clusterId[row] = clusterId[n-1];
  }


}



/* ******************************************************************** */

static
Node* pslcluster (int nrows, int ncolumns, float* data,
  float weight[], float* distmatrix, char dist, int transpose)


{ int i, j, k;
  const int nelements = transpose ? ncolumns : nrows;
  const int nnodes = nelements - 1;
  int* vector;
  float* temp;
  int* index;
  Node* result;
  temp = (float*) malloc(nnodes*sizeof(float));
  if(!temp) { printf( "3. malloc() failed"); return NULL;}
  index = (int*) malloc(nelements*sizeof(int));
  if(!index)
  {
	  printf( "4. malloc() failed");
	  free(temp);
    return NULL;
  }
  vector = (int*) malloc(nnodes*sizeof(int));
  if(!vector)
  {
	printf( "5. malloc() failed");
	free(index);
    free(temp);
    return NULL;
  }
  result = (Node*) malloc(nelements*sizeof(Node));
  if(!result)
  {
	printf( "6. malloc() failed");
	free(vector);
    free(index);
    free(temp);
    return NULL;
  }

  for (i = 0; i < nnodes; i++) vector[i] = i;

  if(distmatrix)
  { for (i = 0; i < nrows; i++)
    { result[i].distance = DBL_MAX;
      for (j = 0; j < i; j++) temp[j] = distmatrix[TRI_COUNT(i)+j];
      for (j = 0; j < i; j++)
      { k = vector[j];
        if (result[j].distance >= temp[j])
        { if (result[j].distance < temp[k]) temp[k] = result[j].distance;
          result[j].distance = temp[j];
          vector[j] = i;
        }
        else if (temp[j] < temp[k]) temp[k] = temp[j];
      }
      for (j = 0; j < i; j++)
      {
        if (result[j].distance >= result[vector[j]].distance) vector[j] = i;
      }
    }
  }

  free(temp);

  for (i = 0; i < nnodes; i++) result[i].left = i;
  qsort(result, nnodes, sizeof(Node), nodecompare);

  for (i = 0; i < nelements; i++) index[i] = i;
  for (i = 0; i < nnodes; i++)
  { j = result[i].left;
    k = vector[j];
    result[i].left = index[j];
    result[i].right = index[k];
    index[k] = -i-1;
  }
  free(vector);
  free(index);


  //result = (Node*) realloc(result, nnodes*sizeof(Node));

  return result;
}

/* ******************************************************************* */
static
Node* treecluster(int nrows, int ncolumns, float* data, float *weight, char dist, char method, float *distmatrix_host,  cudaTextureObject_t texData, cudaTextureObject_t texWt, cudaTextureObject_t  texIdx)

{
//	printf("\n\tMaking a new dendogram...\n");
	cudaError_t err;

	int transpose = 0;
	const int nelements = (transpose == 0) ? nrows : ncolumns;
	const int ldistmatrix = 0;

	int *maxGridSize;
	int *maxThreadsPerBlock;

	int gridSize;    // The actual grid size needed, based on input size
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	CUDA_CHECK_RETURN(cudaMalloc(&maxGridSize, sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc(&maxThreadsPerBlock, sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemcpy(maxGridSize, &prop.maxGridSize[1], sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(maxThreadsPerBlock, &prop.maxThreadsPerBlock, sizeof(int), cudaMemcpyHostToDevice));
//	*maxGridSize = prop.maxGridSize[1];
//	*maxThreadsPerBlock = prop.maxThreadsPerBlock;


	if (nelements < 2) {
		DEBUG_PRINT("1. nelements < 2");
		return NULL;
	}

	Node *result = (Node*) malloc(nelements * sizeof(Node));
	Node* result_device;

	// Used to average linkage to keep track of number of clusters
	int *number_device;
	int *clusterid_device;

	CUDA_CHECK_RETURN(cudaMalloc(&clusterid_device, nelements*sizeof(int)));
	assert(clusterid_device != NULL);
	CUDA_CHECK_RETURN(cudaMalloc(&number_device, nelements*sizeof(int)));
	assert(number_device != NULL);

	// wrap raw pointer with a device_ptr
	thrust::device_ptr<int> number_ptr(number_device);

	// use device_ptr in thrust algorithms
	thrust::fill(number_ptr, number_ptr + nelements, (int) 1);

	// wrap raw pointer with a device_ptr
	thrust::device_ptr<int> clusterid_ptr(clusterid_device);

	// initialize X to 0,1,2,3, ....
    thrust::sequence(clusterid_ptr, clusterid_ptr + nelements);


	float *distMatrix;

	float *distMatrix_device;
	CUDA_CHECK_RETURN(cudaMalloc(&distMatrix_device, TRI_COUNT(nelements)*sizeof(float)));
	assert(distMatrix_device != NULL);

	/**
	 * Step 1: Check the memory requirement on the device to compute all distance at once.
	 * If yes then just get all pair wise distance and the build NN list.
	 * Else If no compute NN list one object at a time...
	 */
	size_t available , total;
	cudaMemGetInfo(&available, &total);

	CUDA_CHECK_RETURN(cudaMemcpy(distMatrix_device, distmatrix_host, TRI_COUNT(nelements)*sizeof(float), cudaMemcpyHostToDevice));

	if (cudaSuccess != (err=cudaGetLastError())) {
		printf("\n9. @distancematrix copy error: %s\n", cudaGetErrorString(err));
		return NULL;
	}

//	if(available > (sizeof(float) * TRI_COUNT(nelements) - nelements)){
//		int maxThreads = THREADS_PER_BLOCK_512;
//		int threads = (TRI_COUNT(nelements)+nelements < maxThreads) ? nextPow2Host((TRI_COUNT(nelements)+nelements)) : maxThreads;
//		int gridSize = ( TRI_COUNT(nelements) + threads - 1) / threads < prop.maxGridSize[1]? (TRI_COUNT(nelements) + threads - 1)/ threads : prop.maxGridSize[1];
//
//		//	DEBUG_PRINT("\nCalling distance matrix calculation kernel with gridSize = %d Threads = %d for TRI_COUNT(%d)=%d", gridSize, threads, nrows, TRI_COUNT(nrows));
//		if(threads > 0 && gridSize > 0)
//			distancematrix<<< gridSize, threads>>>(nelements, ncolumns, dist, distMatrix_device, texData, texWt,texIdx, maxGridSize, maxThreadsPerBlock);
//
//		// wait for child to complete
//		if (cudaSuccess != cudaThreadSynchronize()) {
//			return NULL;
//		}
//		if (cudaSuccess != (err=cudaGetLastError())) {
//			DEBUG_PRINT("\n9. @distancematrix kernel error: %s\n", cudaGetErrorString(err));
//			return NULL;
//		}
//
//	}
//	else {
//		DEBUG_PRINT("Distance matrix for %d elements is more than current device can handle...", nelements);
//		return NULL;
//	}

	CUDA_CHECK_RETURN(cudaMalloc(&result_device, nelements *sizeof(Node)));

#ifdef MEASURE_TIME
	clock_t begin = clock();
#endif

	switch(method)
	{
	case 's':
		distMatrix = (float*)malloc(TRI_COUNT(nelements)*sizeof(float));
		assert(distMatrix != NULL);
		CUDA_CHECK_RETURN(cudaMemcpy(distMatrix, distMatrix_device,
				TRI_COUNT(nelements) * sizeof(float), cudaMemcpyDeviceToHost));

		if (cudaSuccess != (err = cudaGetLastError())) {
			DEBUG_PRINT("\n10. cudaMemcpy error: %s\n", cudaGetErrorString(err));
			//return NULL;
		}
		result = pslcluster(nrows, ncolumns, data, weight, distMatrix, dist, transpose);
		free(distMatrix);
		break;
	case 'm':
		pmlcluster<<< 1, 1>>>(nrows, ncolumns, distMatrix_device, dist, transpose, result_device, clusterid_device, texIdx, maxGridSize, maxThreadsPerBlock);

		CUDA_CHECK_RETURN(cudaGetLastError());
		// Finalize.
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		CUDA_CHECK_RETURN(cudaMemcpy((void**)result, result_device, nelements * sizeof(Node), cudaMemcpyDeviceToHost));
		break;
	case 'c':
	{
		float* data1d_device;
		int *mask_device;
		CUDA_CHECK_RETURN(cudaMalloc(&data1d_device, nrows*ncolumns * sizeof(float)));

		CUDA_CHECK_RETURN(cudaMemcpy(data1d_device, data, nrows*ncolumns * sizeof(float), cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMalloc(&mask_device, nrows * ncolumns * sizeof(int)));


		// wrap raw pointer with a device_ptr
		thrust::device_ptr<float> data_ptr(data1d_device);
		thrust::device_ptr<int> mask_ptr(mask_device);

		thrust::transform(data_ptr, data_ptr+nrows*ncolumns , mask_ptr, mask_functor());

		pclcluster<<< 1, 1>>>(nrows, ncolumns, data1d_device, mask_device, texWt, distMatrix_device, dist, transpose, result_device, clusterid_device, texIdx, maxGridSize, maxThreadsPerBlock);

		CUDA_CHECK_RETURN(cudaGetLastError());
		// Finalize.
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		CUDA_CHECK_RETURN(cudaMemcpy((void**)result, result_device, nelements * sizeof(Node), cudaMemcpyDeviceToHost));
		// free memory

		CUDA_CHECK_RETURN(cudaFree(data1d_device));
		CUDA_CHECK_RETURN(cudaFree(mask_device));
	}
	break;
	case 'a':
		palcluster<<< 1, 1>>>(nrows, ncolumns, texWt, distMatrix_device, dist, transpose, result_device, clusterid_device, texIdx, number_device, maxGridSize, maxThreadsPerBlock);

		CUDA_CHECK_RETURN(cudaGetLastError());
		// Finalize.
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		CUDA_CHECK_RETURN(cudaMemcpy((void**)result, result_device, nelements * sizeof(Node), cudaMemcpyDeviceToHost));

		break;
	case 'w':
	{
		float* data1d_device;
		int *mask_device;
		CUDA_CHECK_RETURN(cudaMalloc(&data1d_device, nrows*ncolumns * sizeof(float)));

		CUDA_CHECK_RETURN(cudaMemcpy(data1d_device, data, nrows*ncolumns * sizeof(float), cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMalloc(&mask_device, nrows * ncolumns * sizeof(int)));


		// wrap raw pointer with a device_ptr
		thrust::device_ptr<float> data_ptr(data1d_device);
		thrust::device_ptr<int> mask_ptr(mask_device);

		thrust::transform(data_ptr, data_ptr+nrows*ncolumns , mask_ptr, mask_functor());

		pwlcluster<<< 1, 1>>>(nelements, ncolumns, data1d_device, mask_device, texWt, distMatrix_device, dist, transpose, result_device, clusterid_device, texIdx, number_device, maxGridSize, maxThreadsPerBlock);

		CUDA_CHECK_RETURN(cudaGetLastError());
		// Finalize.
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		//copy result
		CUDA_CHECK_RETURN(cudaMemcpy((void**)result, result_device, nelements * sizeof(Node), cudaMemcpyDeviceToHost));
		// free memory
		CUDA_CHECK_RETURN(cudaFree(data1d_device));
		CUDA_CHECK_RETURN(cudaFree(mask_device));
	}
	break;
	}

	//    for (int i = 0; i < nelements; i++){
	//  	  for (int j = 0; j < i; j++){
	//  		if(distmatrix[i][j]==0)printf("%.4f ", distmatrix[i][j]);
	//  		else printf("     ");
	//  	  }
	//  	  	printf("\n");
	//    }


//	  /* Deallocate space for distance matrix, if it was allocated by treecluster */
//	  if(ldistmatrix)
//	  { free(distMatrix[0]);
//	    free (distMatrix);
//	  }



//#ifdef MEASURE_TIME
//	clock_t end = clock();
//	double time_spent = (double) (end - begin)
//			/ (double) CLOCKS_PER_SEC;
//	DEBUG_PRINT("\nTime spent to compute dendogram for %d objects using is %f\n",
//			nrows, time_spent);
//#endif
	/* Deallocate space for distance matrix, if it was allocated by treecluster */

	if (!result) {
		DEBUG_PRINT("8. malloc() failed");
		//free(clusterid);
		return NULL;
	}


	CUDA_CHECK_RETURN(cudaFree(number_device));
	CUDA_CHECK_RETURN(cudaFree(clusterid_device));
	CUDA_CHECK_RETURN(cudaFree(result_device));
	CUDA_CHECK_RETURN(cudaFree(distMatrix_device));
	CUDA_CHECK_RETURN(cudaFree(maxGridSize));
	CUDA_CHECK_RETURN(cudaFree(maxThreadsPerBlock));


	return result;

}

/* ========================================================================= */

Node* cuda_agglomerative(int nclusters, int nrows, int ncols, float* data1d, float *weight, char method, char dist, int *clusterid, float *distmatrix_host, Node* tree, cudaTextureObject_t texData, cudaTextureObject_t texWt, cudaTextureObject_t  texIdx)
/* Perform hierarchical clustering on data */
{
	cudaError_t err;
//	int i/*, j, k*/;
	int transpose = 0;
	const int nelements = (transpose == 0) ? nrows : ncols;
//	const int nnodes = nelements - 1;
//	Node* tree = NULL;

	if(tree == NULL)
		tree = treecluster(nrows, ncols, data1d, weight, dist, method, distmatrix_host, texData, texWt, texIdx);

	if (!tree) {
		/* Indication that the treecluster routine failed */
		DEBUG_PRINT("treecluster routine failed due to insufficient memory\n");
		return NULL;
	}

	memset(clusterid, 0, nrows * sizeof(int));


	cuttree(nrows, tree, nclusters, clusterid);

// 	free(tree);

	return tree;
}
/* ========================================================================= */
