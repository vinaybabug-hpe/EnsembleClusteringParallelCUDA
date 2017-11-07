/*
 =====================================================================================================================
 Name        : mex2cuda.cu
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
#include <numeric>
#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <cassert>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <cuda_runtime.h>
#include <math_functions.h>
#include "common/clustLib.h"
#include "common/wrapper.h"
#include "common/wrapperFuncs.h"
#include "common/distCalcMthds.h"


__device__ bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

__device__ unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

bool isPow2Host(unsigned int x)
{
    return ((x&(x-1))==0);
}

unsigned int nextPow2Host(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}
////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// We set threads / block to the minimum of maxThreads and
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreadsHost(/*int whichKernel,*/ int n, int maxBlocks,
	int maxThreads, int &blocks, int &threads, int *maxGridSize, int *maxThreadsPerBlock) {

	threads = (n < maxThreads) ? nextPow2Host(n) : maxThreads;
	blocks = (n + threads - 1) / threads;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
__device__ void getNumBlocksAndThreadsReduceMin6(int whichKernel, int n, int maxBlocks,
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
		//printf("n (%d) is too large, please choose a smaller number!\n", n);
	}

	if (blocks > (*maxGridSize)) {
//		printf(
//				"Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
//				blocks, (*maxGridSize), threads * 2, threads);

		blocks /= 2;
		threads *= 2;
	}

	if (whichKernel == 6) {
		blocks = MIN(maxBlocks, blocks);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// We set threads / block to the minimum of maxThreads and
////////////////////////////////////////////////////////////////////////////////
__device__ void getNumBlocksAndThreadsDevice(/*int whichKernel,*/ int n, int maxBlocks,
	int maxThreads, int &blocks, int &threads, int *maxGridSize, int *maxThreadsPerBlock) {

	threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
	blocks = (n + threads - 1) / threads;
}

/*
 * Purpose
 * =======
 *
 *	TODO: write explaination
 *
 */
static __global__ void initMatrix(int* number, int width, int height, int val) {


	for(int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
			tIdx < width*height;
			tIdx += blockDim.x * gridDim.x){
        number[tIdx]=val;
    }
}


/* ========================================================================= */
void cuda_spectral_adapter_full_distmat(int nrows, int ncols, int numclusters,float *distmatrix, int* idxs)
{
	if(numclusters == 1){
		memset(idxs, 0, nrows*sizeof(int));
		return;
	}
	fastsc_full_distmat(nrows, ncols, numclusters, distmatrix, idxs);
	return;
}

/* ========================================================================= */
void cuda_spectral_adapter(int nclusters, int nrows, int ncols, /*float* data1d,*/ char method, char dist, int *clusterid, cudaTextureObject_t texData, cudaTextureObject_t texWt,cudaTextureObject_t  texIdx)
/* Perform hierarchical clustering on data */
{
	cudaError_t err;
//	int i/*, j, k*/;
	int transpose = 0;
	const int nelements = (transpose == 0) ? nrows : ncols;

	float *distmatrix_m;

	float* weight;
	float* data1d_managed;
	int *d_idxs_m;

	int *maxGridSize;
	int *maxThreadsPerBlock;
	int numBlocks = 0;
	int numThreads = 0;
	int maxThreads = THREADS_PER_BLOCK;  // number of threads per block
	int maxBlocks = 64;


	if(nclusters == 1){
		memset(clusterid,0, nrows*sizeof(int));
		return;
	}

	CUDA_CHECK_RETURN(cudaMallocManaged(&distmatrix_m, TRI_COUNT(nelements)*sizeof(float)));

	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	CUDA_CHECK_RETURN(cudaMallocManaged(&maxGridSize, sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged(&maxThreadsPerBlock, sizeof(int)));

	*maxGridSize = prop.maxGridSize[1];
	*maxThreadsPerBlock = prop.maxThreadsPerBlock;

	maxThreads= *maxThreadsPerBlock;

	int threads = (TRI_COUNT(nelements)+nelements < maxThreads) ? nextPow2Host((TRI_COUNT(nelements)+nelements)) : maxThreads;
	int gridSize = ( TRI_COUNT(nelements) + threads - 1) / threads < *maxGridSize? (TRI_COUNT(nelements) + threads - 1)/ threads : *maxGridSize;

//	printf("\nCalling distance matrix calculation kernel with gridSize = %d Threads = %d for TRI_COUNT(%d)=%d", gridSize, threads, nrows, TRI_COUNT(nrows));
	if(threads > 0 && gridSize > 0)
	distancematrix<<< gridSize, threads>>>(nrows, ncols , dist, distmatrix_m, texData, texWt,texIdx, maxGridSize, maxThreadsPerBlock);

	if (cudaSuccess != (err=cudaGetLastError())) {

		printf("\n9. @distancematrix kernel error: %s\n", cudaGetErrorString(err));
		return;
	}

	// wait for child to complete
	if (cudaSuccess != cudaThreadSynchronize()) {
		return;
	}

//	for (int i = 0; i < nelements; i++){
//		for (int j = 0; j < i; j++){
//			distmatrix_h [i * nrows + j] = distmatrix_h [j * nrows + i] =  distmatrix_m[TRI_COUNT(i)+j];
////			printf("%.4f ", distmatrix_m[TRI_COUNT(i)+j]);
//		}
////		printf("\n");
//	}


//	for(int row=0; row<nrows; row++){
//
//		for(int col=0; col<row; col++){
//
//			printf("%.3f - %d ", distmatrix_m[TRI_COUNT(row)+col], TRI_COUNT(row)+col);
//
//			distmatrix_h [row * nrows * col] = distmatrix_h [col * nrows + row] = distmatrix_m[TRI_COUNT(row)+col];
//
//
//		}
//			printf("\n");
//
//	}
//


//	// Call fastsc
//	hello_fastsc();
	fastsc(nrows, ncols,nclusters, distmatrix_m, clusterid);


//	for(i=0; i<nrows; i++)
//		printf("obj[%2d]: cluster, %2d\n", i, clusteridManaged[i]);
//	printf("\n");


	CUDA_CHECK_RETURN(cudaFree(distmatrix_m));
	CUDA_CHECK_RETURN(cudaFree(maxGridSize));
	CUDA_CHECK_RETURN(cudaFree(maxThreadsPerBlock));


	return;
}



/* ========================================================================= */

void mex2cu_spectral_adapter(int nclusters, int nrows, int ncols, float* data1d,
	char* _method, char* _dist, int *clusterid)
	/* Perform hierarchical clustering on data */
{

	if(nclusters == 1){
		memset(clusterid,0, nrows*sizeof(int));
		return;
	}

	char method, dist;
	//int i, j;

	// Assign lib specific parameters to link method
	// and distance function
	method = link_str2c(_method);

	dist = dist_str2c(_dist);

	cuda_spectral_adapter_full_distmat(nrows, ncols, nclusters >= nrows ? nrows : nclusters, data1d, clusterid);

}

/* ========================================================================= */


void mex2cu_spectral_adapter(int nclusters, int nrows, int ncols, /*float* data1d,*/
	char* _method, char* _dist, int *clusterid, cudaTextureObject_t texData, cudaTextureObject_t texWt,cudaTextureObject_t  texIdx)
	/* Perform hierarchical clustering on data */
{

	if(nclusters == 1){
		memset(clusterid,0, nrows*sizeof(int));
		return;
	}

	char method, dist;
	//int i, j;

	// Assign lib specific parameters to link method
	// and distance function
	if(_method == NULL){
		method = 's';
	}
	else
	if (strcmp(_method, LNK_CODE_AVG) == 0) {
		method = 'a';
	}
	else if (strcmp(_method, LNK_CODE_CEN) == 0) {
		method = 'c';
	}
	else if (strcmp(_method, LNK_CODE_COM) == 0) {
		method = 'm';
	}
	else if (strcmp(_method, LNK_CODE_SIN) == 0) {
		method = 's';
	}
	else if (strcmp(_method, LNK_CODE_MED) == 0) {
		method = 'a'; // TODO: median linkage should be implemented later
	}
	else if (strcmp(_method, LNK_CODE_WAR) == 0) {
		method = 'w'; // TODO: ward linkage should be implemented later
	}
	else if (strcmp(_method, LNK_CODE_WEI) == 0) {
		method = 'a'; // TODO: weighted linkage should be implemented later
	}

	if (_dist == NULL) {
			dist = 'e';
	}
	else
	if (strcmp(_dist, DIST_MTRC_EUC) == 0) {
		dist = 'e';
	}
	else if (strcmp(_dist, DIST_MTRC_SEU) == 0) {
		dist = 'l';
	}
	else if (strcmp(_dist, DIST_MTRC_CIT) == 0) {
		dist = 'b';
	}
	else if (strcmp(_dist, DIST_MTRC_COR) == 0) {
		dist = 'c';
	}
	else if (strcmp(_dist, DIST_MTRC_ACOR) == 0) {
		dist = 'a';
	}
	else if (strcmp(_dist, DIST_MTRC_UCOR) == 0) {
		dist = 'u';
	}
	else if (strcmp(_dist, DIST_MTRC_AUCOR) == 0) {
		dist = 'x';
	}
	else if (strcmp(_dist, DIST_MTRC_COS) == 0) {
		dist = 'o';
	}
	else if (strcmp(_dist, DIST_MTRC_KEN) == 0) {
		dist = 'k';
	}
	else if (strcmp(_dist, DIST_MTRC_MAH) == 0) {
		dist = 'm';
	}
	else if (strcmp(_dist, DIST_MTRC_JAC) == 0) {
		dist = 'j';
	}
	else if (strcmp(_dist, DIST_MTRC_CHE) == 0) {
		dist = 'h';
	}
	else if (strcmp(_dist, DIST_MTRC_SPE) == 0) {
		dist = 's';
	}
	else if (strcmp(_dist, DIST_MTRC_HAM) == 0) {
		dist = 'g';
	}

//	printf("\n5. Calling CUDA Spectral Clustering Adapter...\n");
	cuda_spectral_adapter(nclusters >= nrows ? nrows : nclusters, nrows, ncols, /*data1d,*/ method, dist, clusterid, texData, texWt, texIdx);



}

/* ========================================================================= */

void mex2cu_kmeans_adapter(int nclusters, int nrows, int ncols, float* data1d,
	char* _method, char* _dist,float threshold, int *loop_iterations, int *clusterid, cudaTextureObject_t texData, cudaTextureObject_t texWt)
	/* Perform hierarchical clustering on data */
{

	if(nclusters == 1){
		memset(clusterid,0, nrows*sizeof(int));
		return;
	}

	char method, dist;
	//int i, j;

	// Assign lib specific parameters to link method
	// and distance function
	if(_method == NULL){
		method = 's';
	}
	else
	if (strcmp(_method, LNK_CODE_AVG) == 0) {
		method = 'a';
	}
	else if (strcmp(_method, LNK_CODE_CEN) == 0) {
		method = 'c';
	}
	else if (strcmp(_method, LNK_CODE_COM) == 0) {
		method = 'm';
	}
	else if (strcmp(_method, LNK_CODE_SIN) == 0) {
		method = 's';
	}
	else if (strcmp(_method, LNK_CODE_MED) == 0) {
		method = 'a'; // TODO: median linkage should be implemented later
	}
	else if (strcmp(_method, LNK_CODE_WAR) == 0) {
		method = 'w'; // TODO: ward linkage should be implemented later
	}
	else if (strcmp(_method, LNK_CODE_WEI) == 0) {
		method = 'a'; // TODO: weighted linkage should be implemented later
	}

	if (_dist == NULL) {
			dist = 'e';
	}
	else
	if (strcmp(_dist, DIST_MTRC_EUC) == 0) {
		dist = 'e';
	}
	else if (strcmp(_dist, DIST_MTRC_SEU) == 0) {
		dist = 'l';
	}
	else if (strcmp(_dist, DIST_MTRC_CIT) == 0) {
		dist = 'b';
	}
	else if (strcmp(_dist, DIST_MTRC_COR) == 0) {
		dist = 'c';
	}
	else if (strcmp(_dist, DIST_MTRC_ACOR) == 0) {
		dist = 'a';
	}
	else if (strcmp(_dist, DIST_MTRC_UCOR) == 0) {
		dist = 'u';
	}
	else if (strcmp(_dist, DIST_MTRC_AUCOR) == 0) {
		dist = 'x';
	}
	else if (strcmp(_dist, DIST_MTRC_COS) == 0) {
		dist = 'o';
	}
	else if (strcmp(_dist, DIST_MTRC_KEN) == 0) {
		dist = 'k';
	}
	else if (strcmp(_dist, DIST_MTRC_MAH) == 0) {
		dist = 'm';
	}
	else if (strcmp(_dist, DIST_MTRC_JAC) == 0) {
		dist = 'j';
	}
	else if (strcmp(_dist, DIST_MTRC_CHE) == 0) {
		dist = 'h';
	}
	else if (strcmp(_dist, DIST_MTRC_SPE) == 0) {
		dist = 's';
	}
	else if (strcmp(_dist, DIST_MTRC_HAM) == 0) {
		dist = 'g';
	}

	int *maxGridSize;
	int *maxThreadsPerBlock;
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	CUDA_CHECK_RETURN(cudaMallocManaged(&maxGridSize, sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged(&maxThreadsPerBlock, sizeof(int)));

	*maxGridSize = prop.maxGridSize[1];
	*maxThreadsPerBlock = prop.maxThreadsPerBlock;

//	printf("\n5. Calling CUDA Spectral Clustering Adapter...\n");
	cuda_kmeans(dist, data1d, ncols, nrows, nclusters, threshold, clusterid, loop_iterations, texData, texWt, maxGridSize, maxThreadsPerBlock);

	CUDA_CHECK_RETURN(cudaFree(maxGridSize));
	CUDA_CHECK_RETURN(cudaFree(maxThreadsPerBlock));

}

/* ========================================================================= */

void mex2cu_kmedians_adapter(int nclusters, int nrows, int ncols, float* data1d,
	char* _method, char* _dist,float threshold, int *loop_iterations, int *clusterid, cudaTextureObject_t texData, cudaTextureObject_t texWt)
	/* Perform hierarchical clustering on data */
{

	if(nclusters == 1){
		memset(clusterid,0, nrows*sizeof(int));
		return;
	}

	char method, dist;
	//int i, j;

	// Assign lib specific parameters to link method
	// and distance function
	if(_method == NULL){
		method = 's';
	}
	else
	if (strcmp(_method, LNK_CODE_AVG) == 0) {
		method = 'a';
	}
	else if (strcmp(_method, LNK_CODE_CEN) == 0) {
		method = 'c';
	}
	else if (strcmp(_method, LNK_CODE_COM) == 0) {
		method = 'm';
	}
	else if (strcmp(_method, LNK_CODE_SIN) == 0) {
		method = 's';
	}
	else if (strcmp(_method, LNK_CODE_MED) == 0) {
		method = 'a'; // TODO: median linkage should be implemented later
	}
	else if (strcmp(_method, LNK_CODE_WAR) == 0) {
		method = 'w'; // TODO: ward linkage should be implemented later
	}
	else if (strcmp(_method, LNK_CODE_WEI) == 0) {
		method = 'a'; // TODO: weighted linkage should be implemented later
	}

	if (_dist == NULL) {
			dist = 'e';
	}
	else
	if (strcmp(_dist, DIST_MTRC_EUC) == 0) {
		dist = 'e';
	}
	else if (strcmp(_dist, DIST_MTRC_SEU) == 0) {
		dist = 'l';
	}
	else if (strcmp(_dist, DIST_MTRC_CIT) == 0) {
		dist = 'b';
	}
	else if (strcmp(_dist, DIST_MTRC_COR) == 0) {
		dist = 'c';
	}
	else if (strcmp(_dist, DIST_MTRC_ACOR) == 0) {
		dist = 'a';
	}
	else if (strcmp(_dist, DIST_MTRC_UCOR) == 0) {
		dist = 'u';
	}
	else if (strcmp(_dist, DIST_MTRC_AUCOR) == 0) {
		dist = 'x';
	}
	else if (strcmp(_dist, DIST_MTRC_COS) == 0) {
		dist = 'o';
	}
	else if (strcmp(_dist, DIST_MTRC_KEN) == 0) {
		dist = 'k';
	}
	else if (strcmp(_dist, DIST_MTRC_MAH) == 0) {
		dist = 'm';
	}
	else if (strcmp(_dist, DIST_MTRC_JAC) == 0) {
		dist = 'j';
	}
	else if (strcmp(_dist, DIST_MTRC_CHE) == 0) {
		dist = 'h';
	}
	else if (strcmp(_dist, DIST_MTRC_SPE) == 0) {
		dist = 's';
	}
	else if (strcmp(_dist, DIST_MTRC_HAM) == 0) {
		dist = 'g';
	}

	int *maxGridSize;
	int *maxThreadsPerBlock;
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	CUDA_CHECK_RETURN(cudaMallocManaged(&maxGridSize, sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged(&maxThreadsPerBlock, sizeof(int)));

	*maxGridSize = prop.maxGridSize[1];
	*maxThreadsPerBlock = prop.maxThreadsPerBlock;

//	printf("\n5. Calling CUDA Spectral Clustering Adapter...\n");
	cuda_kmedians(dist, data1d, ncols, nrows, nclusters, threshold, clusterid, loop_iterations, texData, texWt, maxGridSize, maxThreadsPerBlock);

	CUDA_CHECK_RETURN(cudaFree(maxGridSize));
	CUDA_CHECK_RETURN(cudaFree(maxThreadsPerBlock));

}

void mex2cu_gmm_adapter(int nclusters, int nrows, int ncols, float* data1d,
	char* _method, char* _dist, int *clusterid, cudaTextureObject_t texData, cudaTextureObject_t texWt)
	/* Perform hierarchical clustering on data */
{

	if(nclusters == 1){
		memset(clusterid,0, nrows*sizeof(int));
		return;
	}

	char method, dist;
	//int i, j;

	// Assign lib specific parameters to link method
	// and distance function
	if(_method == NULL){
		method = 's';
	}
	else
	if (strcmp(_method, LNK_CODE_AVG) == 0) {
		method = 'a';
	}
	else if (strcmp(_method, LNK_CODE_CEN) == 0) {
		method = 'c';
	}
	else if (strcmp(_method, LNK_CODE_COM) == 0) {
		method = 'm';
	}
	else if (strcmp(_method, LNK_CODE_SIN) == 0) {
		method = 's';
	}
	else if (strcmp(_method, LNK_CODE_MED) == 0) {
		method = 'a'; // TODO: median linkage should be implemented later
	}
	else if (strcmp(_method, LNK_CODE_WAR) == 0) {
		method = 'w'; // TODO: ward linkage should be implemented later
	}
	else if (strcmp(_method, LNK_CODE_WEI) == 0) {
		method = 'a'; // TODO: weighted linkage should be implemented later
	}

	if (_dist == NULL) {
			dist = 'e';
	}
	else
	if (strcmp(_dist, DIST_MTRC_EUC) == 0) {
		dist = 'e';
	}
	else if (strcmp(_dist, DIST_MTRC_SEU) == 0) {
		dist = 'l';
	}
	else if (strcmp(_dist, DIST_MTRC_CIT) == 0) {
		dist = 'b';
	}
	else if (strcmp(_dist, DIST_MTRC_COR) == 0) {
		dist = 'c';
	}
	else if (strcmp(_dist, DIST_MTRC_ACOR) == 0) {
		dist = 'a';
	}
	else if (strcmp(_dist, DIST_MTRC_UCOR) == 0) {
		dist = 'u';
	}
	else if (strcmp(_dist, DIST_MTRC_AUCOR) == 0) {
		dist = 'x';
	}
	else if (strcmp(_dist, DIST_MTRC_COS) == 0) {
		dist = 'o';
	}
	else if (strcmp(_dist, DIST_MTRC_KEN) == 0) {
		dist = 'k';
	}
	else if (strcmp(_dist, DIST_MTRC_MAH) == 0) {
		dist = 'm';
	}
	else if (strcmp(_dist, DIST_MTRC_JAC) == 0) {
		dist = 'j';
	}
	else if (strcmp(_dist, DIST_MTRC_CHE) == 0) {
		dist = 'h';
	}
	else if (strcmp(_dist, DIST_MTRC_SPE) == 0) {
		dist = 's';
	}
	else if (strcmp(_dist, DIST_MTRC_HAM) == 0) {
		dist = 'g';
	}

	int *maxGridSize;
	int *maxThreadsPerBlock;
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	CUDA_CHECK_RETURN(cudaMallocManaged(&maxGridSize, sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged(&maxThreadsPerBlock, sizeof(int)));

	*maxGridSize = prop.maxGridSize[1];
	*maxThreadsPerBlock = prop.maxThreadsPerBlock;

//	printf("\n5. Calling CUDA Spectral Clustering Adapter...\n");
	cuda_gmm_main(nclusters, data1d, ncols, nrows, clusterid);

	CUDA_CHECK_RETURN(cudaFree(maxGridSize));
	CUDA_CHECK_RETURN(cudaFree(maxThreadsPerBlock));

}

/* ========================================================================= */

Node* mex2cu_agglomerative_adapter(int nclusters, int nrows, int ncols, float* data1d, float *weight,
	char* _method, char* _dist, int *clusterid, float* distmatrix_host, Node *tree, cudaTextureObject_t texData, cudaTextureObject_t texWt, cudaTextureObject_t  texIdx)
	/* Perform hierarchical clustering on data */
{

	if(nclusters == 1){
		memset(clusterid,0, nrows*sizeof(int));
		return NULL;
	}

	char method, dist;
	//int i, j;


	// Assign lib specific parameters to link method
	// and distance function
	if (strcmp(_method, LNK_CODE_AVG) == 0) {
		method = 'a';
	}
	else if (strcmp(_method, LNK_CODE_CEN) == 0) {
		method = 'c';
	}
	else if (strcmp(_method, LNK_CODE_COM) == 0) {
		method = 'm';
	}
	else if (strcmp(_method, LNK_CODE_SIN) == 0) {
		method = 's';
	}
	else if (strcmp(_method, LNK_CODE_MED) == 0) {
		method = 'a'; // TODO: median linkage should be implemented later
	}
	else if (strcmp(_method, LNK_CODE_WAR) == 0) {
		method = 'w'; // TODO: ward linkage should be implemented later
	}
	else if (strcmp(_method, LNK_CODE_WEI) == 0) {
		method = 'a'; // TODO: weighted linkage should be implemented later
	}

	if (strcmp(_dist, DIST_MTRC_EUC) == 0) {
		dist = 'e';
	}
	else if (strcmp(_dist, DIST_MTRC_SEU) == 0) {
		dist = 'e';
	}
	else if (strcmp(_dist, DIST_MTRC_CIT) == 0) {
		dist = 'b';
	}
	else if (strcmp(_dist, DIST_MTRC_COR) == 0) {
		dist = 'c';
	}
	else if (strcmp(_dist, DIST_MTRC_COS) == 0) {
		dist = 'o';
	}
	else if (strcmp(_dist, DIST_MTRC_MAH) == 0) {
		dist = 'm';
	}
	else if (strcmp(_dist, DIST_MTRC_JAC) == 0) {
		dist = 'j';
	}
	else if (strcmp(_dist, DIST_MTRC_CHE) == 0) {
		dist = 'h';
	}
	else if (strcmp(_dist, DIST_MTRC_SPE) == 0) {
		dist = 's';
	}
	else if (strcmp(_dist, DIST_MTRC_HAM) == 0) {
		dist = 'g';
	}

	tree = cuda_agglomerative(nclusters >= nrows ? nrows : nclusters, nrows, ncols, data1d, weight, method, dist, clusterid, distmatrix_host, tree, texData, texWt, texIdx);

	return tree;
}

__global__ void cudaKernel();

/*
 * Wrapper function
 */

void mex2CudaWrapper(){

#ifdef MATLAB
	mexPrintf("\nHello World from cuda kernel\n");
#else
	printf("\nHello World from cuda kernel\n");
#endif


	cudaKernel<<<1, 1>>>();

}

/*
 * Host code
 */
__global__ void cudaKernel(){

	int tid = blockIdx.x;


}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
/*static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}*/

