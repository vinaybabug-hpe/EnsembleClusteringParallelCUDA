/*
 =====================================================================================================================
 Name        : cluster_data_outliers.cu
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

/* Include files */


//#include "sort1.h"
//#include "zscore.h"
//#include "rand.h"
#include <stdbool.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <regex.h>
#include <string>
#include <cstring>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <stdlib.h>
#include "mat.h"
#include "cuda_runtime.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "common/wrapper.h"
#include "common/wrapperFuncs.h"
#include "common/distCalcMthds.h"
#include "common/zscore.h"

struct outlier_functor
{
  const float cutoff;


  outlier_functor(float _cutoff) : cutoff(_cutoff) {}

  __host__ __device__
  int operator()(const float& x) const  {
	  if(fabsf(x) >= cutoff)
		  return 1;
	  else
		  return 0;

  }
};

using namespace std;
/* Function Definitions */
float* cluster_data_outliers(int nrows, int ncols, float* X, char* distFun, float cutoff, int *newNElements)
{

	int *maxGridSize;
	int *maxThreadsPerBlock;
	int numBlocks = 0;
	int numThreads = 0;
	int maxThreads = THREADS_PER_BLOCK;  // number of threads per block
	int maxBlocks = 64;

	char method, dist;
	std::string distFunStr(distFun);

  /*  function [outlierMask,outlierIdxs,outlierZ,outlierDS] = cluster_data_outliers(X,distFun,cutoff) */
  /* CLUSTER_DATA_OUTLIERS - Finds outliers in [n,p] data X where p are features */
  /*  */
  /*  syntax: [outlierMask,outlierIdxs,outlierZ,outlierDS] = ... */
  /*            cluster_data_outliers(X,distFun,cutoff) */
  /*  */
  /*   X        [n,p] */
  /*   distFun  'mahal' or 'euc' */
  /*   cutoff   #stdevs  */
  /*  */


	if(cutoff <= 3){
		cutoff = 3;
	}

	if (distFunStr.compare("mahal") == 0 || distFunStr.compare("mahalanobis") == 0){

		dist = 'm';
	}
	else if (distFunStr.compare("euc") == 0 || distFunStr.compare("euclidean") == 0){

		dist = 'e';
	}

	// Find distance(X, X)
	cudaError_t err;
	//	int i/*, j, k*/;
	int transpose = 0;
	const int nelements = (transpose == 0) ? nrows : ncols;
	float *distmatrix_m;
	float* weight;
	float* data1d_managed;
	int *d_idxs_m;

	size_t memDataSz = 0, memWtSz = 0, memIdxSz=0;

	memDataSz = nrows * ncols * sizeof(float);
	CUDA_CHECK_RETURN(cudaMallocManaged(&data1d_managed, memDataSz));

	CUDA_CHECK_RETURN(cudaMemcpy(data1d_managed, X, memDataSz, cudaMemcpyHostToDevice));


	// TODO: Debug code remove later!
	//		printf("\n-------------------------------------------------------------------------------------- \n\n");
	//		for (int m = 0; m < m_data_bootstrap; m++) {
	//			for (int n = 0; n < n_data_bootstrap; n++) {
	//				printf("%.3f ", data_bootstrap[n + m * n_data_bootstrap]);
	//			}
	//			printf(" \n");
	//		}



	memWtSz = ncols * sizeof(float);

	CUDA_CHECK_RETURN(cudaMallocManaged(&weight, memWtSz));

	memIdxSz = TRI_COUNT(nrows)*sizeof(int);

	CUDA_CHECK_RETURN(cudaMallocManaged(&d_idxs_m, memIdxSz));

	assert(weight != NULL);

	for (int i = 0; i < ncols; i++)
		weight[i] = 1.0;


	int jCount = 0;
	for (int i = 1; i < nrows; i++){
		for (int j = 0; j < i; j++){
			d_idxs_m[jCount] = TRI_COUNT(i)+j;
			jCount++;
		}
	}

	// bind texture to buffer
	// create texture object
	cudaResourceDesc resDescData;
	memset(&resDescData, 0, sizeof(resDescData));
	resDescData.resType = cudaResourceTypeLinear;
	resDescData.res.linear.devPtr = data1d_managed;
	resDescData.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDescData.res.linear.desc.x = 32; // bits per channel
	resDescData.res.linear.sizeInBytes = memDataSz;


	cudaResourceDesc resDescWt;
	memset(&resDescWt, 0, sizeof(resDescWt));
	resDescWt.resType = cudaResourceTypeLinear;
	resDescWt.res.linear.devPtr = weight;
	resDescWt.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDescWt.res.linear.desc.x = 32; // bits per channel
	resDescWt.res.linear.sizeInBytes = memWtSz;

	cudaResourceDesc resDescIdx;
	memset(&resDescIdx, 0, sizeof(resDescIdx));
	resDescIdx.resType = cudaResourceTypeLinear;
	resDescIdx.res.linear.devPtr = d_idxs_m;
	resDescIdx.res.linear.desc.f = cudaChannelFormatKindSigned;
	resDescIdx.res.linear.desc.x = 32; // bits per channel
	resDescIdx.res.linear.sizeInBytes = memIdxSz;

	cudaTextureDesc texDescData;
	memset(&texDescData, 0, sizeof(texDescData));
	texDescData.readMode = cudaReadModeElementType;

	cudaTextureDesc texDescWt;
	memset(&texDescWt, 0, sizeof(texDescWt));
	texDescWt.readMode = cudaReadModeElementType;

	cudaTextureDesc texDescIdx;
	memset(&texDescIdx, 0, sizeof(texDescIdx));
	texDescIdx.readMode = cudaReadModeElementType;

	// create texture object: we only have to do this once!
	cudaTextureObject_t texData = 0;
	cudaCreateTextureObject(&texData, &resDescData, &texDescData, NULL);
	if (cudaSuccess != (err = cudaGetLastError()))
	{
		printf("\n6. @texData error: %s\n",	cudaGetErrorString(err));
		return X;
	}


	cudaTextureObject_t texWt = 0;
	cudaCreateTextureObject(&texWt, &resDescWt, &texDescWt, NULL);
	if (cudaSuccess != (err = cudaGetLastError()))
	{
		printf("\n8. @texWt error: %s\n", cudaGetErrorString(err));
		return X;
	}

	cudaTextureObject_t texIdx = 0;
	cudaCreateTextureObject(&texIdx, &resDescIdx, &texDescIdx, NULL);
	if (cudaSuccess != (err = cudaGetLastError())) {
		printf("\n8. @texIdx error: %s\n", cudaGetErrorString(err));
		return X;
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
		return X;
	}

	// wait for child to complete
	if (cudaSuccess != cudaThreadSynchronize()) {
		return X;
	}

	 thrust::host_vector<float> Z(nelements);

		for (int i = 0; i < nelements; i++){
			for (int j = 0; j < i; j++){
//				distmatrix_h [i * nrows + j] = distmatrix_h [j * nrows + i] =  distmatrix_m[TRI_COUNT(i)+j];
				Z[i] += distmatrix_m[TRI_COUNT(i)+j];
				Z[j] += distmatrix_m[TRI_COUNT(i)+j];
//				printf("%.4f ", distmatrix_m[TRI_COUNT(i)+j]);
			}
//			printf("\n");
			Z[i] /= nelements;
		}

	 zscore(nelements, Z);

	 thrust::device_vector<float> dZ = Z;
	 thrust::device_vector<int> outlierIdx(nelements);
	 thrust::device_vector<int> outlierIdxHost(nelements);

	 // compute  outlierIdx[k] = dZ[k] >= cutoff?1:0;
	 thrust::transform(dZ.begin(), dZ.end(), outlierIdx.begin(), outlier_functor(cutoff));

	 // copy all of H back to the beginning of D
	 thrust::copy(outlierIdx.begin(), outlierIdx.end(), outlierIdxHost.begin());

	 int numOutliers = thrust::reduce(outlierIdx.begin(), outlierIdx.end());

	 *newNElements = nelements-numOutliers;
	 float *newX;
	 newX = (float*) malloc((*newNElements)*ncols * sizeof(float));

	 if(numOutliers > 0){
		 for (int j = 0, j1=0 ; j < nelements; j++) {
			 if(outlierIdxHost[j] == 0){
				 memcpy(&newX[j1 * ncols], &X[j * ncols], ncols * sizeof(float));
				 j1++;
			 }
		 }
	 }
	 else{
		 memcpy(newX, X, (*newNElements)*ncols * sizeof(float));

	 }

		cudaDestroyTextureObject(texData);
		cudaDestroyTextureObject(texWt);
		cudaDestroyTextureObject(texIdx);
		CUDA_CHECK_RETURN(cudaFree(d_idxs_m));
		CUDA_CHECK_RETURN(cudaFree(data1d_managed));
		CUDA_CHECK_RETURN(cudaFree(weight));


	 //TODO: Debug code remove later!
//	 printf("\n-------------------------------------------------------------------------------------- \n\n");
//	 for (int m = 0; m < *newNElements; m++) {
//		 for (int n = 0; n < ncols; n++) {
//			 printf("%.3f ", newX[n + m * ncols]);
//		 }
//		 printf(" \n");
//	 }

	 return newX;

}

/* End of code generation (cluster_data_outliers.c) */
