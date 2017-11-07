/*
 =====================================================================================================================
 Name        : distCalcMthds.h
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


#ifndef DISTCALCMTHDS_H_
#define DISTCALCMTHDS_H_

__global__ void distancematrix(int nrows, int ncolumns, char dist, float *distancematix_m,
		cudaTextureObject_t texData, cudaTextureObject_t texWt,cudaTextureObject_t  texIdx,
		int *maxGridSize, int *maxThreadsPerBlock);

__global__ void distancematrix (int nrows, int ncolumns, cudaTextureObject_t texData , float* _centroids, cudaTextureObject_t texWt, int *active, char dist, int row, int transpose, float *distancematix, int *maxGridSize, int *maxThreadsPerBlock);

__global__ void distancematrix(int nrows, int ncolumns, char dist, float *distancematix_m,
		cudaTextureObject_t texData, cudaTextureObject_t texWt, int row,
		int *maxGridSize, int *maxThreadsPerBlock);

__device__ float calcDistMetricGPU(char dist, int cols, int rows1,
		int rows2, int index1, int index2, int transpose, cudaTextureObject_t texData, cudaTextureObject_t texWt,int *maxGridSize, int *maxThreadsPerBlock);

__device__ float calcDistMetricGPU(char dist, int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock);

__device__ float calcDistMetricGPU(char dist, int cols, int rows1, int rows2, float* data1, float* data2,
		cudaTextureObject_t texWt, int index1, int index2, int transpose);

__global__ void distancematrix(int nrows, int ncolumns, char dist, float *distancematix_m,
		cudaTextureObject_t texData, cudaTextureObject_t texMask, cudaTextureObject_t texWt,cudaTextureObject_t  texIdx,
		int *maxGridSize, int *maxThreadsPerBlock);

#endif /* DISTCALCMTHDS_H_ */
