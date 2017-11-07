/*
 =====================================================================================================================
 Name        : wrapperFuncs.h
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


#ifndef WRAPPERFUNCS_H_
#define WRAPPERFUNCS_H_

#include "common/wrapper.h"
#include "common/clustLib.h"

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


#if defined(DEBUG) && DEBUG > 0
 #define DEBUG_PRINT(fmt, args...) fprintf(stderr, "\nDEBUG: %s:%d:%s(): " fmt " \n", \
    __FILE__, __LINE__, __func__, ##args)
#else
 #define DEBUG_PRINT(fmt, args...) /* Don't do anything in release builds */
#endif



int getModelType(char *clustMethod, char *methodList);
void getClusteringMethodsList(int length, char *methodList[], char **clustList);
int doKmeans(char **clustList);
int doKmedoids(char **clustList);
int doGMM(char **clustList);
int doSpectral(char **clustList);
int doAgglom(char **clustList);
int getMthdLstbyClust(char*clustMethod, char **methodList,
		int length, char **mthdLstbyClust);
int getClustMthdCnt(char*clustMethod, char **methodList,
		int length);
void getDistMtrcnCntrFunByKmeans(char*clustMthd, char *model,
		char *distmetric, char *centerfun);
void getDistMtrcnCntrFunByKmedoid(char*clustMthd, char *model,
		char *distmetric, char *centerfun);
void getDistMtrcnCntrFunByAgglo(char*clustMthd, char *model,
		char *distmetric, char *centerfun, char *linkcode);
void getDistMtrcnCntrFunBySpectral(char*clustMthd, char *model,
		char *distmetric, char *centerfun);

void mex2CudaWrapper();

void mex2cu_spectral_adapter(int nclusters, int nrows, int ncols, float* data1d,
	char* _method, char* _dist, int *clusterid);

void mex2cu_spectral_adapter(int nclusters,
							  int nrows,
							  int ncols,
							  /*float* data1d,*/
							  char* _method,
							  char* _dist,
							  int *clusterid,
							  cudaTextureObject_t texData,
							  cudaTextureObject_t texWt,
							  cudaTextureObject_t texIdx);

/* return an array of cluster centers of size [numClusters][numCoords]       */
float** cuda_kmeans(
				   char distType,		/* Distance measure to use */
				   float *objects,      /* in: [numObjs][numCoords] */
                   int     numCoords,    /* no. features */
                   int     numObjs,      /* no. objects */
                   int     numClusters,  /* no. clusters */
                   float   threshold,    /* % objects change membership */
                   int    *membership,   /* out: [numObjs] */
                   int    *loop_iterations,
                   cudaTextureObject_t texData,
                   cudaTextureObject_t texWt,
                   int *maxGridSize,
                   int *maxThreadsPerBlock);

void mex2cu_kmeans_adapter(int nclusters,
						   int nrows,
						   int ncols,
						   float* data1d,
						   char* _method,
						   char* _dist,
						   float threshold,
						   int *loop_iterations,
						   int *clusterid,
						   cudaTextureObject_t texData,
						   cudaTextureObject_t texWt);

/* return an array of cluster centers of size [numClusters][numCoords]       */
float** cuda_kmedians(
				   char distType,		/* Distance measure to use */
				   float *objects,      /* in: [numObjs][numCoords] */
                   int     numCoords,    /* no. features */
                   int     numObjs,      /* no. objects */
                   int     numClusters,  /* no. clusters */
                   float   threshold,    /* % objects change membership */
                   int    *membership,   /* out: [numObjs] */
                   int    *loop_iterations,
                   cudaTextureObject_t texData,
                   cudaTextureObject_t texWt,
                   int *maxGridSize,
                   int *maxThreadsPerBlock);

void mex2cu_kmedians_adapter(int nclusters,
						   int nrows,
						   int ncols,
						   float* data1d,
						   char* _method,
						   char* _dist,
						   float threshold,
						   int *loop_iterations,
						   int *clusterid,
						   cudaTextureObject_t texData,
						   cudaTextureObject_t texWt);

int cuda_gmm_main(int desired_num_clusters, float* fcs_data_by_event, int num_dimensions, int num_events, int *clusterIdx);

void mex2cu_gmm_adapter(int nclusters,
						int nrows,
						int ncols,
						float* data1d,
						char* _method,
						char* _dist,
						int *clusterid,
						cudaTextureObject_t texData,
						cudaTextureObject_t texWt);

Node* cuda_agglomerative(int nclusters, int nrows, int ncols, float* data1d,  float *weight, char method, char dist, int *clusterid,float *distmatrix_host,Node* tree, cudaTextureObject_t texData, cudaTextureObject_t texWt, cudaTextureObject_t  texIdx);

Node* mex2cu_agglomerative_adapter(int nclusters, int nrows, int ncols, float* data1d,  float *weight,
	char* _method, char* _dist, int *clusterid, float* distmatrix_host, Node* tree, cudaTextureObject_t texData, cudaTextureObject_t texWt, cudaTextureObject_t  texIdx);

int run_qsort(unsigned int size, float *data, int debug);

__device__ bool isPow2(unsigned int x);

__device__ unsigned int nextPow2(unsigned int x);

bool isPow2Host(unsigned int x);

unsigned int nextPow2Host(unsigned int x);

void getNumBlocksAndThreadsHost(/*int whichKernel,*/ int n, int maxBlocks,
	int maxThreads, int &blocks, int &threads, int *maxGridSize, int *maxThreadsPerBlock);

__device__ void getNumBlocksAndThreadsReduceMin6(int whichKernel, int n, int maxBlocks,
	int maxThreads, int &blocks, int &threads, int *maxGridSize, int *maxThreadsPerBlock);

__device__ void getNumBlocksAndThreadsDevice(/*int whichKernel,*/ int n, int maxBlocks,
	int maxThreads, int &blocks, int &threads, int *maxGridSize, int *maxThreadsPerBlock);


int unique_length(int *a, size_t len);

void matrixTranspose(float *gold, float *idata, const  int size_x, const  int size_y);

void
matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB);

//void sort_models(char **actual, int size, char **alph1, char **alph2, char **alph3);
void sort_models(char actual[][MODEL_STR_LEN], int size, char alph1[][MODEL_STR_LEN], char alph2[][MODEL_STR_LEN], char alph3[][MODEL_STR_LEN], char **alpha4);

char link_str2c(char* _method);

char dist_str2c(char* _dist);

#endif /* WRAPPERFUNCS_H_ */
