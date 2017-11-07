/*
 =====================================================================================================================
 Name        : cluster_util_ssw.cu
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
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_set>
#include <set>
#include<cassert>

#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "common/indices_count.h"
#include "common/wrapperFuncs.h"

/* Function Definitions */
void cluster_util_ssw(float *X, int nSamples, int nFeatures, int  *I, double *ssw_data, int nClusters)
{


  /* CLUSTER_UTIL_SSW Compute the total SSE within clusters */
  /*  */
  /*    syntax:  [SSW,ssw] = cluster_util_ssw(X,I,metric) */
  /*  */
  /*    SSW is the total sum of squares within for all clusters (ergo, Total) */
  /*    ssw is an array with the sum squared error within, for each cluster */
  /*  */
  /*    X       : [n,d] data matrix */
  /*    I       : [n,1] column of cluster indices */
  /*   metric   : 'mean' (default) or 'median' */
  /*  set default for metric if not provided */
  /*  compute size variables */

  memset(ssw_data, 0, nClusters * sizeof(double));

  std::unordered_set<int> s(I, I + nSamples);
  std::set<int> nValues(s.begin(), s.end());

  int *valueList = (int*) malloc(nClusters * sizeof(int));
  indices_count(I, nSamples, valueList, nClusters);

  float *M_data = (float*) malloc(nFeatures * sizeof(float));
  assert(M_data);
  float *sw = (float*) malloc(nFeatures * nFeatures * sizeof(float));
  assert(sw);
  float *sw_dev_ptr;
  CUDA_CHECK_RETURN(cudaMalloc((void **) &sw_dev_ptr, nFeatures * nFeatures * sizeof(float)));
//  printf("\n");
  for(int iValue = 0; iValue < nClusters; iValue++){
	  int label = *std::next(nValues.begin(), iValue);
//	  printf("%d ", valueList[iValue]);
	  /**
	   * number of points in the cluster
	   */
	  int n = valueList[iValue];
	 /**
	  * extract subset of data points in cluster k
	  */
	  float *x_data = (float*)malloc(n * nFeatures * sizeof(float));
	  int n_cnt = 0;
	  for (int j = 0; j < nSamples; j++) {
		  if(I[j] == label){
			  memcpy(&x_data[n_cnt * nFeatures], &X[j * nFeatures], nFeatures * sizeof(float));
			  n_cnt++;
		  }
	  }

	  /**
	   * cluster center
	   */
	  memset(M_data, 0, nFeatures * sizeof(float));
	  for(int col =0; col < nFeatures; col++){
		  float mean = 0;
		  int cntMean = 0;

		  for(int row =0; row < n; row++){

				  mean += X[row * nFeatures + col];


		  }
		  if(cntMean != 0)
			  M_data[col] = mean/cntMean;
		  else
			  M_data[col] = 0;
	  }

	  /**
	   * centered the data
	   */
	  for(int row =0; row < n; row++){
		  for(int col =0; col < nFeatures; col++){
			  x_data[row *nFeatures + col] = x_data[row *nFeatures + col] - M_data[col];
		  }
	  }

	  memset(sw, 0, nFeatures * nFeatures * sizeof(float));

//	  float *transpose = (float *) malloc(n * nFeatures * sizeof(float));
//	  assert(transpose);

	  // Compute transpose of centered data
//	  matrixTranspose(transpose, x_data, n, nFeatures);
		// raw pointer to device memory
	  float *x_dev_ptr;
	  CUDA_CHECK_RETURN(cudaMalloc((void **) &x_dev_ptr, n * nFeatures * sizeof(float)));

	  CUDA_CHECK_RETURN(cudaMemcpy(x_dev_ptr, x_data,  n * nFeatures * sizeof(float), cudaMemcpyHostToDevice));

	  float *x_transpose_dev_ptr;
	  CUDA_CHECK_RETURN(cudaMalloc((void **) &x_transpose_dev_ptr, n * nFeatures * sizeof(float)));



	  float const alpha(1.0);
	  float const beta(0.0);
	  cublasHandle_t handle;
	  cublasCreate(&handle);
	  cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, n, nFeatures, &alpha, x_dev_ptr, nFeatures, &beta, x_dev_ptr, n, x_transpose_dev_ptr, n);


	  /**
	   * scatter matrix for the cluster
	   */
//	  matrixMulCPU(sw, transpose, x_data, nFeatures, n, nFeatures);

	  cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, nFeatures, nFeatures, n, &alpha, x_dev_ptr, nFeatures, x_transpose_dev_ptr, n, &beta, sw_dev_ptr, nFeatures );

	  CUDA_CHECK_RETURN(cudaMemcpy(sw, sw_dev_ptr,  nFeatures * nFeatures * sizeof(float), cudaMemcpyDeviceToHost));
	  /**
	   * sum sq error within cluster k
	   * essentially it is the sum of the diagonal elements of the matrix sw
	   */
	  float trace = 0;
	  for (int k = 0; k < nFeatures; k++) {
	    trace += sw[k + nFeatures * k];
	  }

	  ssw_data[iValue] = trace;

//	  free(transpose);
	  cublasDestroy(handle);
	  free(x_data);
	  CUDA_CHECK_RETURN(cudaFree(x_dev_ptr));
	  CUDA_CHECK_RETURN(cudaFree(x_transpose_dev_ptr));
  }


//  printf("\n");
  free(sw);
  free(M_data);
  CUDA_CHECK_RETURN(cudaFree(sw_dev_ptr));

  /*  total within-cluster sum squares  */
}

/* End of code generation (cluster_util_ssw.c) */
