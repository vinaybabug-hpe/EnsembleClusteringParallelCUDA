/*
 =====================================================================================================================
 Name        : cluster_ensemble2cam.cu
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
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cassert>

#include "common/cluster_util_partition2cam.h"

/* Function Definitions */
void cluster_ensemble2cam(int *I,  float *A, float *N, int nSamples, int nPartitions)
{

	// make dissimQ always true
	bool dissimQ = true;

  /* CLUSTER_ENSEMBLE2CAM Generates co-association matrix from ensemble of partitions */
  /*    */
  /*  Syntax:  A = cluster_partitions2cam(E)    */
  /*           A = cluster_partitions2cam(E,dissimQ) */
  /*  */
  /*    INPUT ARGUMENTS */
  /*         E : ensemble struct with dataset .partitions and field .indices */
  /*             or, dataset with field .indices */
  /*             or,  [n,e] array of partition indices, each partition is a column */
  /*  */
  /*    dissimQ : if true, A is returned as a dissimilarity matrix rather than */
  /*              a similarity matrix (default), i.e. A = (1-A) */
  /*  */
  /*  */
  /*    OUTPUT ARGUMENTS */
  /*    	    A :  Co-association matrix (percent)  [n,n] where n = num samples */
  /*            N :  Tracks which pairs co-occured in the partitions.  Not all pairs */
  /*                 will co-occurr if partitions generated with bootstrapping.  */
  /*  */
  /*  Author:         Lee I Newman */
  /*  Affiliation:    University of Michigan, Depts. Psychology, EECS */
  /*  email:          leenewm@umich.edu */
  /*  Website:        http://www.leenewman.org */
  /*  Created:        May 2009 */
  /*  Revised by/on:  person, date */
  /*  */
  /* ------------- BEGIN CODE -------------- */
  /* % Process input arguments */
  /*  ///////////////////////////////////////////////////////// */
	 memset(A, 0, nSamples * nSamples * sizeof(float));
	 memset(N, 0, nSamples * nSamples * sizeof(float));

  /* Generate the co-association matrix */
  /* will hold co-association counts */
	int *A_Temp, *N_Temp;
	A_Temp = (int*) malloc(nSamples * nSamples * sizeof(int));
	assert(A_Temp != NULL);
	N_Temp = (int*) malloc(nSamples * nSamples * sizeof(int));
	assert(N_Temp != NULL);

  /*  will hold co-occurence counts */
  /*  For each partition, get co-association matrix A and co-occurence matrix N,  */
  /*  and accumulate them them across partitions. */
  for (int i = 0; i < nPartitions; i++) {
    /* [Anew,Nnew] =  cluster_util_partition2cam(I(:,i));     */
	cluster_util_partition2cam(&I[i*nSamples], A_Temp, N_Temp, nSamples, nPartitions);

    for (int i0 = 0; i0 < nSamples * nSamples; i0++) {
      A[i0] += A_Temp[i0];
      N[i0] += N_Temp[i0];
    }
  }

  free(A_Temp);
  free(N_Temp);

//  /*  don't want to divide by 0, but note if element of N is zero (two points */
//  /*  were never in same bootsample, then they can't co-associate, so will */
//  /*  always be 0/0  which is Nan...so don't need code below */
//  /*  divide Co-association by co-occurrence to get percent co-association */
  for (int i0 = 0; i0 < nSamples * nSamples; i0++) {
	if(N[i0] != 0 && A[i0] != 0) A[i0] /= N[i0];
  }

  /* A = my_setdiag(A,ones(nSamples,1)); */
  if (dissimQ) {
    for (int i0 = 0; i0 < nSamples * nSamples; i0++) {
     if(A[i0] != 0) A[i0] = 1.0 - A[i0];
    }
  }

  /* ------------- END OF CODE -------------- */
}

/* End of code generation (cluster_ensemble2cam.c) */
