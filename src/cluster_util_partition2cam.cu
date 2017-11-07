/*
 =====================================================================================================================
 Name        : cluster_util_partition2cam.cu
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



/* Function Definitions */
void cluster_util_partition2cam(int* I, int* A, int* N, int nSamples, int nPartitions)
{


  /* CLUSTER_UTIL_PARTITION2CAM Generates a co-association matrix from a partition  */
  /*  */
  /*  Syntax:  A = cluster_util_partition2cam(I)    */
  /*  */
  /*    INPUT ARGUMENTS */
  /*         I : [n,1] column of partition indices */
  /*  */
  /*    OUTPUT ARGUMENTS */
  /*    	    A :  Co-association matrix  [n,n] where n = num samples */
  /*                 Tracks which pairs of samples are in the same cluster. */
  /*                  */
  /*            N : Co-occurrence matrix [n,n]    */
  /*                Tracks which pairs are both present in the sample. If I */
  /*                does not contain zeros or NaNs, then all pairs are present */
  /*                and N is just the identity matrix.  If there are zeros or */
  /*                NaNs present (as a result of bootstrapping), then not all */
  /*                pairs of samples are present, N will contain zeros. */
  /*                  */
  /*  Author:         Lee I Newman */
  /*  Affiliation:    University of Michigan, Depts. Psychology, EECS */
  /*  email:          leenewm@umich.edu */
  /*  Website:        http://www.leenewman.org */
  /*  Created:        May 2009 */
  /*  Revised by/on:  person, date */
  /*  */
  /* ------------- BEGIN CODE -------------- */
  /* % Process input arguments and initialize variables */
  /*  ///////////////////////////////////////////////////////// */
  /*  size variables */
  /*  identify samples that have 0 or NaN in the partition */
  /* % Compute the co-association matrix (similarity format) */
  /*  initialize output variables with 1's on diagonal for non-zero/non-NaN */
  /*  elements and zeros everywhere else */

  memset(A, 0, nSamples * nSamples * sizeof(int));
  memset(N, 0, nSamples * nSamples * sizeof(int));



//  /*  co-occurrence matrix */
//  /*  loop over all pairs of points */
  for (int i = 0; i < nSamples; i++) {

    for (int j = i+1; j < nSamples; j++) {

    	if(I[i] == I[j] && I[i] != 0){
    		 A[j + nSamples * i] = 1;
    		 A[i + nSamples * j] = 1;
    	}

    	if(I[i]!= 0 && I[j] != 0){
    		N[j + nSamples * i] = 1;
    		N[i + nSamples * j] = 1;
    	}

    }

    A[i + nSamples * i] = 1.0;
   	N[i + nSamples * i] = 1.0;

    /*  end loop over j */
  }

  /*  end loop over i */
  /* ------------- END OF CODE -------------- */
}

/* End of code generation (cluster_util_partition2cam.c) */
