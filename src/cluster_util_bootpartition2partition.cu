/*
 =====================================================================================================================
 Name        : cluster_util_bootpartition2partition.cu
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


/* Function Definitions */
void cluster_util_bootpartition2partition(int n, int *b, int* bi, int *i)
{


  /* CLUSTER_UTIL_BOOTPARTITION2PARTITION - Converts boot partition to sample partition */
  /*  */
  /*    Syntax:  M = cluster_util_bootpartition2partition(X,I,metric) */
  /*  */
  /*    B      :[n,1] column of bootstrap data set indices */
  /*                  where elements are indices of original samples. E.g. if */
  /*                  B(3)= 220  then sample 220 from the original data set is */
  /*                  the third sample in the boostrapped data set */
  /*  */
  /*    BI     : [1,n] or[n,1] array of cluster labels/indices assigned to each */
  /*             boostrap data element by a clustering algorithm. */
  /*  */
  /*    I      : [1,n] or [n,1] array of cluster labels for each element of */
  /*             original data set.  E.g., if BI(3) = 4 and B(3) = 220 then the */
  /*             element 200 in the original sample was assigned to cluster 4 */
  /*             and therefore I(220) = 4 */
  /*  */
  /*  not all of original samples are included in a bootstrap data set, so */
  /*  initialize I with NaN so that these excluded elements remain NaN after */
  /*  the loop below is finished.   Note: if indices are at some point converted to  */
  /*  integers, the NaNs will become zeros.  */


  /*  main loop */
  for (int count = 0; count < n; count++) {
    i[b[count]] = bi[count];
  }
  return;
}

/* End of code generation (cluster_util_bootpartition2partition.c) */
