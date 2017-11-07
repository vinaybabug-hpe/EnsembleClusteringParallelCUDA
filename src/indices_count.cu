/*
 =====================================================================================================================
 Name        : indices_count.cu
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
#include <iostream>
#include <unordered_set>
#include <set>

#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/count.h>


struct cnt_eq_lable_functor
{
  const int label;

  cnt_eq_lable_functor(int _label) : label(_label) {}

  __host__ __device__ bool operator()(const int a) const {
         return (a==label);
     }
};


/* Function Definitions */
void indices_count(int *X, int nSamples, int *valueList, int nClusters)
{

  /* MY_COUNT Computes the counts of each unique value in X */
  /*    INPUT ARGS */
  /*        X           :  column or row vector of values */
  /*        valueList   :  list of unique values to be counted */
  /*                       OPTIONAL: if not provided, counts are based on data in X */
  /*  */
  /*    OUTPUT ARGS */
  /*        counts      :  count of each value */
  /*        valueList   :  list of values associated with each count */
  /*                       will be in same order as valueList if valueList given */
  /*                       as argument, otherwise will be in sorted order. */
  /*  Ensure X is a column vector */
  /*  if size(X,2)>1 */
  /*      X=X'; */
  /*  end */
  /*  If value list not provided, base list on unique values...in sorted order */
  /*  if ~exist('valueList','var') */

	std::unordered_set<int> s(X, X + nSamples);
	std::set<int> nValues(s.begin(), s.end());

	for(int iValue = 0; iValue < nClusters; iValue++){
		int label = *std::next(nValues.begin(), iValue);
		int counter = 0;

		// doing count if
//		for(int count = 0; count < nSamples; count++){
//			if(label == X[count]){
//				counter++;
//			}
//		}
//		valueList[iValue] = counter;

		thrust::device_vector<float> d_x(X, X + nSamples);
		valueList[iValue] =  thrust::count(d_x.begin(), d_x.end(), label);//(d_x.begin(), d_x.end(), cnt_eq_lable_functor(label));

	}

	s.clear();
	nValues.clear();
	return;
}

/* End of code generation (my_count.c) */
