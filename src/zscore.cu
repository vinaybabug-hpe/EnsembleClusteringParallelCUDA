/*
 =====================================================================================================================
 Name        : zscore.cu
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
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include "cusparse.h"
#include "cuda_runtime.h"
#include <vector>

#include "common/wrapper.h"
#include "common/wrapperFuncs.h"


struct zscore_functor
{
  const float mu;
  const float sigma;

  zscore_functor(float _mu, float _sigma) : mu(_mu),sigma(_sigma) {}

  __host__ __device__
  float operator()(const float& x, const float& z) const  {
	  return (x - mu)/sigma;

  }
};

struct sigma_functor
{
  const float mu;

  sigma_functor(float _mu) : mu(_mu) {}

  __host__ __device__
  float operator()(const float& x) const  {
	float r = x - mu;
    return r*r;
  }
};


/* Function Definitions */
void zscore(int n, thrust::host_vector<float>& z)
{
  float sigma;
  int k;
  float mu;


//  sigma = x[0];

  thrust::device_vector<float> Z = z;

  mu = thrust::reduce(Z.begin(), Z.end());

  mu /= n;

  // setup arguments
  float init = 0;

  // compute norm
  sigma = std::sqrt( thrust::transform_reduce(Z.begin(), Z.end(), sigma_functor(mu), init, thrust::plus<float>())/n);

  if (sigma == 0.0) {
    sigma = 1.0;
  }

  thrust::transform(Z.begin(), Z.end(), Z.begin(), Z.begin(), zscore_functor(mu, sigma));

  for (k = 0; k < n; k++) {
    z[k] = Z[k];
  }

}


