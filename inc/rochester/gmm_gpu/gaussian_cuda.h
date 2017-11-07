/*
 * gaussian_cuda.h
 *
 *  Created on: Apr 15, 2016
 *      Author: vinaya
 *	   Version:
 *	 Copyright: This program is free software: you can redistribute it and/or modify
 *   			it under the terms of the GNU General Public License as published by
 *   			the Free Software Foundation, either version 3 of the License, or
 *   			(at your option) any later version.
 *
 *    			This program is distributed in the hope that it will be useful,
 *    			but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    			MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    			GNU General Public License for more details.
 *
 *
 *    			You should have received a copy of the GNU General Public License
 *  			along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * Description: 
 */

#ifndef GAUSSIAN_CUDA_H_
#define GAUSSIAN_CUDA_H_

// Since cutil timers aren't thread safe, we do it manually with cuda events
// Removing dependence on cutil is always nice too...
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
    float* et;
} cudaTimer_t;


typedef struct
{
    // Key for array lengths
    //  N = number of events
    //  M = number of clusters
    //  D = number of dimensions
    float* N;        // expected # of pixels in cluster: [M]
    float* pi;       // probability of cluster in GMM: [M]
    float* constant; // Normalizing constant [M]
    float* avgvar;    // average variance [M]
    float* means;   // Spectral mean for the cluster: [M*D]
    float* R;      // Covariance matrix: [M*D*D]
    float* Rinv;   // Inverse of covariance matrix: [M*D*D]
    float* memberships; // Fuzzy memberships: [N*M]
} clusters_t;

// Structure to hold the timers for the different kernel.
//  One of these structs per GPU for profiling.
typedef struct {
    cudaTimer_t e_step;
    cudaTimer_t m_step;
    cudaTimer_t constants;
    cudaTimer_t reduce;
    cudaTimer_t memcpy;
    cudaTimer_t cpu;
} profile_t;

__global__ void seed_clusters( float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events);

__global__ void constants_kernel(clusters_t* clusters, int num_clusters, int num_dimensions/*, float *matrix*/);

__global__ void estep1(float* data, clusters_t* clusters, int num_dimensions, int num_events);

__global__ void estep2(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events, float* likelihood);

__global__ void mstep_means(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events);

__global__ void mstep_N(clusters_t* clusters, int num_dimensions, int num_clusters, int num_events);

__global__ void mstep_covariance1(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events);

__global__ void mstep_covariance2(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events);

#endif /* GAUSSIAN_CUDA_H_ */
