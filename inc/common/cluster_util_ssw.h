/*
 =====================================================================================================================
 Name        : cluster_util_ssw.h
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

#ifndef CLUSTER_UTIL_SSW_H
#define CLUSTER_UTIL_SSW_H



/* Function Declarations */
void cluster_util_ssw(float *X, int nSamples, int nFeatures, int  *I, double *ssw_data, int nClusters);

#endif

/* End of code generation (cluster_util_ssw.h) */
