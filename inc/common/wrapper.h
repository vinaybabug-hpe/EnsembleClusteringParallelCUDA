/*
 =====================================================================================================================
 Name        : wrapper.h
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

#ifndef WRAPPER_H_
#define WRAPPER_H_
#include <stdbool.h>


#define EXTRACT_ENSEMBLE_SOLUTION 1

#define MEASURE_TIME 1

#define DEBUG 3

// Each block transposes/copies a tile of TILE_DIM x TILE_DIM elements
// using TILE_DIM x BLOCK_ROWS threads, so that each thread transposes
// TILE_DIM/BLOCK_ROWS elements.  TILE_DIM must be an integral multiple of BLOCK_ROWS

#define TRANPOSE_TILE_DIM    16
#define TRANPOSE_BLOCK_ROWS  16


//int TRANPOSE_MUL_FACTOR    = TRANPOSE_TILE_DIM;

#define FLOOR(a,b) (a-(a%b))



/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#ifndef TRI_COUNT
#define TRI_COUNT(n) (((n) * ((n)+1))/2)
#endif


#define MODEL_STR_LEN 64

//#define d_printf(fmt, args...) if(PRINT_DEBUG) { printf("[%s | %d | %s]: ",__FILE__,  __LINE__,__FUNCTION__); printf(fmt, ## args ); }

/**
* % gmm, kmeansxxx, medoidxxx, spectralxxx and aggxxxyyy  where:
* %  xxx denotes distance metric  (euc,seu,cit,cor,cos,mah,che,spe,ham,jac)
* %      (for kmeans, can only use: (euc,cit,cor,cos,ham)
* %   yyy denotes linkage metric     (avg,cen,com,med,sin,war,wei)
*/

#define KMEANS_SHRT 		"kme"
#define KMEDOIDS_SHRT 		"med"
#define GMM_SHRT 			"gmm"
#define SPECTRAL_SHRT 		"spe"
#define AGGLO_SHRT 			"agg"

#define KMEANS_LNG 			"kmeans"
#define KMEDOIDS_LNG 		"medoid"
#define GMM_LNG 			"gmm"
#define SPECTRAL_LNG 		"spectral"
#define AGGLO_LNG 			"agg"

#define DIST_MTRC_EUC		"euc" // euclidean
#define DIST_MTRC_SEU		"seu" // seuclidean
#define DIST_MTRC_CIT		"cit" // cityblock
#define DIST_MTRC_COR		"cor" // correlation
#define DIST_MTRC_ACOR		"aco" // acorrelation
#define DIST_MTRC_UCOR		"uco" // ucorrelation
#define DIST_MTRC_AUCOR		"auc" // aucorrelation
#define DIST_MTRC_COS		"cos" // cosine
#define DIST_MTRC_HAM		"ham" // hamming
#define DIST_MTRC_MAH		"mah" // mahalanobis
#define DIST_MTRC_JAC		"jac" // jaccard
#define DIST_MTRC_CHE		"che" // chebychev
#define DIST_MTRC_SPE		"spe" // spearman
#define DIST_MTRC_KEN		"ken" // spearman
#define DIST_MTRC_CODE_LN		3

#define CNTR_FUN_MEAN 		"mean"
#define CNTR_FUN_MEDIAN 	"median"

#define LNK_CODE_AVG	"avg" // average
#define LNK_CODE_CEN	"cen" // centroid
#define LNK_CODE_COM	"com" // complete
#define LNK_CODE_SIN	"sin" // single
#define LNK_CODE_MED	"med" // median
#define LNK_CODE_WAR	"war" // ward
#define LNK_CODE_WEI	"wei" // weighted
#define LNK_CODE_LN		3


#define NaN								9999

//#define THREADSPERBLOCK 256
#define THREADS_PER_BLOCK 1024
#define THREADS_PER_BLOCK_1 1
#define THREADS_PER_BLOCK_2 2
#define THREADS_PER_BLOCK_4 4
#define THREADS_PER_BLOCK_8 8
#define THREADS_PER_BLOCK_16 16
#define THREADS_PER_BLOCK_32 32
#define THREADS_PER_BLOCK_64 64
#define THREADS_PER_BLOCK_128 128
#define THREADS_PER_BLOCK_256 256
#define THREADS_PER_BLOCK_512 512
// Agglomorative Clustering Related

#define MEX_STRUCT_PST_NME 				"PST"
#define MEX_STRUCT_DATA_NME 			"X"
#define MEX_STRUCT_BOOTIDXS_NME 		"bootIdxs"
#define MEX_STRUCT_OUTLIERCUTOFF_NME	"outlierCutoff"
#define MEX_STRUCT_OUTLIERMETRIC_NME	"outlierMetric"

/*
 * Field names for PST passed by RDM Ensemble Clustering
 */
#define MEX_STRUCT_PST_FIELD_NAME 							"name"
#define MEX_STRUCT_PST_FIELD_VERBOSE						"verboseQ"
#define MEX_STRUCT_PST_FIELD_OUTLIER_CUTOFF 				"outlierCutoff"
#define MEX_STRUCT_PST_FIELD_OUTLIER_METRIC 				"outlierMetric"
#define MEX_STRUCT_PST_FIELD_PREPROCESS_LIST 				"preprocessList"
#define MEX_STRUCT_PST_FIELD_ENSEMBLE						"ensemble"
#define MEX_STRUCT_PST_FIELD_EXTRACT 						"extract"
#define MEX_STRUCT_PST_FIELD_VALIDATE 						"validate"
#define MEX_STRUCT_PST_FIELD_COMPARE 						"compare"
#define MEX_STRUCT_PST_FIELD_KMEANS 						"kmeans"
#define MEX_STRUCT_PST_FIELD_MEDOID 						"medoid"
#define MEX_STRUCT_PST_FIELD_GMM 							"gmm"
#define MEX_STRUCT_PST_FIELD_SPECTRAL 						"spectral"
#define MEX_STRUCT_PST_FIELD_AGGLO 							"agg"

#define MEX_STRUCT_PST_FIELD_EXTRACT_KLIST 					"kList"
#define MEX_STRUCT_PST_FIELD_ENSEMBLE_KLIST 				"kList"
#define MEX_STRUCT_PST_FIELD_ENSEMBLE_MODELLIST 			"modelList"
#define MEX_STRUCT_PST_FIELD_ENSEMBLE_NBOOTSTRAPS 			"nBootstraps"
#define MEX_STRUCT_PST_FIELD_ENSEMBLE_NREPS					"nReps"
#define MEX_STRUCT_PST_FIELD_ENSEMBLE_INCLUDEREPSQ			"includeRepsQ"
#define MEX_STRUCT_PST_FIELD_ENSEMBLE_INCLUDECENTERSQ		"includeCentersQ"
#define MEX_STRUCT_PST_FIELD_ENSEMBLE_VERBOSEQ				"verboseQ"



#endif /* WRAPPER_H_ */


