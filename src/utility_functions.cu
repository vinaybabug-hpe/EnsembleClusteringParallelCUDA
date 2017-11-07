/*
 =====================================================================================================================
 Name        : utility_functions.cu
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

#include "common/wrapper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h> //TODO: migth give problems
#include <unordered_set>


void getClusteringMethodsList(int length, char *methodList[], char **clustList) {
	//char clustList[5][4]={"xxx","xxx","xxx","xxx","xxx"};
	int clustList_idx = 0;
	int count;

	for (count = 0; count < length; count++) {


		if (strncmp(clustList[clustList_idx == 0 ? 0 : clustList_idx - 1],
				methodList[count], 3) == 0) {
			continue;
		} else {
			strncpy(clustList[clustList_idx], methodList[count], 3);
			clustList[clustList_idx][3]='\0';
			//memcpy( clustList[clustList_idx], methodList[count], 3*sizeof(char) );
			clustList_idx++;
		}
	}

}

int doKmeans(char **clustList) {
	int count;
	for (count = 0; count < 5; count++) {
		if (strcmp(clustList[count], KMEANS_SHRT) == 0) {
			return 1;
		}
	}
	return 0;
}

int doKmedoids(char **clustList) {
	int count;
	for (count = 0; count < 5; count++) {
		if (strcmp(clustList[count], KMEDOIDS_SHRT) == 0) {
			return 1;
		}
	}
	return 0;
}

int doGMM(char **clustList) {
	int count;
	for (count = 0; count < 5; count++) {
		if (strcmp(clustList[count], GMM_SHRT) == 0) {
			return 1;
		}
	}
	return 0;
}

int doSpectral(char **clustList) {
	int count;
	for (count = 0; count < 5; count++) {
		if (strcmp(clustList[count], SPECTRAL_SHRT) == 0) {
			return 1;
		}
	}
	return 0;
}

int doAgglom(char **clustList) {
	int count;
	for (count = 0; count < 5; count++) {
		if (strcmp(clustList[count], AGGLO_SHRT) == 0) {
			return 1;
		}
	}
	return 0;
}

int getClustMthdCnt(char*clustMethod, char **methodList, int length) {
	int count;
	int count_kmeans_mthd = 0;
	for (count = 0; count < length; count++) {
		if (strncmp(clustMethod, methodList[count], 3) == 0) {
			count_kmeans_mthd++;
		}
	}
	return count_kmeans_mthd;
}

int getMthdLstbyClust(char*clustMethod, char **methodList,
		int length, char **mthdLstbyClust) {
	int count;
	int count_kmeans_mthd = 0;
	for (count = 0; count < length; count++) {
		if (strncmp(clustMethod, methodList[count], 3) == 0) {
			strcpy(mthdLstbyClust[count_kmeans_mthd], methodList[count]);
//			d_printf(" %s",mthdLstbyClust[count_kmeans_mthd]);
			count_kmeans_mthd++;

		}
	}
	return count_kmeans_mthd;
}

int getModelType(char*clustMethod, char *methodList) {

		if (strncmp(clustMethod, methodList, 3) == 0) {

			return 1;

		}

	return 0;
}

/**
 * % gmm, kmeansxxx, medoidxxx, spectralxxx and aggxxxyyy  where:
 * %  xxx denotes distance metric  (euc,seu,cit,cor,cos,mah,che,spe,ham,jac)
 * %      (for kmeans, can only use: (euc,cit,cor,cos,ham)
 * %   yyy denotes linkage metric     (avg,cen,com,med,sin,war,wei)
 */
void getDistMtrcnCntrFunByKmeans(char*clustMthd, char *model,
		char *distmetric, char *centerfun) {
	char *dist;

	if (strcmp(clustMthd, KMEANS_SHRT) == 0) {

		if ((dist = strstr(model, DIST_MTRC_EUC)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);

		} else if ((dist = strstr(model, DIST_MTRC_SEU)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_CIT)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEDIAN);
		} else if ((dist = strstr(model, DIST_MTRC_COR)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_COS)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_MAH)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_JAC)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_CHE)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_SPE)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_HAM)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEDIAN);
		}

		strtok(distmetric, " \t\n");
		strtok(centerfun, " \t\n");
		distmetric[DIST_MTRC_CODE_LN] = '\0';

	}
}

/**
 * % gmm, kmeansxxx, medoidxxx, spectralxxx and aggxxxyyy  where:
 * %  xxx denotes distance metric  (euc,seu,cit,cor,cos,mah,che,spe,ham,jac)
 * %      (for kmeans, can only use: (euc,cit,cor,cos,ham)
 * %   yyy denotes linkage metric     (avg,cen,com,med,sin,war,wei)
 */
void getDistMtrcnCntrFunByKmedoid(char*clustMthd, char *model,
		char *distmetric, char *centerfun) {
	char *dist;

	if (strcmp(clustMthd, KMEDOIDS_SHRT) == 0) {

		if ((dist = strstr(model, DIST_MTRC_EUC)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);

		} else if ((dist = strstr(model, DIST_MTRC_SEU)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_CIT)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEDIAN);
		} else if ((dist = strstr(model, DIST_MTRC_COR)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_COS)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_MAH)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_JAC)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_CHE)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_SPE)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_HAM)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEDIAN);
		}

		strtok(distmetric, " \t\n");
		strtok(centerfun, " \t\n");
		distmetric[DIST_MTRC_CODE_LN] = '\0';

	}
}

/**
 * % gmm, kmeansxxx, medoidxxx, spectralxxx and aggxxxyyy  where:
 * %  xxx denotes distance metric  (euc,seu,cit,cor,cos,mah,che,spe,ham,jac)
 * %      (for kmeans, can only use: (euc,cit,cor,cos,ham)
 * %   yyy denotes linkage metric     (avg,cen,com,med,sin,war,wei)
 */
void getDistMtrcnCntrFunByAgglo(char*clustMthd, char *model,
		char *distmetric, char *centerfun, char *linkcode) {
	char *dist;
	char * linkageMetric;
	if (strcmp(clustMthd, AGGLO_SHRT) == 0) {

		if ((dist = strstr(model, DIST_MTRC_EUC)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);

		} else if ((dist = strstr(model, DIST_MTRC_SEU)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_CIT)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEDIAN);
		} else if ((dist = strstr(model, DIST_MTRC_COR)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_COS)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_MAH)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_JAC)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_CHE)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_SPE)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_HAM)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEDIAN);
		}

		// get link code
		if ((linkageMetric = strstr(model, LNK_CODE_AVG)) != NULL)
		{
			strncpy(linkcode, linkageMetric, LNK_CODE_LN);
		} else if ((linkageMetric = strstr(model, LNK_CODE_CEN)) != NULL)
		{
			strncpy(linkcode, linkageMetric, LNK_CODE_LN);
		} else if ((linkageMetric = strstr(model, LNK_CODE_COM)) != NULL)
		{
			strncpy(linkcode, linkageMetric, LNK_CODE_LN);
		} else if ((linkageMetric = strstr(model, LNK_CODE_SIN)) != NULL)
		{
			strncpy(linkcode, linkageMetric, LNK_CODE_LN);
		} else if ((linkageMetric = strstr(model, LNK_CODE_MED)) != NULL)
		{
			strncpy(linkcode, linkageMetric, LNK_CODE_LN);
		} else if ((linkageMetric = strstr(model, LNK_CODE_WAR)) != NULL)
		{
			strncpy(linkcode, linkageMetric, LNK_CODE_LN);
		}else if ((linkageMetric = strstr(model, LNK_CODE_WEI)) != NULL)
		{
			strncpy(linkcode, linkageMetric, LNK_CODE_LN);
		}

		strtok(distmetric, " \t\n");
		strtok(centerfun, " \t\n");
		strtok(linkcode, " \t\n");
		distmetric[DIST_MTRC_CODE_LN] = '\0';
		linkcode[LNK_CODE_LN] = '\0';



	}
}

/**
 * % gmm, kmeansxxx, medoidxxx, spectralxxx and aggxxxyyy  where:
 * %  xxx denotes distance metric  (euc,seu,cit,cor,cos,mah,che,spe,ham,jac)
 * %      (for kmeans, can only use: (euc,cit,cor,cos,ham)
 * %   yyy denotes linkage metric     (avg,cen,com,med,sin,war,wei)
 */
void getDistMtrcnCntrFunBySpectral(char*clustMthd, char *model,
		char *distmetric, char *centerfun) {
	char *dist;

	if (strcmp(clustMthd, SPECTRAL_SHRT) == 0) {

		if ((dist = strstr(model, DIST_MTRC_EUC)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_SEU)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_CIT)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEDIAN);
		} else if ((dist = strstr(model, DIST_MTRC_COR)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_COS)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_MAH)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_JAC)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_CHE)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_SPE)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEAN);
		} else if ((dist = strstr(model, DIST_MTRC_HAM)) != NULL)
		{
			strncpy(distmetric, dist, DIST_MTRC_CODE_LN);
			strcpy(centerfun, CNTR_FUN_MEDIAN);
		}

		strtok(distmetric, " \t\n");
		strtok(centerfun, " \t\n");
		distmetric[DIST_MTRC_CODE_LN] = '\0';

	}
}


/**
 *
 */
void cluster_util_bootpartition2partition(unsigned short *boot, float *bootIdxs,
		int *idx, int length) {
	int count = 0;

	printf("\n");

	for (count = 0; count < length; count++) {
		idx[(int) boot[count]-1] = (int) bootIdxs[count];
		//printf("[%d %d] ",(int)boot[count], (int)bootIdxs[count]);
		//printf("%d ",(int)idx[count]);

	}

}

int unique_length(int *a, size_t len)
{
	    std::unordered_set<int> s(a, a + len);
	    return s.size();
}

// -------------------------------------
// utility routine to transpose matrix
// -------------------------------------

void matrixTranspose(float *gold, float *idata, const  int size_x, const  int size_y)
{
    for (int y = 0; y < size_y; ++y)
    {
        for (int x = 0; x < size_x; ++x)
        {
            gold[(x * size_y) + y] = idata[(y * size_x) + x];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void
matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{

    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }

}

////////////////////////////////////////////////////////////////////////////////
//! Performs parallel sort on CPU and also sorts alpha arrays basded on actual
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
//Selection sort function
void sort_models(char actual[][MODEL_STR_LEN], int size, char alph1[][MODEL_STR_LEN], char alph2[][MODEL_STR_LEN], char alph3[][MODEL_STR_LEN], char **alph4)
{
	int startScan;
	int minIndex;
	char minValue[MODEL_STR_LEN];
//	char temp[MODEL_STR_LEN];

	for (startScan = 0; startScan < (size - 1); startScan++)    //Moves through the elements
	{
		minIndex = startScan;
		strcpy(minValue, actual[startScan]);

		//temp = alph[startScan];

		int index = 0;

		for (index = startScan + 1; index < size; index++)  //Compares the elements
		{
			if (strcmp(actual[index], minValue)>0)
			{
				strcpy(minValue, actual[index]);
//				minValue = actual[index];
				minIndex = index;
//				temp = alph[index];
			}
		}

		std::swap(actual[minIndex], actual[startScan]);
		std::swap(alph1[minIndex],alph1[startScan]);
		std::swap(alph2[minIndex],alph2[startScan]);
		std::swap(alph3[minIndex],alph3[startScan]);
		std::swap(alph4[minIndex],alph4[startScan]);

//		actual[minIndex] = actual[startScan];
//		actual[startScan] = minValue;
//
//		alph[minIndex] = alph[startScan];
//		alph[startScan] = temp;
	}
}

char link_str2c(char* _method) {
	char method;
	//int i, j;
	// Assign lib specific parameters to link method
	// and distance function
	if (_method == NULL) {
		method = 's';
	} else if (strcmp(_method, LNK_CODE_AVG) == 0) {
		method = 'a';
	} else if (strcmp(_method, LNK_CODE_CEN) == 0) {
		method = 'c';
	} else if (strcmp(_method, LNK_CODE_COM) == 0) {
		method = 'm';
	} else if (strcmp(_method, LNK_CODE_SIN) == 0) {
		method = 's';
	} else if (strcmp(_method, LNK_CODE_MED) == 0) {
		method = 'a'; // TODO: median linkage should be implemented later
	} else if (strcmp(_method, LNK_CODE_WAR) == 0) {
		method = 'w'; // TODO: ward linkage should be implemented later
	} else if (strcmp(_method, LNK_CODE_WEI) == 0) {
		method = 'a'; // TODO: weighted linkage should be implemented later
	}
	return method;
}

char dist_str2c(char* _dist) {
	char dist;
	if (_dist == NULL) {
		dist = 'e';
	} else if (strcmp(_dist, DIST_MTRC_EUC) == 0) {
		dist = 'e';
	} else if (strcmp(_dist, DIST_MTRC_SEU) == 0) {
		dist = 'l';
	} else if (strcmp(_dist, DIST_MTRC_CIT) == 0) {
		dist = 'b';
	} else if (strcmp(_dist, DIST_MTRC_COR) == 0) {
		dist = 'c';
	} else if (strcmp(_dist, DIST_MTRC_ACOR) == 0) {
		dist = 'a';
	} else if (strcmp(_dist, DIST_MTRC_UCOR) == 0) {
		dist = 'u';
	} else if (strcmp(_dist, DIST_MTRC_AUCOR) == 0) {
		dist = 'x';
	} else if (strcmp(_dist, DIST_MTRC_COS) == 0) {
		dist = 'o';
	} else if (strcmp(_dist, DIST_MTRC_KEN) == 0) {
		dist = 'k';
	} else if (strcmp(_dist, DIST_MTRC_MAH) == 0) {
		dist = 'm';
	} else if (strcmp(_dist, DIST_MTRC_JAC) == 0) {
		dist = 'j';
	} else if (strcmp(_dist, DIST_MTRC_CHE) == 0) {
		dist = 'h';
	} else if (strcmp(_dist, DIST_MTRC_SPE) == 0) {
		dist = 's';
	} else if (strcmp(_dist, DIST_MTRC_HAM) == 0) {
		dist = 'g';
	}
	return dist;
}
