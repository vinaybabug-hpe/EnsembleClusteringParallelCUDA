/*
 =====================================================================================================================
 Name        : distCalcMthds.cu
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

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cfloat>
#include "common/wrapper.h"

/* ********************************************************************* */

__device__ static
float euclid(int cols, int rows1, int rows2, float* data1, float* data2,
cudaTextureObject_t texWt, int index1, int index2, int transpose)

/*
Purpose
=======

The euclid routine calculates the weighted Euclidean distance between two
rows or columns in a matrix.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	float result = 0.;
	float tweight = 0;
	int i;
	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				float term = data1[index1 * cols + i] - data2[index2 * cols + i];


				result += tex1Dfetch<float>(texWt, i) * term*term;
				tweight += tex1Dfetch<float>(texWt, i);

		}

	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				float term = data1[rows1 * i + index1] - data2[rows2 * i + index2];
				result += tex1Dfetch<float>(texWt, i) * term*term;
				tweight += tex1Dfetch<float>(texWt, i);

		}
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result = sqrt(result);
	result /= tweight;
	return result;
}

__device__ static
float euclid(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
	 Purpose
	 =======

	 The euclid routine calculates the weighted Euclidean distance between two
	 rows or columns in a matrix.

	 Arguments
	 =========

	 n      (input) int
	 The number of elements in a row or column. If transpose==0, then n is the number
	 of columns; otherwise, n is the number of rows.

	 index1     (input) int
	 Index of the first row or column.

	 index2     (input) int
	 Index of the second row or column.

	 transpose (input) int
	 If transpose==0, the distance between two rows in the matrix is calculated.
	 Otherwise, the distance between two columns in the matrix is calculated.

 	 ============================================================================
 */
{
	float result = 0.;
	float tweight = 0;
	int i;

	float data1 = 0;
	float data2 = 0;
	float weight = 0;

	/* if number of features is less than 128 Calculate the distance between two rows in sequntial manner*/

		if (transpose == 0) /* Calculate the distance between two rows */
		{
#pragma unroll
			for (i = 0; i < cols; i++) {

				data1 = tex1Dfetch<float>(texData, index1 * cols + i);
				data2 = tex1Dfetch<float>(texData, index2 * cols + i);
				weight = tex1Dfetch<float>(texWt, i);

				float term = data1 - data2;
				result += weight * term * term;
				tweight += weight;

			}

		} else {
#pragma unroll
			for (i = 0; i < cols; i++) {

				data1 = tex1Dfetch<float>(texData, rows1 * i + index1);
				data2 = tex1Dfetch<float>(texData, rows2 * i + index2);
				weight = tex1Dfetch<float>(texWt, i);

				float term = data1 - data2;
				result += weight * term * term;
				tweight += weight;
			}
		}


	if (!tweight && result == 0)
		return 0; /* usually due to empty clusters */
	result = sqrt(result);
	result /= tweight;

//	printf("\n RESULT = %.5f\n", result);
	return result;
}

/**
 * Override method that works using a combination of texture and shared memory data
 */
__device__ static
float euclid(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* _data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
	 Purpose
	 =======

	 The euclid routine calculates the weighted Euclidean distance between two
	 rows or columns in a matrix.

	 Arguments
	 =========

	 n      (input) int
	 The number of elements in a row or column. If transpose==0, then n is the number
	 of columns; otherwise, n is the number of rows.

	 index1     (input) int
	 Index of the first row or column.

	 index2     (input) int
	 Index of the second row or column.

	 transpose (input) int
	 If transpose==0, the distance between two rows in the matrix is calculated.
	 Otherwise, the distance between two columns in the matrix is calculated.

 	 ============================================================================
 */
{
	float result = 0.;
	float tweight = 0;
	int i;
	int mask1 =0;
	int mask2 =0;
	float data1 = 0;
	float data2 = 0;
	float weight = 0;

	/* if number of features is less than 128 Calculate the distance between two rows in sequntial manner*/

		if (transpose == 0) /* Calculate the distance between two rows */
		{
#pragma unroll
			for (i = 0; i < cols; i++) {

				data1 = tex1Dfetch<float>(texData1, index1 * cols + i);
				data2 = _data2[index2 * cols + i];
				weight = tex1Dfetch<float>(texWt, i);

				float term = data1 - data2;
				result += weight * term * term;
				tweight += weight;

			}

		} else {
#pragma unroll
			for (i = 0; i < cols; i++) {

				data1 = tex1Dfetch<float>(texData1, rows1 * i + index1);
				data2 = _data2[ rows2 * i + index2];
				weight = tex1Dfetch<float>(texWt, i);

				float term = data1 - data2;
				result += weight * term * term;
				tweight += weight;

			}
		}


	if (!tweight && result == 0)
		return 0; /* usually due to empty clusters */
	result = sqrt(result);
	result /= tweight;

//	printf("\n RESULT = %.5f\n", result);
	return result;
}

/* ********************************************************************* */
__device__ static
float seuclid(int cols, int rows1, int rows2, float* data1, float* data2,
cudaTextureObject_t texWt, int index1, int index2, int transpose)

/*
Purpose
=======

The seuclid routine calculates the weighted squared Euclidean distance between two
rows or columns in a matrix.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	float result = 0.;
	float tweight = 0;
	int i;
	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				float term = data1[index1 * cols + i] - data2[index2 * cols + i];

				result += tex1Dfetch<float>(texWt, i) * term*term;
				tweight += tex1Dfetch<float>(texWt, i);

		}

	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				float term = data1[rows1 * i + index1] - data2[rows2 * i + index2];
				result += tex1Dfetch<float>(texWt, i) * term*term;
				tweight += tex1Dfetch<float>(texWt, i);

		}
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	return result;
}

__device__ static
float seuclid(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
	 Purpose
	 =======

	 The euclid routine calculates the weighted Euclidean distance between two
	 rows or columns in a matrix.

	 Arguments
	 =========

	 n      (input) int
	 The number of elements in a row or column. If transpose==0, then n is the number
	 of columns; otherwise, n is the number of rows.

	 index1     (input) int
	 Index of the first row or column.

	 index2     (input) int
	 Index of the second row or column.

	 transpose (input) int
	 If transpose==0, the distance between two rows in the matrix is calculated.
	 Otherwise, the distance between two columns in the matrix is calculated.

 	 ============================================================================
 */
{
	float result = 0.;
	float tweight = 0;
	int i;
	int mask1 =0;
	int mask2 =0;
	float data1 = 0;
	float data2 = 0;
	float weight = 0;

	/* if number of features is less than 128 Calculate the distance between two rows in sequntial manner*/

		if (transpose == 0) /* Calculate the distance between two rows */
		{
#pragma unroll
			for (i = 0; i < cols; i++) {

				data1 = tex1Dfetch<float>(texData, index1 * cols + i);
				data2 = tex1Dfetch<float>(texData, index2 * cols + i);
				weight = tex1Dfetch<float>(texWt, i);


				float term = data1 - data2;
				result += weight * term * term;
				tweight += weight;

			}

		} else {
#pragma unroll
			for (i = 0; i < cols; i++) {

				data1 = tex1Dfetch<float>(texData, rows1 * i + index1);
				data2 = tex1Dfetch<float>(texData, rows2 * i + index2);
				weight = tex1Dfetch<float>(texWt, i);

				float term = data1 - data2;
				result += weight * term * term;
				tweight += weight;

			}
		}


	if (!tweight && result == 0)
		return 0; /* usually due to empty clusters */
	result /= tweight;

//	printf("\n RESULT = %.5f\n", result);
	return result;
}

/**
 * Override method that works using a combination of texture and shared memory data
 */
__device__ static
float seuclid(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* _data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
	 Purpose
	 =======

	 The euclid routine calculates the weighted Euclidean distance between two
	 rows or columns in a matrix.

	 Arguments
	 =========

	 n      (input) int
	 The number of elements in a row or column. If transpose==0, then n is the number
	 of columns; otherwise, n is the number of rows.

	 index1     (input) int
	 Index of the first row or column.

	 index2     (input) int
	 Index of the second row or column.

	 transpose (input) int
	 If transpose==0, the distance between two rows in the matrix is calculated.
	 Otherwise, the distance between two columns in the matrix is calculated.

 	 ============================================================================
 */
{
	float result = 0.;
	float tweight = 0;
	int i;
	int mask1 =0;
	int mask2 =0;
	float data1 = 0;
	float data2 = 0;
	float weight = 0;

	/* if number of features is less than 128 Calculate the distance between two rows in sequntial manner*/

		if (transpose == 0) /* Calculate the distance between two rows */
		{
#pragma unroll
			for (i = 0; i < cols; i++) {

				data1 = tex1Dfetch<float>(texData1, index1 * cols + i);
				data2 = _data2[ index2 * cols + i];
				weight = tex1Dfetch<float>(texWt, i);

				float term = data1 - data2;
				result += weight * term * term;
				tweight += weight;

			}

		} else {
#pragma unroll
			for (i = 0; i < cols; i++) {

				data1 = tex1Dfetch<float>(texData1, rows1 * i + index1);
				data2 = _data2[ rows2 * i + index2];
				weight = tex1Dfetch<float>(texWt, i);

				float term = data1 - data2;
				result += weight * term * term;
				tweight += weight;

			}
		}


	if (!tweight && result == 0)
		return 0; /* usually due to empty clusters */
	result /= tweight;

//	printf("\n RESULT = %.5f\n", result);
	return result;
}


/* ********************************************************************* */

__device__ static
float spearman(int cols, int rows1, int rows2, float* data1, float* data2,
cudaTextureObject_t texWt, int index1, int index2, int transpose)

/*
Purpose
=======

The seuclid routine calculates the weighted squared Euclidean distance between two
rows or columns in a matrix.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	float result = 0.;
	float tweight = 0;
	int i;

	int denom = cols * (cols * cols - 1);

	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				float term = data1[index1 * cols + i] - data2[index2 * cols + i];


				result += tex1Dfetch<float>(texWt, i) * term*term;
				tweight += tex1Dfetch<float>(texWt, i);

		}

	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				float term = data1[rows1 * i + index1] - data2[rows2 * i + index2];
				result += tex1Dfetch<float>(texWt, i) * term*term;
				tweight += tex1Dfetch<float>(texWt, i);

		}
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result = 1.0 - (6.0 * (result / ((double) denom)));
	result /= tweight;
	return result;
}

__device__ static
float spearman(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
	 Purpose
	 =======

	 The spearman routine calculates the weighted spearman distance between two
	 rows or columns in a matrix.

	 Arguments
	 =========

	 n      (input) int
	 The number of elements in a row or column. If transpose==0, then n is the number
	 of columns; otherwise, n is the number of rows.

	 index1     (input) int
	 Index of the first row or column.

	 index2     (input) int
	 Index of the second row or column.

	 transpose (input) int
	 If transpose==0, the distance between two rows in the matrix is calculated.
	 Otherwise, the distance between two columns in the matrix is calculated.

 	 ============================================================================
 */
{
	float result = 0.;
	float tweight = 0;
	int i;

	float data1 = 0;
	float data2 = 0;
	float weight = 0;

	int denom = cols * (cols * cols - 1);

	/* if number of features is less than 128 Calculate the distance between two rows in sequntial manner*/

		if (transpose == 0) /* Calculate the distance between two rows */
		{
#pragma unroll
			for (i = 0; i < cols; i++) {

				data1 = tex1Dfetch<float>(texData, index1 * cols + i);
				data2 = tex1Dfetch<float>(texData, index2 * cols + i);
				weight = tex1Dfetch<float>(texWt, i);

				float term = data1 - data2;
				result += weight * term * term;
				tweight += weight;

			}

		} else {
#pragma unroll
			for (i = 0; i < cols; i++) {

				data1 = tex1Dfetch<float>(texData, rows1 * i + index1);
				data2 = tex1Dfetch<float>(texData, rows2 * i + index2);
				weight = tex1Dfetch<float>(texWt, i);

				float term = data1 - data2;
				result += weight * term * term;
				tweight += weight;

			}
		}


	if (!tweight && result == 0)
		return 0; /* usually due to empty clusters */
	result = 1.0 - (6.0 * (result / ((double) denom)));
	result /= tweight;

//	printf("\n RESULT = %.5f\n", result);
	return result;
}

/**
 * Override method that works using a combination of texture and shared memory data
 */
__device__ static
float spearman(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* _data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
	 Purpose
	 =======

	 The spearman routine calculates the weighted spearman distance between two
	 rows or columns in a matrix.

	 Arguments
	 =========

	 n      (input) int
	 The number of elements in a row or column. If transpose==0, then n is the number
	 of columns; otherwise, n is the number of rows.

	 index1     (input) int
	 Index of the first row or column.

	 index2     (input) int
	 Index of the second row or column.

	 transpose (input) int
	 If transpose==0, the distance between two rows in the matrix is calculated.
	 Otherwise, the distance between two columns in the matrix is calculated.

 	 ============================================================================
 */
{
	float result = 0.;
	float tweight = 0;
	int i;
	int mask1 =0;
	int mask2 =0;
	float data1 = 0;
	float data2 = 0;
	float weight = 0;

	int denom = cols * (cols * cols - 1);

	/* if number of features is less than 128 Calculate the distance between two rows in sequntial manner*/

		if (transpose == 0) /* Calculate the distance between two rows */
		{
#pragma unroll
			for (i = 0; i < cols; i++) {

				data1 = tex1Dfetch<float>(texData1, index1 * cols + i);
				data2 = _data2 [index2 * cols + i];
				weight = tex1Dfetch<float>(texWt, i);


					float term = data1 - data2;
					result += weight * term * term;
					tweight += weight;

			}

		} else {
#pragma unroll
			for (i = 0; i < cols; i++) {

				data1 = tex1Dfetch<float>(texData1, rows1 * i + index1);
				data2 = _data2[ rows2 * i + index2];
				weight = tex1Dfetch<float>(texWt, i);


					float term = data1 - data2;
					result += weight * term * term;
					tweight += weight;

			}
		}


	if (!tweight && result == 0)
		return 0; /* usually due to empty clusters */
	result = 1.0 - (6.0 * (result / ((double) denom)));
	result /= tweight;

//	printf("\n RESULT = %.5f\n", result);
	return result;
}


/* ********************************************************************* */

//__device__ static
//float cityblock(int cols, int rows1, int rows2, float* data1, float* data2, int* mask1, int* mask2,
//const float weight[], int index1, int index2, int transpose)

__device__ static
float cityblock(int cols, int rows1, int rows2, float* data1, float* data2,
cudaTextureObject_t texWt, int index1, int index2, int transpose)

/*
Purpose
=======

The cityblock routine calculates the weighted "City Block" distance between
two rows or columns in a matrix. City Block distance is defined as the
absolute value of X1-X2 plus the absolute value of Y1-Y2 plus..., which is
equivalent to taking an "up and over" path.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================ */
{
	double result = 0.;
	double tweight = 0;
	int i;
	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term = data1[index1 * cols + i] - data2[index2 * cols + i];
				result += tex1Dfetch<float>(texWt, i) * fabs(term);
				tweight += tex1Dfetch<float>(texWt, i);

		}
	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term = data1[rows1 * i + index1] - data2[rows2 * i + index2];
				result += tex1Dfetch<float>(texWt, i) * fabs(term);
				tweight += tex1Dfetch<float>(texWt, i);

		}
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	return result;
}


__device__ static
float cityblock(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
Purpose
=======

The cityblock routine calculates the weighted "City Block" distance between
two rows or columns in a matrix. City Block distance is defined as the
absolute value of X1-X2 plus the absolute value of Y1-Y2 plus..., which is
equivalent to taking an "up and over" path.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================ */
{
	double result = 0.;
	double tweight = 0;
	int i;
	int mask1 =0;
	int mask2 =0;
	float data1 = 0;
	float data2 = 0;
	float weight = 0;

	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, index1 * cols + i);
			data2 = tex1Dfetch<float>(texData, index2 * cols + i);
			weight = tex1Dfetch<float>(texWt, i);


				double term = data1 - data2;
				result += weight * fabs(term);
				tweight += weight;

		}
	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, rows1 * i + index1);
			data2 = tex1Dfetch<float>(texData, rows2 * i + index2);
			weight = tex1Dfetch<float>(texWt, i);

			double term = data1 - data2;
			result += weight * fabs(term);
			tweight += weight;

		}
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	return result;
}


/**
 * Override method that works using a combination of texture and shared memory data
 */
__device__ static
float cityblock(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* _data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
Purpose
=======

The cityblock routine calculates the weighted "City Block" distance between
two rows or columns in a matrix. City Block distance is defined as the
absolute value of X1-X2 plus the absolute value of Y1-Y2 plus..., which is
equivalent to taking an "up and over" path.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================ */
{
	double result = 0.;
	double tweight = 0;
	int i;
	float data1 = 0;
	float data2 = 0;
	float weight = 0;

	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, index1 * cols + i);
			data2 = _data2[ index2 * cols + i];
			weight = tex1Dfetch<float>(texWt, i);


				double term = data1 - data2;
				result += weight * fabs(term);
				tweight += weight;

		}
	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, rows1 * i + index1);
			data2 = _data2[ rows2 * i + index2];
			weight = tex1Dfetch<float>(texWt, i);


				double term = data1 - data2;
				result += weight * fabs(term);
				tweight += weight;

		}
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	return result;
}

/* ********************************************************************* */

//__device__ static
//float correlation(int cols, int rows1, int rows2, float* data1, float* data2, int* mask1, int* mask2,
//const float weight[], int index1, int index2, int transpose)

__device__ static
float correlation(int cols, int rows1, int rows2, float* data1, float* data2,
cudaTextureObject_t texWt, int index1, int index2, int transpose)
/*
Purpose
=======

The correlation routine calculates the weighted Pearson distance between two
rows or columns in a matrix. We define the Pearson distance as one minus the
Pearson correlation.
This definition yields a semi-metric: d(a,b) >= 0, and d(a,b) = 0 iff a = b.
but the triangular inequality d(a,b) + d(b,c) >= d(a,c) does not hold
(e.g., choose b = a + c).

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.
============================================================================
*/
{
	double result = 0.;
	double sum1 = 0.;
	double sum2 = 0.;
	double denom1 = 0.;
	double denom2 = 0.;
	double tweight = 0.;
	if (transpose == 0) /* Calculate the distance between two rows */
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term1 = data1[index1 * cols + i];
				double term2 = data2[index2 * cols + i];
				double w = tex1Dfetch<float>(texWt, i);
				sum1 += w*term1;
				sum2 += w*term2;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				tweight += w;

		}
	}
	else
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term1 = data1[rows1 * i + index1];
				double term2 = data2[rows2 * i + index2];
				double w = tex1Dfetch<float>(texWt, i);
				sum1 += w*term1;
				sum2 += w*term2;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				tweight += w;

		}
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result -= sum1 * sum2 / tweight;
	denom1 -= sum1 * sum1 / tweight;
	denom2 -= sum2 * sum2 / tweight;
	if (denom1 <= 0) return 1; /* include '<' to deal with roundoff errors */
	if (denom2 <= 0) return 1; /* include '<' to deal with roundoff errors */
	result = result / sqrt(denom1*denom2);
	result = 1. - result;
	return result;
}

__device__ static
float correlation(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
Purpose
=======

The correlation routine calculates the weighted Pearson distance between two
rows or columns in a matrix. We define the Pearson distance as one minus the
Pearson correlation.
This definition yields a semi-metric: d(a,b) >= 0, and d(a,b) = 0 iff a = b.
but the triangular inequality d(a,b) + d(b,c) >= d(a,c) does not hold
(e.g., choose b = a + c).

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.
============================================================================
*/
{
	double result = 0.;
	double sum1 = 0.;
	double sum2 = 0.;
	double denom1 = 0.;
	double denom2 = 0.;
	double tweight = 0.;
	int mask1 =0;
	int mask2 =0;
	float data1 = 0;
	float data2 = 0;
	float weight = 0;

	if (transpose == 0) /* Calculate the distance between two rows */
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, index1 * cols + i);
			data2 = tex1Dfetch<float>(texData, index2 * cols + i);
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double w = weight;
				sum1 += w*term1;
				sum2 += w*term2;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				tweight += w;

		}
	}
	else
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, rows1 * i + index1);
			data2 = tex1Dfetch<float>(texData, rows2 * i + index2);
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double w = weight;
				sum1 += w*term1;
				sum2 += w*term2;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				tweight += w;

		}
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result -= sum1 * sum2 / tweight;
	denom1 -= sum1 * sum1 / tweight;
	denom2 -= sum2 * sum2 / tweight;
	if (denom1 <= 0) return 1; /* include '<' to deal with roundoff errors */
	if (denom2 <= 0) return 1; /* include '<' to deal with roundoff errors */
	result = result / sqrt(denom1*denom2);
	result = 1. - result;
	return result;
}

/**
 * Override method that works using a combination of texture and shared memory data
 */
__device__ static
float correlation(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* _data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
Purpose
=======

The correlation routine calculates the weighted Pearson distance between two
rows or columns in a matrix. We define the Pearson distance as one minus the
Pearson correlation.
This definition yields a semi-metric: d(a,b) >= 0, and d(a,b) = 0 iff a = b.
but the triangular inequality d(a,b) + d(b,c) >= d(a,c) does not hold
(e.g., choose b = a + c).

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.
============================================================================
*/
{
	double result = 0.;
	double sum1 = 0.;
	double sum2 = 0.;
	double denom1 = 0.;
	double denom2 = 0.;
	double tweight = 0.;

	float data1 = 0;
	float data2 = 0;
	float weight = 0;

	if (transpose == 0) /* Calculate the distance between two rows */
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, index1 * cols + i);
			data2 = _data2[ index2 * cols + i];
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double w = weight;
				sum1 += w*term1;
				sum2 += w*term2;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				tweight += w;

		}
	}
	else
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, rows1 * i + index1);
			data2 = _data2 [rows2 * i + index2];
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double w = weight;
				sum1 += w*term1;
				sum2 += w*term2;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				tweight += w;

		}
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result -= sum1 * sum2 / tweight;
	denom1 -= sum1 * sum1 / tweight;
	denom2 -= sum2 * sum2 / tweight;
	if (denom1 <= 0) return 1; /* include '<' to deal with roundoff errors */
	if (denom2 <= 0) return 1; /* include '<' to deal with roundoff errors */
	result = result / sqrt(denom1*denom2);
	result = 1. - result;
	return result;
}


/* ********************************************************************* */

//__device__ static
//float acorrelation(int cols, int rows1, int rows2, float* data1, float* data2, int* mask1, int* mask2,
//const float weight[], int index1, int index2, int transpose)

__device__ static
float acorrelation(int cols, int rows1, int rows2, float* data1, float* data2,
cudaTextureObject_t texWt, int index1, int index2, int transpose)
/*
Purpose
=======

The acorrelation routine calculates the weighted Pearson distance between two
rows or columns, using the absolute value of the correlation.
This definition yields a semi-metric: d(a,b) >= 0, and d(a,b) = 0 iff a = b.
but the triangular inequality d(a,b) + d(b,c) >= d(a,c) does not hold
(e.g., choose b = a + c).

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.
============================================================================
*/
{
	double result = 0.;
	double sum1 = 0.;
	double sum2 = 0.;
	double denom1 = 0.;
	double denom2 = 0.;
	double tweight = 0.;
	if (transpose == 0) /* Calculate the distance between two rows */
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term1 = data1[index1 * cols + i];
				double term2 = data2[index2 * cols + i];
				double w = tex1Dfetch<float>(texWt, i);
				sum1 += w*term1;
				sum2 += w*term2;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				tweight += w;

		}
	}
	else
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term1 = data1[rows1 * i + index1];
				double term2 = data2[rows2 * i + index2];
				double w = tex1Dfetch<float>(texWt, i);
				sum1 += w*term1;
				sum2 += w*term2;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				tweight += w;

		}
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result -= sum1 * sum2 / tweight;
	denom1 -= sum1 * sum1 / tweight;
	denom2 -= sum2 * sum2 / tweight;
	if (denom1 <= 0) return 1; /* include '<' to deal with roundoff errors */
	if (denom2 <= 0) return 1; /* include '<' to deal with roundoff errors */
	result = fabs(result) / sqrt(denom1*denom2);
	result = 1. - result;
	return result;
}


__device__ static
float acorrelation(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
Purpose
=======

The acorrelation routine calculates the weighted Pearson distance between two
rows or columns, using the absolute value of the correlation.
This definition yields a semi-metric: d(a,b) >= 0, and d(a,b) = 0 iff a = b.
but the triangular inequality d(a,b) + d(b,c) >= d(a,c) does not hold
(e.g., choose b = a + c).

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.
============================================================================
*/
{
	double result = 0.;
	double sum1 = 0.;
	double sum2 = 0.;
	double denom1 = 0.;
	double denom2 = 0.;
	double tweight = 0.;
	float data1 = 0;
	float data2 = 0;
	float weight = 0;


	if (transpose == 0) /* Calculate the distance between two rows */
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, index1 * cols + i);
			data2 = tex1Dfetch<float>(texData, index2 * cols + i);
			weight = tex1Dfetch<float>(texWt, i);



				double term1 = data1;
				double term2 = data2;
				double w = weight;
				sum1 += w*term1;
				sum2 += w*term2;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				tweight += w;

		}
	}
	else
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, rows1 * i + index1);
			data2 = tex1Dfetch<float>(texData, rows2 * i + index2);
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double w = weight;
				sum1 += w*term1;
				sum2 += w*term2;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				tweight += w;

		}
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result -= sum1 * sum2 / tweight;
	denom1 -= sum1 * sum1 / tweight;
	denom2 -= sum2 * sum2 / tweight;
	if (denom1 <= 0) return 1; /* include '<' to deal with roundoff errors */
	if (denom2 <= 0) return 1; /* include '<' to deal with roundoff errors */
	result = fabs(result) / sqrt(denom1*denom2);
	result = 1. - result;
	return result;
}

/**
 * Override method that works using a combination of texture and shared memory data
 */
__device__ static
float acorrelation(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* _data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
Purpose
=======

The acorrelation routine calculates the weighted Pearson distance between two
rows or columns, using the absolute value of the correlation.
This definition yields a semi-metric: d(a,b) >= 0, and d(a,b) = 0 iff a = b.
but the triangular inequality d(a,b) + d(b,c) >= d(a,c) does not hold
(e.g., choose b = a + c).

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.
============================================================================
*/
{
	double result = 0.;
	double sum1 = 0.;
	double sum2 = 0.;
	double denom1 = 0.;
	double denom2 = 0.;
	double tweight = 0.;

	float data1 = 0;
	float data2 = 0;
	float weight = 0;


	if (transpose == 0) /* Calculate the distance between two rows */
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, index1 * cols + i);
			data2 = _data2[ index2 * cols + i];
			weight = tex1Dfetch<float>(texWt, i);

				double term1 = data1;
				double term2 = data2;
				double w = weight;
				sum1 += w*term1;
				sum2 += w*term2;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				tweight += w;

		}
	}
	else
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, rows1 * i + index1);
			data2 = _data2[ rows2 * i + index2];
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double w = weight;
				sum1 += w*term1;
				sum2 += w*term2;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				tweight += w;

		}
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result -= sum1 * sum2 / tweight;
	denom1 -= sum1 * sum1 / tweight;
	denom2 -= sum2 * sum2 / tweight;
	if (denom1 <= 0) return 1; /* include '<' to deal with roundoff errors */
	if (denom2 <= 0) return 1; /* include '<' to deal with roundoff errors */
	result = fabs(result) / sqrt(denom1*denom2);
	result = 1. - result;
	return result;
}


/* ********************************************************************* */

//__device__ static
//float ucorrelation(int cols, int rows1, int rows2, float* data1, float* data2, int* mask1, int* mask2,
//const float weight[], int index1, int index2, int transpose)

__device__ static
float ucorrelation(int cols, int rows1, int rows2, float* data1, float* data2,
cudaTextureObject_t texWt, int index1, int index2, int transpose)
/*
Purpose
=======

The ucorrelation routine calculates the weighted Pearson distance between two
rows or columns, using the uncentered version of the Pearson correlation. In the
uncentered Pearson correlation, a zero mean is used for both vectors even if
the actual mean is nonzero.
This definition yields a semi-metric: d(a,b) >= 0, and d(a,b) = 0 iff a = b.
but the triangular inequality d(a,b) + d(b,c) >= d(a,c) does not hold
(e.g., choose b = a + c).

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.
============================================================================
*/
{
	double result = 0.;
	double denom1 = 0.;
	double denom2 = 0.;
	int flag = 0;
	/* flag will remain zero if no nonzero combinations of mask1 and mask2 are
	* found.
	*/
	if (transpose == 0) /* Calculate the distance between two rows */
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term1 = data1[index1 * cols + i];
				double term2 = data2[index2 * cols + i];
				double w = tex1Dfetch<float>(texWt, i);
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				flag = 1;

		}
	}
	else
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{
				double term1 = data1[rows1 * i + index1];
				double term2 = data2[rows2 * i + index2];
				double w = tex1Dfetch<float>(texWt, i);
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				flag = 1;

		}
	}
	if (!flag) return 0.;
	if (denom1 == 0.) return 1.;
	if (denom2 == 0.) return 1.;
	result = result / sqrt(denom1*denom2);
	result = 1. - result;
	return result;
}

__device__ static
float ucorrelation(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)


/*
Purpose
=======

The ucorrelation routine calculates the weighted Pearson distance between two
rows or columns, using the uncentered version of the Pearson correlation. In the
uncentered Pearson correlation, a zero mean is used for both vectors even if
the actual mean is nonzero.
This definition yields a semi-metric: d(a,b) >= 0, and d(a,b) = 0 iff a = b.
but the triangular inequality d(a,b) + d(b,c) >= d(a,c) does not hold
(e.g., choose b = a + c).

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.
============================================================================
*/
{
	double result = 0.;
	double denom1 = 0.;
	double denom2 = 0.;
	int flag = 0;

	float data1 = 0;
	float data2 = 0;
	float weight = 0;
	/* flag will remain zero if no nonzero combinations of mask1 and mask2 are
	* found.
	*/
	if (transpose == 0) /* Calculate the distance between two rows */
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, index1 * cols + i);
			data2 = tex1Dfetch<float>(texData, index2 * cols + i);
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double w = weight;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				flag = 1;

		}
	}
	else
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, rows1 * i + index1);
			data2 = tex1Dfetch<float>(texData, rows2 * i + index2);
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double w = weight;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				flag = 1;

		}
	}
	if (!flag) return 0.;
	if (denom1 == 0.) return 1.;
	if (denom2 == 0.) return 1.;
	result = result / sqrt(denom1*denom2);
	result = 1. - result;
	return result;
}

/**
 * Override method that works using a combination of texture and shared memory data
 */
__device__ static
float ucorrelation(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* _data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)
/*
Purpose
=======

The ucorrelation routine calculates the weighted Pearson distance between two
rows or columns, using the uncentered version of the Pearson correlation. In the
uncentered Pearson correlation, a zero mean is used for both vectors even if
the actual mean is nonzero.
This definition yields a semi-metric: d(a,b) >= 0, and d(a,b) = 0 iff a = b.
but the triangular inequality d(a,b) + d(b,c) >= d(a,c) does not hold
(e.g., choose b = a + c).

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.
============================================================================
*/
{
	double result = 0.;
	double denom1 = 0.;
	double denom2 = 0.;
	int flag = 0;

	float data1 = 0;
	float data2 = 0;
	float weight = 0;
	/* flag will remain zero if no nonzero combinations of mask1 and mask2 are
	* found.
	*/
	if (transpose == 0) /* Calculate the distance between two rows */
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, index1 * cols + i);
			data2 = _data2[ index2 * cols + i];
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double w = weight;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				flag = 1;

		}
	}
	else
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, rows1 * i + index1);
			data2 = _data2[ rows2 * i + index2];
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double w = weight;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				flag = 1;

		}
	}
	if (!flag) return 0.;
	if (denom1 == 0.) return 1.;
	if (denom2 == 0.) return 1.;
	result = result / sqrt(denom1*denom2);
	result = 1. - result;
	return result;
}

/* ********************************************************************* */

//__device__ static
//float uacorrelation(int cols, int rows1, int rows2, float* data1, float* data2, int* mask1, int* mask2,
//const float weight[], int index1, int index2, int transpose)

__device__ static
float uacorrelation(int cols, int rows1, int rows2, float* data1, float* data2,
cudaTextureObject_t texWt, int index1, int index2, int transpose)
/*
Purpose
=======

The uacorrelation routine calculates the weighted Pearson distance between two
rows or columns, using the absolute value of the uncentered version of the
Pearson correlation. In the uncentered Pearson correlation, a zero mean is used
for both vectors even if the actual mean is nonzero.
This definition yields a semi-metric: d(a,b) >= 0, and d(a,b) = 0 iff a = b.
but the triangular inequality d(a,b) + d(b,c) >= d(a,c) does not hold
(e.g., choose b = a + c).

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.
============================================================================
*/
{
	double result = 0.;
	double denom1 = 0.;
	double denom2 = 0.;
	int flag = 0;
	/* flag will remain zero if no nonzero combinations of mask1 and mask2 are
	* found.
	*/
	if (transpose == 0) /* Calculate the distance between two rows */
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term1 = data1[index1 * cols + i];
				double term2 = data2[index2 * cols + i];
				double w = tex1Dfetch<float>(texWt, i);
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				flag = 1;

		}
	}
	else
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term1 = data1[rows1 * i + index1];
				double term2 = data2[rows2 * i + index2];
				double w = tex1Dfetch<float>(texWt, i);
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				flag = 1;

		}
	}
	if (!flag) return 0.;
	if (denom1 == 0.) return 1.;
	if (denom2 == 0.) return 1.;
	result = fabs(result) / sqrt(denom1*denom2);
	result = 1. - result;
	return result;
}


__device__ static
float uacorrelation(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)
/*
Purpose
=======

The uacorrelation routine calculates the weighted Pearson distance between two
rows or columns, using the absolute value of the uncentered version of the
Pearson correlation. In the uncentered Pearson correlation, a zero mean is used
for both vectors even if the actual mean is nonzero.
This definition yields a semi-metric: d(a,b) >= 0, and d(a,b) = 0 iff a = b.
but the triangular inequality d(a,b) + d(b,c) >= d(a,c) does not hold
(e.g., choose b = a + c).

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.
============================================================================
*/
{
	double result = 0.;
	double denom1 = 0.;
	double denom2 = 0.;
	int flag = 0;
	float data1 = 0;
	float data2 = 0;
	float weight = 0;
	/* flag will remain zero if no nonzero combinations of mask1 and mask2 are
	* found.
	*/
	if (transpose == 0) /* Calculate the distance between two rows */
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, index1 * cols + i);
			data2 = tex1Dfetch<float>(texData, index2 * cols + i);
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double w = weight;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				flag = 1;

		}
	}
	else
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, rows1 * i + index1);
			data2 = tex1Dfetch<float>(texData, rows2 * i + index2);
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double w = weight;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				flag = 1;

		}
	}
	if (!flag) return 0.;
	if (denom1 == 0.) return 1.;
	if (denom2 == 0.) return 1.;
	result = fabs(result) / sqrt(denom1*denom2);
	result = 1. - result;
	return result;
}

/**
 * Override method that works using a combination of texture and shared memory data
 */
__device__ static
float uacorrelation(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* _data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)
/*
Purpose
=======

The uacorrelation routine calculates the weighted Pearson distance between two
rows or columns, using the absolute value of the uncentered version of the
Pearson correlation. In the uncentered Pearson correlation, a zero mean is used
for both vectors even if the actual mean is nonzero.
This definition yields a semi-metric: d(a,b) >= 0, and d(a,b) = 0 iff a = b.
but the triangular inequality d(a,b) + d(b,c) >= d(a,c) does not hold
(e.g., choose b = a + c).

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.
============================================================================
*/
{
	double result = 0.;
	double denom1 = 0.;
	double denom2 = 0.;
	int flag = 0;
	int mask1 =0;
	int mask2 =0;
	float data1 = 0;
	float data2 = 0;
	float weight = 0;
	/* flag will remain zero if no nonzero combinations of mask1 and mask2 are
	* found.
	*/
	if (transpose == 0) /* Calculate the distance between two rows */
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, index1 * cols + i);
			data2 = _data2[ index2 * cols + i];
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double w = weight;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				flag = 1;

		}
	}
	else
	{
		int i;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, rows1 * i + index1);
			data2 = _data2[ rows2 * i + index2];
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double w = weight;
				result += w*term1*term2;
				denom1 += w*term1*term1;
				denom2 += w*term2*term2;
				flag = 1;

		}
	}
	if (!flag) return 0.;
	if (denom1 == 0.) return 1.;
	if (denom2 == 0.) return 1.;
	result = fabs(result) / sqrt(denom1*denom2);
	result = 1. - result;
	return result;
}

/* *********************************************************************  */

//__device__ static
//float kendall(int cols, int rows1, int rows2, float* data1, float* data2, int* mask1, int* mask2,
//const float weight[], int index1, int index2, int transpose)

__device__ static
float kendall(int cols, int rows1, int rows2, float* data1, float* data2,
cudaTextureObject_t texWt, int index1, int index2, int transpose)
/*
Purpose
=======

The kendall routine calculates the Kendall distance between two
rows or columns. The Kendall distance is defined as one minus Kendall's tau.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
These weights are ignored, but included for consistency with other distance
measures.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.
============================================================================
*/
{
	int con = 0;
	int dis = 0;
	int exx = 0;
	int exy = 0;
	int flag = 0;
	/* flag will remain zero if no nonzero combinations of mask1 and mask2 are
	* found.
	*/
	double denomx;
	double denomy;
	double tau;
	int i, j;
	if (transpose == 0)
	{

		for (i = 0; i < cols; i++)
		{

				for (j = 0; j < i; j++)
				{

						double x1 = data1[index1 * cols + i];
						double x2 = data1[index1 * cols + j];
						double y1 = data2[index2 * cols + i];
						double y2 = data2[index2 * cols + j];
						if (x1 < x2 && y1 < y2) con++;
						if (x1 > x2 && y1 > y2) con++;
						if (x1 < x2 && y1 > y2) dis++;
						if (x1 > x2 && y1 < y2) dis++;
						if (x1 == x2 && y1 != y2) exx++;
						if (x1 != x2 && y1 == y2) exy++;
						flag = 1;

				}

		}
	}
	else
	{
		for (i = 0; i < cols; i++)
		{

				for (j = 0; j < i; j++)
				{

						double x1 = data1[rows1 * i + index1];
						double x2 = data1[rows1 * j + index1];
						double y1 = data2[rows2 * i + index2];
						double y2 = data2[rows2 * j + index2];
						if (x1 < x2 && y1 < y2) con++;
						if (x1 > x2 && y1 > y2) con++;
						if (x1 < x2 && y1 > y2) dis++;
						if (x1 > x2 && y1 < y2) dis++;
						if (x1 == x2 && y1 != y2) exx++;
						if (x1 != x2 && y1 == y2) exy++;
						flag = 1;

				}

		}
	}
	if (!flag) return 0.;
	denomx = con + dis + exx;
	denomy = con + dis + exy;
	if (denomx == 0) return 1;
	if (denomy == 0) return 1;
	tau = (con - dis) / sqrt(denomx*denomy);
	return 1. - tau;
}

__device__ static
float kendall(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)
/*
Purpose
=======

The kendall routine calculates the Kendall distance between two
rows or columns. The Kendall distance is defined as one minus Kendall's tau.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
These weights are ignored, but included for consistency with other distance
measures.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.
============================================================================
*/
{
	int con = 0;
	int dis = 0;
	int exx = 0;
	int exy = 0;
	int flag = 0;
	float data1 = 0;
	float data2 = 0;

	/* flag will remain zero if no nonzero combinations of mask1 and mask2 are
	* found.
	*/
	double denomx;
	double denomy;
	double tau;
	int i, j;
	if (transpose == 0)
	{
		for (i = 0; i < cols; i++)
		{
				for (j = 0; j < i; j++)
				{

					float x1 = tex1Dfetch<float>(texData, index1 * cols + i);
					float x2 = tex1Dfetch<float>(texData, index1 * cols + j);
					float y1 = tex1Dfetch<float>(texData, index2 * cols + i);
					float y2 = tex1Dfetch<float>(texData, index2 * cols + j);



						if (x1 < x2 && y1 < y2) con++;
						if (x1 > x2 && y1 > y2) con++;
						if (x1 < x2 && y1 > y2) dis++;
						if (x1 > x2 && y1 < y2) dis++;
						if (x1 == x2 && y1 != y2) exx++;
						if (x1 != x2 && y1 == y2) exy++;
						flag = 1;

				}

		}
	}
	else
	{
		for (i = 0; i < cols; i++)
		{

				for (j = 0; j < i; j++)
				{

					float x1 = tex1Dfetch<float>(texData, rows1 * i + index1);
					float x2 = tex1Dfetch<float>(texData, rows1 * j + index1);
					float y1 = tex1Dfetch<float>(texData, rows2 * i + index2);
					float y2 = tex1Dfetch<float>(texData, rows2 * j + index2);


						if (x1 < x2 && y1 < y2) con++;
						if (x1 > x2 && y1 > y2) con++;
						if (x1 < x2 && y1 > y2) dis++;
						if (x1 > x2 && y1 < y2) dis++;
						if (x1 == x2 && y1 != y2) exx++;
						if (x1 != x2 && y1 == y2) exy++;
						flag = 1;

				}

		}
	}

	if (!flag) return 0.;
	denomx = con + dis + exx;
	denomy = con + dis + exy;
	if (denomx == 0) return 1;
	if (denomy == 0) return 1;
	tau = (con - dis) / sqrt(denomx*denomy);
	return 1. - tau;
}

/**
 * Override method that works using a combination of texture and shared memory data
 */
__device__ static
float kendall(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* _data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)
/*
Purpose
=======

The kendall routine calculates the Kendall distance between two
rows or columns. The Kendall distance is defined as one minus Kendall's tau.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
These weights are ignored, but included for consistency with other distance
measures.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.
============================================================================
*/
{
	int con = 0;
	int dis = 0;
	int exx = 0;
	int exy = 0;
	int flag = 0;
	float data1 = 0;
	float data2 = 0;

	/* flag will remain zero if no nonzero combinations of mask1 and mask2 are
	* found.
	*/
	double denomx;
	double denomy;
	double tau;
	int i, j;
	if (transpose == 0)
	{
		for (i = 0; i < cols; i++)
		{

				for (j = 0; j < i; j++)
				{

					float x1 = tex1Dfetch<float>(texData1, index1 * cols + i);
					float x2 = _data2[ index1 * cols + j];
					float y1 = tex1Dfetch<float>(texData1, index2 * cols + i);
					float y2 = _data2[ index2 * cols + j];



						if (x1 < x2 && y1 < y2) con++;
						if (x1 > x2 && y1 > y2) con++;
						if (x1 < x2 && y1 > y2) dis++;
						if (x1 > x2 && y1 < y2) dis++;
						if (x1 == x2 && y1 != y2) exx++;
						if (x1 != x2 && y1 == y2) exy++;
						flag = 1;

				}

		}
	}
	else
	{
		for (i = 0; i < cols; i++)
		{

				for (j = 0; j < i; j++)
				{

					float x1 = tex1Dfetch<float>(texData1, rows1 * i + index1);
					float x2 = _data2[ rows1 * j + index1];
					float y1 = tex1Dfetch<float>(texData1, rows2 * i + index2);
					float y2 = _data2[ rows2 * j + index2];


						if (x1 < x2 && y1 < y2) con++;
						if (x1 > x2 && y1 > y2) con++;
						if (x1 < x2 && y1 > y2) dis++;
						if (x1 > x2 && y1 < y2) dis++;
						if (x1 == x2 && y1 != y2) exx++;
						if (x1 != x2 && y1 == y2) exy++;
						flag = 1;

				}

		}
	}

	if (!flag) return 0.;
	denomx = con + dis + exx;
	denomy = con + dis + exy;
	if (denomx == 0) return 1;
	if (denomy == 0) return 1;
	tau = (con - dis) / sqrt(denomx*denomy);
	return 1. - tau;
}

/* *********************************************************************
*
* Added new distance measures to existing ones
* @author: Vinay B Gavirangaswamy
*
*  ********************************************************************* */

//__device__ static
//float cosineDist(int cols, int rows1, int rows2, float* data1, float* data2, int* mask1, int* mask2,
//const float weight[], int index1, int index2, int transpose)

__device__ static
float cosineDist(int cols, int rows1, int rows2, float* data1, float* data2,
cudaTextureObject_t texWt, int index1, int index2, int transpose)

/*
Purpose
=======

The  cosineDist routine calculates the weighted Cosine Distance distance between two
rows or columns in a matrix.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	double result = 0.;
	double temp_numerator = 0.;
	double temp_denominator1 = 0.;
	double temp_denominator2 = 0.;
	double tweight = 0;
	int i;

	//printf("\nIn cosine dist calculation()\n");
	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term1 = data1[index1 * cols + i];
				double term2 = data2[index2 * cols + i];
				tweight += tex1Dfetch<float>(texWt, i);

				temp_numerator += term1 * term2;
				temp_denominator1 += pow(term1, 2);
				temp_denominator2 += pow(term2, 2);

				//printf("\nterm1 = %f term2=%f temp_numerator=%f temp_denominator1=%f temp_denominator2=%f \n",term1, term2, temp_numerator, temp_denominator1, temp_denominator2 );

		}

		result = temp_numerator / (sqrt(temp_denominator1)*sqrt(temp_denominator2));
		//printf("\nresult = %f\n", result);
	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term1 = data1[rows1 * i + index1];
				double term2 = data2[rows2 * i + index2];
				tweight += tex1Dfetch<float>(texWt, i);

				temp_numerator += term1 * term2;
				temp_denominator1 += pow(term1, 2);
				temp_denominator2 += pow(term2, 2);


		}
		result = temp_numerator / (sqrt(temp_denominator1)*sqrt(temp_denominator2));
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	//result /= tweight;


	//printf("\nresult = %f\n", result);

	return result;
}

__device__ static
float cosineDist(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)
/*
Purpose
=======

The  cosineDist routine calculates the weighted Cosine Distance distance between two
rows or columns in a matrix.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	double result = 0.;
	double temp_numerator = 0.;
	double temp_denominator1 = 0.;
	double temp_denominator2 = 0.;
	double tweight = 0;
	int i;

	float data1 = 0;
	float data2 = 0;
	float weight = 0;
	//printf("\nIn cosine dist calculation()\n");
	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, index1 * cols + i);
			data2 = tex1Dfetch<float>(texData, index2 * cols + i);
			weight = tex1Dfetch<float>(texWt, i);


				tweight += weight;
				temp_numerator += data1 * data2;
				temp_denominator1 += pow(data1, 2);
				temp_denominator2 += pow(data2, 2);


		}

		result = temp_numerator / (sqrt(temp_denominator1)*sqrt(temp_denominator2));
	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, rows1 * i + index1);
			data2 = tex1Dfetch<float>(texData, rows2 * i + index2);
			weight = tex1Dfetch<float>(texWt, i);


				tweight += weight;
				temp_numerator += data1 * data2;
				temp_denominator1 += pow(data1, 2);
				temp_denominator2 += pow(data2, 2);


		}
		result = temp_numerator / (sqrt(temp_denominator1)*sqrt(temp_denominator2));
	}
	if (!tweight) return 0; /* usually due to empty clusters */

	return result;
}

/**
 * Override method that works using a combination of texture and shared memory data
 */
__device__ static
float cosineDist(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* _data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)
/*
Purpose
=======

The  cosineDist routine calculates the weighted Cosine Distance distance between two
rows or columns in a matrix.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	double result = 0.;
	double temp_numerator = 0.;
	double temp_denominator1 = 0.;
	double temp_denominator2 = 0.;
	double tweight = 0;
	int i;

	float data1 = 0;
	float data2 = 0;
	float weight = 0;
	//printf("\nIn cosine dist calculation()\n");
	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, index1 * cols + i);
			data2 = _data2[ index2 * cols + i];
			weight = tex1Dfetch<float>(texWt, i);

				tweight += weight;
				temp_numerator += data1 * data2;
				temp_denominator1 += pow(data1, 2);
				temp_denominator2 += pow(data2, 2);


		}

		result = temp_numerator / (sqrt(temp_denominator1)*sqrt(temp_denominator2));
	}
	else
	{
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, rows1 * i + index1);
			data2 = _data2[ rows2 * i + index2];
			weight = tex1Dfetch<float>(texWt, i);


				tweight += weight;
				temp_numerator += data1 * data2;
				temp_denominator1 += pow(data1, 2);
				temp_denominator2 += pow(data2, 2);


		}
		result = temp_numerator / (sqrt(temp_denominator1)*sqrt(temp_denominator2));
	}
	if (!tweight) return 0; /* usually due to empty clusters */

	return result;
}

/* ********************************************************************* */
//__device__ static
//float mahalanobisDist(int cols, int rows1, int rows2, float* data1, float* data2, int* mask1, int* mask2,
//const float weight[], int index1, int index2, int transpose)

__device__ static
float mahalanobisDist(int cols, int rows1, int rows2, float* data1, float* data2,
cudaTextureObject_t texWt, int index1, int index2, int transpose)


/*
Purpose
=======

The  mahalanobisDist routine calculates the weighted mahalanobis distance between two
rows or columns in a matrix.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	double result = 0.;
	double tweight = 0;
	int i;
	if (transpose == 0) /* Calculate the distance between two rows */
	{
		double mean1 = 0.;
		double mean2 = 0.;
#pragma unroll
		for (i = 0; i < cols; i++) {

				mean1 += data1[index1 * cols + i];
				mean2 += data2[index2 * cols + i];

		}
		mean1 /= cols;
		mean2 /= cols;
#pragma unroll
		for (i = 0; i < cols; i++) {


				double xdiff = 0., covar = 0.;
				covar = (data1[index1 * cols + i] - mean1)
					* (data2[index2 * cols + i] - mean2);
				xdiff = data1[index1 * cols + i] - data2[index2 * cols + i];
				result += tex1Dfetch<float>(texWt, i) * (xdiff * covar * xdiff);
				tweight += tex1Dfetch<float>(texWt, i);

		}

	}
	else
	{
		double mean1 = 0.;
		double mean2 = 0.;
#pragma unroll
		for (i = 0; i < cols; i++) {

			mean1 += data1[rows1 * i + index1];
			mean2 += data2[rows2 * i + index2];

		}
		mean1 /= cols;
		mean2 /= cols;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term1 = data1[rows1 * i + index1];
				double term2 = data2[rows2 * i + index2];
				double xdiff = 0., covar = 0.;
				covar = (term1 - mean1) * (term2 - mean2);
				xdiff = term1 - term2;
				result += tex1Dfetch<float>(texWt, i) * (xdiff * covar * xdiff);
				tweight += tex1Dfetch<float>(texWt, i);


		}

	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	return result;
}

__device__ static
float mahalanobisDist(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)
/*
Purpose
=======

The  mahalanobisDist routine calculates the weighted mahalanobis distance between two
rows or columns in a matrix.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	double result = 0.;
	double tweight = 0;
	int i;
	int mask1 =0;
	int mask2 =0;
	float data1 = 0;
	float data2 = 0;
	float weight = 0;
	if (transpose == 0) /* Calculate the distance between two rows */
	{
		double mean1 = 0.;
		double mean2 = 0.;
#pragma unroll
		for (i = 0; i < cols; i++) {

			data1 = tex1Dfetch<float>(texData, index1 * cols + i);
			data2 = tex1Dfetch<float>(texData, index2 * cols + i);
			weight = tex1Dfetch<float>(texWt, i);


				mean1 += data1;
				mean2 += data2;

		}
		mean1 /= cols;
		mean2 /= cols;

		for (i = 0; i < cols; i++) {

			data1 = tex1Dfetch<float>(texData, index1 * cols + i);
			data2 = tex1Dfetch<float>(texData, index2 * cols + i);
			weight = tex1Dfetch<float>(texWt, i);


				double xdiff = 0., covar = 0.;
				covar = (data1 - mean1)
					* (data2 - mean2);
				xdiff = data1 - data2;
				result += weight * (xdiff * covar * xdiff);
				tweight += weight;

		}

	}
	else
	{
		double mean1 = 0.;
		double mean2 = 0.;

		for (i = 0; i < cols; i++) {


			data1 = tex1Dfetch<float>(texData, rows1 * i + index1);
			data2 = tex1Dfetch<float>(texData, rows2 * i + index2);
			weight = tex1Dfetch<float>(texWt, i);


				mean1 += data1;
				mean2 += data2;

		}
		mean1 /= cols;
		mean2 /= cols;

		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, rows1 * i + index1);
			data2 = tex1Dfetch<float>(texData, rows2 * i + index2);
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double xdiff = 0., covar = 0.;
				covar = (term1 - mean1) * (term2 - mean2);
				xdiff = term1 - term2;
				result += weight * (xdiff * covar * xdiff);
				tweight += weight;


		}

	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	return result;
}

/**
 * Override method that works using a combination of texture and shared memory data
 */
__device__ static
float mahalanobisDist(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* _data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)
/*
Purpose
=======

The  mahalanobisDist routine calculates the weighted mahalanobis distance between two
rows or columns in a matrix.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	double result = 0.;
	double tweight = 0;
	int i;
	float data1 = 0;
	float data2 = 0;
	float weight = 0;
	if (transpose == 0) /* Calculate the distance between two rows */
	{
		double mean1 = 0.;
		double mean2 = 0.;
#pragma unroll
		for (i = 0; i < cols; i++) {

			data1 = tex1Dfetch<float>(texData1, index1 * cols + i);
			data2 = _data2[ index2 * cols + i];
			weight = tex1Dfetch<float>(texWt, i);

			mean1 += data1;
			mean2 += data2;

		}
		mean1 /= cols;
		mean2 /= cols;
#pragma unroll
		for (i = 0; i < cols; i++) {

			data1 = tex1Dfetch<float>(texData1, index1 * cols + i);
			data2 = _data2[ index2 * cols + i];
			weight = tex1Dfetch<float>(texWt, i);

				double xdiff = 0., covar = 0.;
				covar = (data1 - mean1)
					* (data2 - mean2);
				xdiff = data1 - data2;
				result += weight * (xdiff * covar * xdiff);
				tweight += weight;

		}

	}
	else
	{
		double mean1 = 0.;
		double mean2 = 0.;
#pragma unroll
		for (i = 0; i < cols; i++) {


			data1 = tex1Dfetch<float>(texData1, rows1 * i + index1);
			data2 = _data2[ rows2 * i + index2];
			weight = tex1Dfetch<float>(texWt, i);

			mean1 += data1;
			mean2 += data2;

		}
		mean1 /= cols;
		mean2 /= cols;
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, rows1 * i + index1);
			data2 = _data2[ rows2 * i + index2];
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				double xdiff = 0., covar = 0.;
				covar = (term1 - mean1) * (term2 - mean2);
				xdiff = term1 - term2;
				result += weight * (xdiff * covar * xdiff);
				tweight += weight;


		}

	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	return result;
}

/* ********************************************************************* */

//__device__ static
//float jaccardDist(int cols, int rows1, int rows2, float* data1, float* data2, int* mask1, int* mask2,
//const float weight[], int index1, int index2, int transpose)

__device__ static
float jaccardDist(int cols, int rows1, int rows2, float* data1, float* data2,
cudaTextureObject_t texWt, int index1, int index2, int transpose)

/*
Purpose
=======

The  jaccardDist routine calculates the weighted jaccard distance between two
rows or columns in a matrix.

From Wikipedia: https://en.wikipedia.org/wiki/Jaccard_index

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	double result = 0.;
	double jaccard_similarity = 0.;

	double tweight = 0;
	int i;
	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++) {


				double term1 = data1[index1 * cols + i];
				double term2 = data2[index2 * cols + i];
				jaccard_similarity += tex1Dfetch<float>(texWt, i) * ((term1<term2 ? term1 : term2) / (term1>term2 ? term1 : term2));
				tweight += tex1Dfetch<float>(texWt, i);

		}

		result = 1 - jaccard_similarity;

	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term1 = data1[rows1 * i + index1];
				double term2 = data2[rows2 * i + index2];
				jaccard_similarity += tex1Dfetch<float>(texWt, i) * ((term1<term2 ? term1 : term2) / (term1>term2 ? term1 : term2));
				tweight += tex1Dfetch<float>(texWt, i);

		}
		result = 1 - jaccard_similarity;
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	return result;
}

__device__ static
float jaccardDist(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
Purpose
=======

The  jaccardDist routine calculates the weighted jaccard distance between two
rows or columns in a matrix.

From Wikipedia: https://en.wikipedia.org/wiki/Jaccard_index

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	double result = 0.;
	double jaccard_similarity = 0.;
	int mask1 =0;
	int mask2 =0;
	float data1 = 0;
	float data2 = 0;
	float weight = 0;

	double tweight = 0;
	int i;
	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++) {

			data1 = tex1Dfetch<float>(texData, index1 * cols + i);
			data2 = tex1Dfetch<float>(texData, index2 * cols + i);
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				jaccard_similarity += weight * ((term1<term2 ? term1 : term2) / (term1>term2 ? term1 : term2));
				tweight += weight;

		}

		result = 1 - jaccard_similarity;

	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, rows1 * i + index1);
			data2 = tex1Dfetch<float>(texData, rows2 * i + index2);
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				jaccard_similarity += weight * ((term1<term2 ? term1 : term2) / (term1>term2 ? term1 : term2));
				tweight += weight;

		}
		result = 1 - jaccard_similarity;
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	return result;
}

/**
 * Override method that works using a combination of texture and shared memory data
 */
__device__ static
float jaccardDist(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* _data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)
/*
Purpose
=======

The  jaccardDist routine calculates the weighted jaccard distance between two
rows or columns in a matrix.

From Wikipedia: https://en.wikipedia.org/wiki/Jaccard_index

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	double result = 0.;
	double jaccard_similarity = 0.;
	float data1 = 0;
	float data2 = 0;
	float weight = 0;

	double tweight = 0;
	int i;
	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++) {

			data1 = tex1Dfetch<float>(texData1, index1 * cols + i);
			data2 = _data2[ index2 * cols + i];
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				jaccard_similarity += weight * ((term1<term2 ? term1 : term2) / (term1>term2 ? term1 : term2));
				tweight += weight;

		}

		result = 1 - jaccard_similarity;

	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, rows1 * i + index1);
			data2 = _data2[ rows2 * i + index2];
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				jaccard_similarity += weight * ((term1<term2 ? term1 : term2) / (term1>term2 ? term1 : term2));
				tweight += weight;

		}
		result = 1 - jaccard_similarity;
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	return result;
}

/* ********************************************************************* */
//__device__ static
//float chebyshevDist(int cols, int rows1, int rows2, float* data1, float* data2, int* mask1, int* mask2,
//const float weight[], int index1, int index2, int transpose)

__device__ static
float chebyshevDist(int cols, int rows1, int rows2, float* data1, float* data2,
cudaTextureObject_t texWt, int index1, int index2, int transpose)

/*
Purpose
=======

The  chebyshevDist routine calculates the weighted Chebyshev distance between two
rows or columns in a matrix.

From Wikipedia: https://en.wikipedia.org/wiki/Chebyshev_distance

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	double result = 0.;
	double *chevDist;
	double tweight = 0;
	int i;


	cudaMalloc(&chevDist, cols*sizeof(double));


	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++) {

				double term1 = data1[index1 * cols + i];
				double term2 = data2[index2 * cols + i];
				chevDist[i] = tex1Dfetch<float>(texWt, i) * (term1>term2 ? term1 : term2);
				tweight += tex1Dfetch<float>(texWt, i);

		}
#pragma unroll
		for (i = 0; i < cols; i++) {

			result = result>chevDist[i] ? result : chevDist[i];

		}

	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term1 = data1[rows1 * i + index1];
				double term2 = data2[rows2 * i + index2];
				chevDist[i] = tex1Dfetch<float>(texWt, i) * (term1>term2 ? term1 : term2);
				tweight += tex1Dfetch<float>(texWt, i);

		}

#pragma unroll
		for (i = 0; i < cols; i++) {

				result = result>chevDist[i] ? result : chevDist[i];

		}
	}

	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	return result;
}

__device__ static
float chebyshevDist(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
Purpose
=======

The  chebyshevDist routine calculates the weighted Chebyshev distance between two
rows or columns in a matrix.

From Wikipedia: https://en.wikipedia.org/wiki/Chebyshev_distance

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	double result = 0.;
	double *chevDist;
	double tweight = 0;
	int i;

	float data1 = 0;
	float data2 = 0;
	float weight = 0;

//	cudaMalloc(&chevDist, cols*sizeof(double));
	chevDist = (double*)malloc(cols * sizeof(double));

	if(!chevDist)
		return 0;


	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++) {

			data1 = tex1Dfetch<float>(texData, index1 * cols + i);
			data2 = tex1Dfetch<float>(texData, index2 * cols + i);
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				chevDist[i] = weight * (term1>term2 ? term1 : term2);
				tweight += weight;

		}
#pragma unroll
		for (i = 0; i < cols; i++) {

				result = result>chevDist[i] ? result : chevDist[i];

		}

	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, rows1 * i + index1);
			data2 = tex1Dfetch<float>(texData, rows2 * i + index2);
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				chevDist[i] = weight * (term1>term2 ? term1 : term2);
				tweight += weight;

		}
#pragma unroll
		for (i = 0; i < cols; i++) {

				result = result>chevDist[i] ? result : chevDist[i];

		}
	}

	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	free(chevDist);
	return result;
}

/**
 * Override method that works using a combination of texture and shared memory data
 */
__device__ static
float chebyshevDist(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* _data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)
/*
Purpose
=======

The  chebyshevDist routine calculates the weighted Chebyshev distance between two
rows or columns in a matrix.

From Wikipedia: https://en.wikipedia.org/wiki/Chebyshev_distance

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	double result = 0.;
	double *chevDist;
	double tweight = 0;
	int i;

	float data1 = 0;
	float data2 = 0;
	float weight = 0;

//	cudaMalloc(&chevDist, cols*sizeof(double));
	chevDist = (double*)malloc(cols * sizeof(double));

	if(!chevDist)
		return 0;


	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++) {

			data1 = tex1Dfetch<float>(texData1, index1 * cols + i);
			data2 = _data2[ index2 * cols + i];
			weight = tex1Dfetch<float>(texWt, i);


				double term1 = data1;
				double term2 = data2;
				chevDist[i] = weight * (term1>term2 ? term1 : term2);
				tweight += weight;

		}
#pragma unroll
		for (i = 0; i < cols; i++) {

			result = result>chevDist[i] ? result : chevDist[i];

		}

	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, rows1 * i + index1);
			data2 = _data2[ rows2 * i + index2];
			weight = tex1Dfetch<float>(texWt, i);

			double term1 = data1;
			double term2 = data2;
			chevDist[i] = weight * (term1>term2 ? term1 : term2);
			tweight += weight;

		}
#pragma unroll
		for (i = 0; i < cols; i++) {

				result = result>chevDist[i] ? result : chevDist[i];

		}
	}

	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	free(chevDist);
	return result;
}

/* ********************************************************************* */

__device__ static int hamming_distance(unsigned x, unsigned y)
{
	int dist = 0;
	unsigned val = x ^ y;

	// Count the number of bits set
	while (val != 0)
	{
		// A bit is set, so increment the count and clear the bit
		dist++;
		val &= val - 1;
	}

	// Return the number of differing bits
	return dist;
}

/* ----------------------------------------------------------------------- */

//__device__ static
//float hammingDist(int cols, int rows1, int rows2, float* data1, float* data2, int* mask1, int* mask2,
//const float weight[], int index1, int index2, int transpose)

__device__ static
float hammingDist(int cols, int rows1, int rows2, float* data1, float* data2,
cudaTextureObject_t texWt, int index1, int index2, int transpose)

/*
Purpose
=======

The hammingDist routine calculates the weighted Hamming distance between two
rows or columns in a matrix.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	double result = 0.;
	double tweight = 0;
	int i;
	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term = hamming_distance(data1[index1 * cols + i], data2[index2 * cols + i]);
				result += tex1Dfetch<float>(texWt, i) * term*term;
				tweight += tex1Dfetch<float>(texWt, i);

		}
	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

				double term = hamming_distance(data1[rows1 * i + index1], data2[rows2 * i + index2]);
				result += tex1Dfetch<float>(texWt, i) * term*term;
				tweight += tex1Dfetch<float>(texWt, i);

		}
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	return result;
}

__device__ static
float hammingDist(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
Purpose
=======

The hammingDist routine calculates the weighted Hamming distance between two
rows or columns in a matrix.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	double result = 0.;
	double tweight = 0;
	int i;

	float data1 = 0;
	float data2 = 0;
	float weight = 0;

	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, index1 * cols + i);
			data2 = tex1Dfetch<float>(texData, index2 * cols + i);
			weight = tex1Dfetch<float>(texWt, i);


				double term = hamming_distance(data1, data2);
				result += weight * term*term;
				tweight += weight;

		}
	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData, rows1 * i + index1);
			data2 = tex1Dfetch<float>(texData, rows2 * i + index2);
			weight = tex1Dfetch<float>(texWt, i);


			double term = hamming_distance(data1, data2);
			result += weight * term*term;
			tweight += weight;

		}
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	return result;
}


/**
 * Override method that works using a combination of texture and shared memory data
 */
__device__ static
float hammingDist(int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* _data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock)

/*
Purpose
=======

The hammingDist routine calculates the weighted Hamming distance between two
rows or columns in a matrix.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{
	double result = 0.;
	double tweight = 0;
	int i;

	float data1 = 0;
	float data2 = 0;
	float weight = 0;

	if (transpose == 0) /* Calculate the distance between two rows */
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, index1 * cols + i);
			data2 = _data2[ index2 * cols + i];
			weight = tex1Dfetch<float>(texWt, i);


				double term = hamming_distance(data1, data2);
				result += weight * term*term;
				tweight += weight;

		}
	}
	else
	{
#pragma unroll
		for (i = 0; i < cols; i++)
		{

			data1 = tex1Dfetch<float>(texData1, rows1 * i + index1);
			data2 = _data2[ rows2 * i + index2];
			weight = tex1Dfetch<float>(texWt, i);


				double term = hamming_distance(data1, data2);
				result += weight * term*term;
				tweight += weight;

		}
	}
	if (!tweight) return 0; /* usually due to empty clusters */
	result /= tweight;
	return result;
}


/* ********************************************************************* */


__device__ float calcDistMetricGPU(char dist, int cols, int rows1, int rows2, float* data1, float* data2,
		cudaTextureObject_t texWt, int index1, int index2, int transpose)
{
	switch (dist)
	{
	case 'e': return euclid(cols, rows1, rows2, data1, data2, texWt, index1, index2, transpose);
	case 'l': return seuclid(cols, rows1, rows2, data1, data2, texWt, index1, index2, transpose);
	case 'b': return cityblock(cols, rows1, rows2, data1, data2, texWt, index1, index2, transpose);
	case 'c': return correlation(cols, rows1, rows2, data1, data2, texWt, index1, index2, transpose);
	case 'a': return acorrelation(cols, rows1, rows2, data1, data2, texWt, index1, index2, transpose);
	case 'u': return ucorrelation(cols, rows1, rows2, data1, data2, texWt, index1, index2, transpose);
	case 'x': return uacorrelation(cols, rows1, rows2, data1, data2, texWt, index1, index2, transpose);
	case 's': return spearman(cols, rows1, rows2, data1, data2, texWt, index1, index2, transpose);
	case 'k': return kendall(cols, rows1, rows2, data1, data2, texWt, index1, index2, transpose);
	case 'o': return cosineDist(cols, rows1, rows2, data1, data2, texWt, index1, index2, transpose);
	case 'm': return mahalanobisDist(cols, rows1, rows2, data1, data2, texWt, index1, index2, transpose);
	case 'j': return jaccardDist(cols, rows1, rows2, data1, data2, texWt, index1, index2, transpose);
	case 'h': return chebyshevDist(cols, rows1, rows2, data1, data2, texWt, index1, index2, transpose);
	case 'g': return hammingDist(cols, rows1, rows2, data1, data2, texWt, index1, index2, transpose);
	default:  return euclid(cols, rows1, rows2, data1, data2, texWt, index1, index2, transpose);
	}
	//return 0.0; /* Never get here */
}

__device__ float calcDistMetricGPU(char dist, int cols, int rows1,
		int rows2, int index1, int index2, int transpose, cudaTextureObject_t texData, cudaTextureObject_t texWt,int *maxGridSize, int *maxThreadsPerBlock) {
	switch (dist) {
	case 'e': return euclid(cols, rows1, rows2, index1, index2, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);
	case 'l': return seuclid(cols, rows1, rows2, index1, index2, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);
	case 'b': return cityblock(cols, rows1, rows2, index1, index2, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);
	case 'c': return correlation(cols, rows1, rows2, index1, index2, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);
	case 'a': return acorrelation(cols, rows1, rows2, index1, index2, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);
	case 'u': return ucorrelation(cols, rows1, rows2, index1, index2, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);
	case 'x': return uacorrelation(cols, rows1, rows2, index1, index2, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);
	case 's': return spearman(cols, rows1, rows2, index1, index2, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);
	case 'k': return kendall(cols, rows1, rows2, index1, index2, transpose, texData,texWt, maxGridSize, maxThreadsPerBlock);
	case 'o': return cosineDist(cols, rows1, rows2, index1, index2, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);
	case 'm': return mahalanobisDist(cols, rows1, rows2, index1, index2, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);
	case 'j': return jaccardDist(cols, rows1, rows2, index1, index2, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);
	case 'h': return chebyshevDist(cols, rows1, rows2, index1, index2, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);
	case 'g': return hammingDist(cols, rows1, rows2, index1, index2, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);
	default:
		return euclid(cols, rows1, rows2, index1, index2, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);
	}
	//return 0.0; /* Never get here */
}

__device__ float calcDistMetricGPU(char dist, int cols, int rows1, int rows2, int index1, int index2,
		int transpose,cudaTextureObject_t texData1, float* data2, cudaTextureObject_t texWt, int *maxGridSize, int *maxThreadsPerBlock){
	switch (dist) {
	case 'e': return euclid(cols, rows1, rows2, index1, index2, transpose, texData1, data2, texWt, maxGridSize, maxThreadsPerBlock);
	case 'l': return seuclid(cols, rows1, rows2, index1, index2, transpose, texData1, data2, texWt, maxGridSize, maxThreadsPerBlock);
	case 'b': return cityblock(cols, rows1, rows2, index1, index2, transpose, texData1, data2, texWt, maxGridSize, maxThreadsPerBlock);
	case 'c': return correlation(cols, rows1, rows2, index1, index2, transpose, texData1, data2, texWt, maxGridSize, maxThreadsPerBlock);
	case 'a': return acorrelation(cols, rows1, rows2, index1, index2, transpose, texData1, data2, texWt, maxGridSize, maxThreadsPerBlock);
	case 'u': return ucorrelation(cols, rows1, rows2, index1, index2, transpose, texData1, data2, texWt, maxGridSize, maxThreadsPerBlock);
	case 'x': return uacorrelation(cols, rows1, rows2, index1, index2, transpose, texData1, data2, texWt, maxGridSize, maxThreadsPerBlock);
	case 's': return spearman(cols, rows1, rows2, index1, index2, transpose, texData1, data2, texWt, maxGridSize, maxThreadsPerBlock);
	case 'k': return kendall(cols, rows1, rows2, index1, index2, transpose, texData1, data2, texWt, maxGridSize, maxThreadsPerBlock);
	case 'o': return cosineDist(cols, rows1, rows2, index1, index2, transpose, texData1, data2, texWt, maxGridSize, maxThreadsPerBlock);
	case 'm': return mahalanobisDist(cols, rows1, rows2, index1, index2, transpose, texData1, data2, texWt, maxGridSize, maxThreadsPerBlock);
	case 'j': return jaccardDist(cols, rows1, rows2, index1, index2, transpose, texData1, data2, texWt, maxGridSize, maxThreadsPerBlock);
	case 'h': return chebyshevDist(cols, rows1, rows2, index1, index2, transpose, texData1, data2, texWt, maxGridSize, maxThreadsPerBlock);
	case 'g': return hammingDist(cols, rows1, rows2, index1, index2, transpose, texData1, data2, texWt, maxGridSize, maxThreadsPerBlock);
	default:
		return euclid(cols, rows1, rows2, index1, index2, transpose, texData1, data2, texWt, maxGridSize, maxThreadsPerBlock);
	}
	//return 0.0; /* Never get here */
}


/* ******************************************************************** */
__global__ void distancematrix(int nrows, int ncolumns, char dist, float *distancematix_m,
		cudaTextureObject_t texData, cudaTextureObject_t texWt,cudaTextureObject_t  texIdx,
		int *maxGridSize, int *maxThreadsPerBlock)
/*
	Purpose
	=======

	The distancematrix routine calculates the distance matrix between genes or
	microarrays using their measured gene expression data. Several distance measures
	can be used. The routine returns a pointer to a ragged array containing the
	distances between the genes. As the distance matrix is symmetric, with zeros on
	the diagonal, only the lower triangular half of the distance matrix is saved.
	The distancematrix routine allocates space for the distance matrix. If the
	parameter transpose is set to a nonzero value, the distances between the columns
	(microarrays) are calculated, otherwise distances between the rows (genes) are
	calculated.
	If sufficient space in memory cannot be allocated to store the distance matrix,
	the routine returns a NULL pointer, and all memory allocated so far for the
	distance matrix is freed.


	Arguments
	=========

	nrows      (input) int
	The number of rows in the gene expression data matrix (i.e., the number of
	genes)

	ncolumns   (input) int
	The number of columns in the gene expression data matrix (i.e., the number of
	microarrays)

	data       (input) double[nrows][ncolumns]
	The array containing the gene expression data.

	mask       (input) int[nrows][ncolumns]
	This array shows which data values are missing. If mask[i][j]==0, then
	data[i][j] is missing.

	weight (input) double[n]
	The weights that are used to calculate the distance. The length of this vector
	is equal to the number of columns if the distances between genes are calculated,
	or the number of rows if the distances between microarrays are calculated.

	dist       (input) char
	Defines which distance measure is used, as given by the table:
	dist=='e': Euclidean distance
	dist=='b': City-block distance
	dist=='c': correlation
	dist=='a': absolute value of the correlation
	dist=='u': uncentered correlation
	dist=='x': absolute uncentered correlation
	dist=='s': Spearman's rank correlation
	dist=='k': Kendall's tau
	For other values of dist, the default (Euclidean distance) is used.

	transpose  (input) int
	If transpose is equal to zero, the distances between the rows is
	calculated. Otherwise, the distances between the columns is calculated.
	The former is needed when genes are being clustered; the latter is used
	when microarrays are being clustered.

	========================================================================
	*/
{

	int transpose = 0;
	/* First determine the size of the distance matrix */
	const int n = (transpose == 0) ? nrows : ncolumns;
	const int ndata = (transpose == 0) ? ncolumns : nrows;

	if (n < 2) return;

	/* Calculate the distances and save them in the ragged array */

	int row = 0, col = 0;

	for(int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
				tIdx < TRI_COUNT(n) - n;
				tIdx += blockDim.x * gridDim.x){
	// Step 1 : solve the quadratic equation n^2 + n - 2 * tIdx to find row#

	int jIdx = tex1Dfetch<int>(texIdx, tIdx);

	row = floorf((-1 + sqrtf(1 - (4 * 1 * (-jIdx * 2)))) / 2);
	col = jIdx - ((row*(row + 1)) / 2);

	// Calculate row and col for the jagged distance matrix from the give tid

	// Step 2: compute distance for that row and column
//	if(row < nrows && col < ncolumns)
	distancematix_m[TRI_COUNT(row) + col] = calcDistMetricGPU(dist, ndata, n, n, row, col, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);

	}

	__syncthreads();

	return;
}

/**
 * Overloaded case for calculating distance from given object1 to all other objects
 */
__global__ void distancematrix(int nrows, int ncolumns, char dist, float *distancematix,
		cudaTextureObject_t texData, cudaTextureObject_t texWt, int row,
		int *maxGridSize, int *maxThreadsPerBlock)
/*
	Purpose
	=======

	The distancematrix routine calculates the distance matrix between genes or
	microarrays using their measured gene expression data. Several distance measures
	can be used. The routine returns a pointer to a ragged array containing the
	distances between the genes. As the distance matrix is symmetric, with zeros on
	the diagonal, only the lower triangular half of the distance matrix is saved.
	The distancematrix routine allocates space for the distance matrix. If the
	parameter transpose is set to a nonzero value, the distances between the columns
	(microarrays) are calculated, otherwise distances between the rows (genes) are
	calculated.
	If sufficient space in memory cannot be allocated to store the distance matrix,
	the routine returns a NULL pointer, and all memory allocated so far for the
	distance matrix is freed.


	Arguments
	=========

	nrows      (input) int
	The number of rows in the gene expression data matrix (i.e., the number of
	genes)

	ncolumns   (input) int
	The number of columns in the gene expression data matrix (i.e., the number of
	microarrays)

	data       (input) double[nrows][ncolumns]
	The array containing the gene expression data.

	mask       (input) int[nrows][ncolumns]
	This array shows which data values are missing. If mask[i][j]==0, then
	data[i][j] is missing.

	weight (input) double[n]
	The weights that are used to calculate the distance. The length of this vector
	is equal to the number of columns if the distances between genes are calculated,
	or the number of rows if the distances between microarrays are calculated.

	dist       (input) char
	Defines which distance measure is used, as given by the table:
	dist=='e': Euclidean distance
	dist=='b': City-block distance
	dist=='c': correlation
	dist=='a': absolute value of the correlation
	dist=='u': uncentered correlation
	dist=='x': absolute uncentered correlation
	dist=='s': Spearman's rank correlation
	dist=='k': Kendall's tau
	For other values of dist, the default (Euclidean distance) is used.

	transpose  (input) int
	If transpose is equal to zero, the distances between the rows is
	calculated. Otherwise, the distances between the columns is calculated.
	The former is needed when genes are being clustered; the latter is used
	when microarrays are being clustered.

	========================================================================
	*/
{

	int transpose = 0;
	/* First determine the size of the distance matrix */
	const int n = (transpose == 0) ? nrows : ncolumns;
	const int ndata = (transpose == 0) ? ncolumns : nrows;

	if (n < 2) return;

	/* Calculate the distances and save them in the ragged array */

	int col = 0;

	for(int i = blockIdx.x * blockDim.x + threadIdx.x;
			i < n;
			i += blockDim.x * gridDim.x){

		if(i == row){
			distancematix[i] = FLT_MAX;
		}
		else{
			distancematix[i] = calcDistMetricGPU(dist, ndata, n, n, row, i, transpose, texData, texWt, maxGridSize, maxThreadsPerBlock);
		}
	}

	__syncthreads();

	return;
}

/**
 * Special case for centroid and ward linkage in agglomerative clustering
 */

__global__ void distancematrix (int nrows, int ncolumns, cudaTextureObject_t texData , float* _centroids, cudaTextureObject_t texWt, int *active, char dist, int row, int transpose, float *distancematix, int *maxGridSize, int *maxThreadsPerBlock)

{ /* First determine the size of the distance matrix */
  const int n = (transpose==0) ? nrows : ncolumns;
  const int ndata = (transpose==0) ? ncolumns : nrows;
  int i,j;

  float *centroids = _centroids;

  if (n < 2) return;


//  for (i = 0; i < n; i++){
  for(int i = blockIdx.x * blockDim.x + threadIdx.x;
  				i < n;
  				i += blockDim.x * gridDim.x){

//			  distancematix[i] = metric(ndata,centroids,centroids,mask,mask,weights,row,active[i],transpose);
			  distancematix[i] = calcDistMetricGPU(dist, ndata, n, n, centroids, centroids, texWt,row, active[i], transpose);

	  }

  __syncthreads();
  return;
}

/* ******************************************************************** */


