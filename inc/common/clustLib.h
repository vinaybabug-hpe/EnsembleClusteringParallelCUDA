/*
* clustLib.h
*
*  Created on: Feb 27, 2016
*      Author: Michiel Jan Laurens de Hoon
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

#ifndef CLUSTLIB_H_
#define CLUSTLIB_H_

/*
* A Node struct describes a single node in a tree created by hierarchical
* clustering. The tree can be represented by an array of n Node structs,
* where n is the number of elements minus one. The integers left and right
* in each Node struct refer to the two elements or subnodes that are joined
* in this node. The original elements are numbered 0..nelements-1, and the
* nodes -1..-(nelements-1). For each node, distance contains the distance
* between the two subnodes that were joined.
*/
typedef struct { int left; int right; double distance; } Node;



//int binomial(int n, double p);
void randomassign(int nclusters, int nelements, int clusterid[]);

void hello_fastsc();
void fastsc(int nrows, int ncols, int numclusters,float *distmatrix, int* idxs);
void fastsc_full_distmat(int nrows, int ncols, int numclusters,float *distmatrix, int* idxs);

#endif /* CLUSTLIB_H_ */
