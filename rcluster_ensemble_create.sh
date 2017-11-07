#!/bin/bash 

#call executable with .mat files as arguments
./cluster_ensemble_create dataset/testcase1/PST_all.mat dataset/testcase1/X_50.mat dataset/testcase1/bootIdxs_50.mat dataset/testcase1/output_idxs.mat

#-nojvm -Dgdb t_mex2cuda.m


