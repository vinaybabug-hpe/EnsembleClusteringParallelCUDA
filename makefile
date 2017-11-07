NAME:=cluster_ensemble_create
LDIR=lib
ODIR = obj
SDIR = src
EXTERNAL=external
CPP_ODIR = obj/cpp
CU_ODIR = obj/cpp

ARPACKPP_DIR = $(HOME)/Documents/cuda_ensemble/methods_fitting_cuda/ClusterEnsembleCreate/external/arpack++
MATLAB_INSTALL_PATH := /public/apps/mdcs/R2015b
CUDA_INSTALL_PATH := /public/apps/cuda/7.5
SDK	:= /public/apps/cuda/7.5/samples



CUDA  := $(CUDA_INSTALL_PATH)

MABLAB := $(MATLAB_INSTALL_PATH)

include  $(ARPACKPP_DIR)/Makefile.inc

INC := -Iinc -Iexternal -I$(CUDA)/include -I$(SDK)/common/inc -I$(MABLAB)/extern/include
LIBS := -L$(CUDA)/lib64 -L$(SDK)/lib/linux/x86_64 -L$(MATLAB_INSTALL_PATH)/bin/glnxa64 -L$(ARPACKPP_DIR)/external


# IMPORTANT : don't forget the CUDA runtime (-lcudart) !
CUDA_LIBS:= -lcudart -lcusparse -lcublas -lcudadevrt
#put non cuda -l* stuff here
NONCU_LIBS:=

CC = gcc
NVCC= nvcc
MEX_EXE=mex
MEX_NVCC=mexcuda
AR = ar

CFLAGS:=-fPIC -fno-inline -w -c -O2
NVCC_CFLAGS:=-std=c++11 -Drestrict=__restrict
#-w -m64 -gencode arch=compute_35,code=sm_35 should be given as direct text otherwise nvcc is throwing error
NVCC_MACHINE:=-w -m64 -dc -gencode arch=compute_35,code=sm_35  
LFLAGS := -Wall
MEX_CFLAGS := "$(MATLAB_INSTALL_PATH)/extern/lib/glnxa64/mexFunction.map"
COMPFLAGS :=-fPIC -Wall


all: clean mkdir_o 
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cluster_util_ssw.o $(SDIR)/cluster_util_ssw.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cluster_util_partition2cam.o $(SDIR)/cluster_util_partition2cam.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cluster_util_indices2centers.o $(SDIR)/cluster_util_indices2centers.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cluster_ensemble2cam.o $(SDIR)/cluster_ensemble2cam.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/indices_count.o $(SDIR)/indices_count.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/utility_functions.o $(SDIR)/utility_functions.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/zscore.o $(SDIR)/zscore.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cluster_data_outliers.o $(SDIR)/cluster_data_outliers.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cluster_util_bootpartition2partition.o $(SDIR)/cluster_util_bootpartition2partition.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/mex2cuda.o $(SDIR)/mex2cuda.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/distCalcMthds.o $(SDIR)/distCalcMthds.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -Xcompiler '$(CPP_FLAGS) -Wno-variadic-macros $(COMPFLAGS)' $(INC) $(CUDA_FLAGS) $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/timer.o $(EXTERNAL)/fastsc/timer.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -Xcompiler '$(CPP_FLAGS) -Wno-variadic-macros $(COMPFLAGS)' $(INC) $(CUDA_FLAGS) $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/labels.o $(EXTERNAL)/fastsc/labels.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -Xcompiler '$(CPP_FLAGS) -Wno-variadic-macros $(COMPFLAGS)' $(INC) $(CUDA_FLAGS) $(NVCC_MACHINE) $(NVCC_CFLAGS) -c  -o $(ODIR)/spectral_clustering.o $(EXTERNAL)/fastsc/spectral_clustering.cu $(ALL_LIBS) $(CUDA_LIBS) $(LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cuda_kmeans.o $(EXTERNAL)/northwestern/ece/wkliao/cuda_kmeans.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cuda_kmedians.o $(EXTERNAL)/northwestern/ece/wkliao/cuda_kmedians.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)	
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cdpAdvancedQuicksort.o $(EXTERNAL)/nvidia7_5/samples/advanced/cdpAdvancedQuicksort.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cdpBitonicSort.o $(EXTERNAL)/nvidia7_5/samples/advanced/cdpBitonicSort.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/invert_matrix.o $(EXTERNAL)/rochester/invert_matrix.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/gaussian_kernel.o $(EXTERNAL)/rochester/gaussian_kernel.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)	
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cuda_gmm.o $(EXTERNAL)/rochester/cuda_gmm.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cuda_agglomerative.o $(EXTERNAL)/cluster_3_0/cuda_agglomerative.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)	
	$(NVCC) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cluster_ensemble_create.o $(SDIR)/cluster_ensemble_create.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) -ccbin g++ -Xcompiler \"-Wl,-rpath,$(MATLAB_INSTALL_PATH)/bin/glnxa64\" $(NVCC_CFLAGS) -w -m64 -gencode arch=compute_35,code=sm_35 -o $(NAME) $(ODIR)/cluster_ensemble_create.o \
	$(ODIR)/mex2cuda.o $(ODIR)/distCalcMthds.o $(ODIR)/timer.o $(ODIR)/labels.o $(ODIR)/spectral_clustering.o $(ODIR)/utility_functions.o $(ODIR)/cuda_kmeans.o $(ODIR)/cuda_kmedians.o $(ODIR)/cdpAdvancedQuicksort.o \
	$(ODIR)/cdpBitonicSort.o $(ODIR)/cuda_gmm.o $(ODIR)/invert_matrix.o $(ODIR)/gaussian_kernel.o $(ODIR)/cuda_agglomerative.o $(ODIR)/cluster_data_outliers.o $(ODIR)/zscore.o $(ODIR)/cluster_util_bootpartition2partition.o \
	$(ODIR)/indices_count.o $(ODIR)/cluster_ensemble2cam.o $(ODIR)/cluster_util_indices2centers.o $(ODIR)/cluster_util_partition2cam.o $(ODIR)/cluster_util_ssw.o \
	-lmat -lmx $(INC) $(ALL_LIBS) $(CUDA_LIBS) $(NONCU_LIBS) $(LIBS) -L$(ODIR)/  

debug: clean mkdir_o	
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cluster_util_ssw.o $(SDIR)/cluster_util_ssw.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cluster_util_partition2cam.o $(SDIR)/cluster_util_partition2cam.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cluster_util_indices2centers.o $(SDIR)/cluster_util_indices2centers.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cluster_ensemble2cam.o $(SDIR)/cluster_ensemble2cam.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/indices_count.o $(SDIR)/indices_count.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/utility_functions.o $(SDIR)/utility_functions.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/zscore.o $(SDIR)/zscore.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cluster_data_outliers.o $(SDIR)/cluster_data_outliers.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cluster_util_bootpartition2partition.o $(SDIR)/cluster_util_bootpartition2partition.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/mex2cuda.o $(SDIR)/mex2cuda.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/distCalcMthds.o $(SDIR)/distCalcMthds.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -Xcompiler '$(CPP_FLAGS) -Wno-variadic-macros $(COMPFLAGS)' $(INC) $(CUDA_FLAGS) $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/timer.o $(EXTERNAL)/fastsc/timer.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -Xcompiler '$(CPP_FLAGS) -Wno-variadic-macros $(COMPFLAGS)' $(INC) $(CUDA_FLAGS) $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/labels.o $(EXTERNAL)/fastsc/labels.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -Xcompiler '$(CPP_FLAGS) -Wno-variadic-macros $(COMPFLAGS)' $(INC) $(CUDA_FLAGS) $(NVCC_MACHINE) $(NVCC_CFLAGS) -c  -o $(ODIR)/spectral_clustering.o $(EXTERNAL)/fastsc/spectral_clustering.cu $(ALL_LIBS) $(CUDA_LIBS) $(LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cuda_kmeans.o $(EXTERNAL)/northwestern/ece/wkliao/cuda_kmeans.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cuda_kmedians.o $(EXTERNAL)/northwestern/ece/wkliao/cuda_kmedians.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)	
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cdpAdvancedQuicksort.o $(EXTERNAL)/nvidia7_5/samples/advanced/cdpAdvancedQuicksort.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cdpBitonicSort.o $(EXTERNAL)/nvidia7_5/samples/advanced/cdpBitonicSort.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/invert_matrix.o $(EXTERNAL)/rochester/invert_matrix.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/gaussian_kernel.o $(EXTERNAL)/rochester/gaussian_kernel.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)	
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cuda_gmm.o $(EXTERNAL)/rochester/cuda_gmm.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cuda_agglomerative.o $(EXTERNAL)/cluster_3_0/cuda_agglomerative.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/fibo.o $(EXTERNAL)/cluster_3_0/fibo.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler '$(COMPFLAGS)' $(INC) -L$(ODIR)/ $(NVCC_MACHINE) $(NVCC_CFLAGS) -c -o $(ODIR)/cluster_ensemble_create.o $(SDIR)/cluster_ensemble_create.cu $(LIBS) $(CUDA_LIBS) $(NONCU_LIBS)
	$(NVCC) $(DFLAGS) -ccbin g++ -Xcompiler \"-Wl,-rpath,$(MATLAB_INSTALL_PATH)/bin/glnxa64\" $(NVCC_CFLAGS) -w -m64 -gencode arch=compute_35,code=sm_35 -o $(NAME) $(ODIR)/cluster_ensemble_create.o \
	$(ODIR)/mex2cuda.o $(ODIR)/distCalcMthds.o $(ODIR)/timer.o $(ODIR)/labels.o $(ODIR)/spectral_clustering.o $(ODIR)/utility_functions.o $(ODIR)/cuda_kmeans.o $(ODIR)/cuda_kmedians.o $(ODIR)/cdpAdvancedQuicksort.o \
	$(ODIR)/cdpBitonicSort.o $(ODIR)/cuda_gmm.o $(ODIR)/invert_matrix.o $(ODIR)/gaussian_kernel.o $(ODIR)/cuda_agglomerative.o $(ODIR)/fibo.o $(ODIR)/cluster_data_outliers.o $(ODIR)/zscore.o $(ODIR)/cluster_util_bootpartition2partition.o \
	$(ODIR)/indices_count.o $(ODIR)/cluster_ensemble2cam.o $(ODIR)/cluster_util_indices2centers.o $(ODIR)/cluster_util_partition2cam.o $(ODIR)/cluster_util_ssw.o \
	-lmat -lmx $(INC) $(ALL_LIBS) $(CUDA_LIBS) $(NONCU_LIBS) $(LIBS) -L$(ODIR)/ 	
	
#$(MEX_EXE) -v -g $(INC) -L$(ODIR)/  -output $(NAME) "obj/cluster_ensemble_create.o" $(ODIR)/mex2cuda.o $(ODIR)/distCalcMthds.o $(ODIR)/timer.o $(ODIR)/labels.o $(ODIR)/spectral_clustering.o -lmat -lmx $(ALL_LIBS) $(CUDA_LIBS) $(LIBS) LDFLAGS="$(ALL_LIBS) $(CUDA_LIBS) $(LIBS)"


Lib_cuda.a: cuda    
	${AR} -r $(ODIR)/libmycuda.a $(ODIR)/mex2cuda.o $(ODIR)/mex2cuda_link.o $(ODIR)/distCalcMthds.o $(ODIR)/timer.o $(ODIR)/labels.o $(ODIR)/spectral_clustering.o	

cuda: mkdir_o spectral_clustering.o
	$(NVCC) $(NVCC_MACHINE) $(NVCC_CFLAGS) $(INC)  $(SDIR)/mex2cuda.cu  -rdc=true -shared -Xcompiler '-fpic' -c -o $(ODIR)/mex2cuda.o
	$(NVCC) $(NVCC_MACHINE) $(NVCC_CFLAGS) $(INC)  $(SDIR)/distCalcMthds.cu  -rdc=true -shared -Xcompiler '-fpic' -c -o $(ODIR)/distCalcMthds.o	
	$(NVCC) $(NVCC_MACHINE) -Xcompiler '$(CPP_FLAGS) -Wno-variadic-macros -fpic' $(CUDA_FLAGS) -c -o $(ODIR)/timer.o $(EXTERNAL)/fastsc/timer.cu $(LIBS) $(CUDA_LIBS)
	$(NVCC) $(NVCC_MACHINE) -Xcompiler '$(CPP_FLAGS) -Wno-variadic-macros -fpic' $(CUDA_FLAGS)  -c -o $(ODIR)/labels.o $(EXTERNAL)/fastsc/labels.cu $(LIBS) $(CUDA_LIBS)
	$(NVCC) $(NVCC_MACHINE) -Xcompiler '$(CPP_FLAGS) -Wno-variadic-macros -fpic' -I$(ARPACKPP_DIR)/include $(CUDA_FLAGS) -c  -o $(ODIR)/spectral_clustering.o $(EXTERNAL)/fastsc/spectral_clustering.cu $(ALL_LIBS) $(CUDA_LIBS) $(LIBS)	
	$(NVCC) -w -m64 -gencode arch=compute_50,code=sm_50 -dlink -shared -Xcompiler '-fpic' -o $(ODIR)/mex2cuda_link.o $(ODIR)/mex2cuda.o $(ODIR)/distCalcMthds.o $(ODIR)/timer.o $(ODIR)/labels.o $(ODIR)/spectral_clustering.o $(CUDA_LIBS) $(INC) $(NVCC_CFLAGS) $(ALL_LIBS)

spectral_clustering.o: timer.o labels.o $(EXTERNAL)/fastsc/kmeans.h $(EXTERNAL)/fastsc/centroids.h
	$(NVCC) $(NVCC_MACHINE) -Xcompiler '$(CPP_FLAGS) -Wno-variadic-macros' $(CUDA_FLAGS) -c  -o $(ODIR)/spectral_clustering.o $(EXTERNAL)/fastsc/spectral_clustering.cu $(ALL_LIBS) $(CUDA_LIBS) 

labels.o: $(EXTERNAL)/fastsc/labels.h
	$(NVCC) $(NVCC_MACHINE) -Xcompiler '$(CPP_FLAGS) -Wno-variadic-macros' $(CUDA_FLAGS)  -c -o $(ODIR)/labels.o $(EXTERNAL)/fastsc/labels.cu 
	   
timer.o: $(EXTERNAL)/fastsc/timer.h
	$(NVCC) $(NVCC_MACHINE) -Xcompiler '$(CPP_FLAGS) -Wno-variadic-macros' $(CUDA_FLAGS) -c -o $(ODIR)/timer.o $(EXTERNAL)/fastsc/timer.cu	

$(ODIR)/%.o: $(SDIR)/%.c mkdir_o
	$(CC) -c $(CFLAGS) $(INC) -o $@ $<  

$(ODIR)/%.ou: $(SDIR)/%.cu mkdir_o
	$(NVCC) -c $(CFLAGS) $(INC) -o $@ $<


mkdir_o:	
	@mkdir -p $(ODIR)
	@mkdir -p $(ODIR)/common	
	
	
	
clean:
	rm -f $(NAME) $(ODIR)/*.o $(ODIR)/*/*.o $(ODIR)/*/*.*.o $(ODIR)/*.*.o $(OUT) $(ODIR)/*/*.a $(ODIR)/*.a *.mexa64
	
	
