ARPACKPP_DIR = $(HOME)/methods_fitting_cuda/ClusterEnsembleCreate/external/arpack++
CUDA_CPP = nvcc
CUDA_ARCH ?= sm_50
include  $(ARPACKPP_DIR)/Makefile.inc
CUDA_FLAGS = -arch=$(CUDA_ARCH) -Xptxas -v
CUDA_LIBS = -lcublas -lcusparse

spectral_clustering: spectral_clustering.cu timer.o labels.o kmeans.h centroids.h
	$(CUDA_CPP) -Xcompiler '$(CPP_FLAGS) -Wno-variadic-macros' -I$(ARPACKPP_DIR)/include $(CUDA_FLAGS) -o spectral_clustering spectral_clustering.cu timer.o labels.o $(ALL_LIBS) $(CUDA_LIBS) 

labels.o: labels.cu labels.h
	$(CUDA_CPP) -Xcompiler '$(CPP_FLAGS) -Wno-variadic-macros' $(CUDA_FLAGS)  -c -o labels.o labels.cu 
	   
timer.o: timer.cu timer.h
	$(CUDA_CPP) -Xcompiler '$(CPP_FLAGS) -Wno-variadic-macros' $(CUDA_FLAGS) -c -o timer.o timer.cu
