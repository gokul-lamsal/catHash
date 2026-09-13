NVCC ?= nvcc
NVCC_ARCH ?= auto
ifeq ($(NVCC_ARCH),auto)
ARCHES := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/[^0-9.]//g' | sort -u | sed 's/\.//')
ifeq ($(strip $(ARCHES)),)
ARCHES := 89 120
endif
GENCODE := $(foreach a,$(ARCHES),-gencode arch=compute_$(a),code=sm_$(a))
else
GENCODE := -gencode arch=compute_$(NVCC_ARCH),code=sm_$(NVCC_ARCH)
endif

all: shibahash_cuda
shibahash_cuda: shibahash_cuda.cu
	$(NVCC) -O3 --use_fast_math $(GENCODE) -o $@ $<
clean:
	rm -f shibahash_cuda
