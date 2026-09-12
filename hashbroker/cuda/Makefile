NVCC ?= nvcc
CFLAGS ?= -O3 --use_fast_math
TARGET ?= hashbroker_cuda

# Set NVCC_ARCH manually to override detection, e.g. NVCC_ARCH=89.
# With NVCC_ARCH=auto, compile one native target for every GPU visible to nvidia-smi.
# With NVCC_ARCH=all, compile the CUDA 12.4-compatible common targets for release builds.
NVCC_ARCH ?= auto

ifeq ($(NVCC_ARCH),auto)
GPU_ARCHES := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/[^0-9.]//g' | sort -u | sed 's/\.//')
ifeq ($(strip $(GPU_ARCHES)),)
GPU_ARCHES := 75 80 86 89 90
endif
GENCODE := $(foreach arch,$(GPU_ARCHES),-gencode arch=compute_$(arch),code=sm_$(arch))
else ifeq ($(NVCC_ARCH),all)
GENCODE := $(foreach arch,75 80 86 89,-gencode arch=compute_$(arch),code=sm_$(arch))
else
ARCH_NUM := $(shell echo $(NVCC_ARCH) | sed 's/[^0-9]//g')
GENCODE := -gencode arch=compute_$(ARCH_NUM),code=sm_$(ARCH_NUM)
endif

$(TARGET): hashbroker_cuda.cu
	$(NVCC) $(CFLAGS) $(GENCODE) -o $@ $<

clean:
	rm -f hashbroker_cuda
