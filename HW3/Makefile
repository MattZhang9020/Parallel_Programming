CXXFLAGS = -lm -O3
CFLAGS = -fopenmp -msse4.2
NVFLAGS = -Xcompiler -fopenmp -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 --default-stream per-thread
HIPCCFLAGS = -fopenmp -std=c++11 -O3 --offload-arch=gfx90a
LDFLAGS = -lm

EXES = seq hw3-1 hw3-2 hw3-3 hw3-2-amd hw3-3-amd

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

hw3-1: hw3-1.cc
	g++ $(CXXFLAGS) $(CFLAGS) -o $@ $?

hw3-2: hw3-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-2-amd: hw3-2.hip
	hipcc $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<

hw3-3-amd: hw3-3.hip
	hipcc $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<