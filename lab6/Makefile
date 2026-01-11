# Compiler
NVCC = nvcc

# Targets
TARGETS = CUDA Drugi Trzeci AddCPUTests AddCPUTests2 Shared_memory

# Default target
all: $(TARGETS)

# Compile AddCPUTests.cu
AddCPUTests: AddCPUTests.cu
	$(NVCC) --std=c++20 -o $@ $<

# Compile AddCPUTests.cu2
AddCPUTests2: AddCPUTests2.cu
	$(NVCC) --std=c++20 -o $@ $<

# Compile CUDA.cu
CUDA: CUDA.cu
	$(NVCC) --std=c++20 -o $@ $<

# Compile Drugi.cu
Drugi: Drugi.cu
	$(NVCC) --std=c++20 -o $@ $<

# Compile Trzeci.cu
Trzeci: Trzeci.cu
	$(NVCC) --std=c++20 -o $@ $<

# Compile Shared_memory.cu
Shared_memory: Shared_memory.cu
	$(NVCC) --std=c++20 -o $@ $<
	
# Run AddCPUTests
run-AddCPUTests: AddCPUTests
	./AddCPUTests
	
# Run AddCPUTests2
run-AddCPUTests2: AddCPUTests2
	./AddCPUTests2

# Run CUDA
run-CUDA: CUDA
	./CUDA

# Run Drugi
run-Drugi: Drugi
	./Drugi

# Run Trzeci
run-Trzeci: Trzeci
	./Trzeci

# Run Shared_memory
run-Shared_memory: Shared_memory
	./Shared_memory

# Clean up
clean:
	rm -f $(TARGETS)