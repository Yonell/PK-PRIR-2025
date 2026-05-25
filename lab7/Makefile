# Compiler
NVCC = nvcc

# Targets
TARGETS = Calki

# Default target
all: $(TARGETS)

# Compile Calki.cu
Calki: Calki.cu
	$(NVCC) --std=c++20 -o $@ $< -arch=native

# Run Calki
run-Calki: Calki
	./Calki

# Clean up
clean:
	rm -f $(TARGETS)
