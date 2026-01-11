LOAD_MODULES = \
	source /etc/profile.d/modules.sh && \
	module load mpi/openmpi-x86_64 && \

all: main

run: run_main

# ------------------------------------------------------------------------------

main.o: main.cpp
	$(LOAD_MODULES) mpicxx -c main.cpp -o main.o

main: main.o
	$(LOAD_MODULES) mpicxx main.o -o main

clean_main:
	rm -f main.o main

run_main: main
	$(LOAD_MODULES) mpirun -np 3 ./main

.PHONY: all clean_main run_main

# ------------------------------------------------------------------------------

clean: clean_main