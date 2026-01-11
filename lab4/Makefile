LOAD_MODULES = \
	source /etc/profile.d/modules.sh && \
	module load mpi/openmpi-x86_64 && \

all: example circle broadcast procedure

run_all: run_example run_circle run_broadcast run_procedure

# ------------------------------------------------------------------------------

example.o: example.cpp
	$(LOAD_MODULES) mpicxx -c example.cpp -o example.o

example: example.o
	$(LOAD_MODULES) mpicxx example.o -o example

clean_example:
	rm -f example.o example

run_example: example
	$(LOAD_MODULES) mpirun -np 2 ./example

.PHONY: all clean_example run_example

# ------------------------------------------------------------------------------

circle.o: circle.cpp
	$(LOAD_MODULES) mpicxx -c circle.cpp -o circle.o

circle: circle.o
	$(LOAD_MODULES) mpicxx circle.o -o circle

clean_circle:
	rm -f circle.o circle

run_circle: circle
	$(LOAD_MODULES) mpirun -np 5 ./circle

.PHONY: clean_circle run_circle

# ------------------------------------------------------------------------------

broadcast.o: broadcast.cpp
	$(LOAD_MODULES) mpicxx -c broadcast.cpp -o broadcast.o

broadcast: broadcast.o
	$(LOAD_MODULES) mpicxx broadcast.o -o broadcast

clean_broadcast:
	rm -f broadcast.o broadcast

run_broadcast: broadcast
	$(LOAD_MODULES) mpirun -np 5 ./broadcast

.PHONY: clean_broadcast run_broadcast

# ------------------------------------------------------------------------------

procedure.o: procedure.cpp
	$(LOAD_MODULES) mpicxx -c procedure.cpp -o procedure.o

procedure: procedure.o
	$(LOAD_MODULES) mpicxx procedure.o -o procedure

clean_procedure:
	rm -f procedure.o procedure

run_procedure: procedure
	$(LOAD_MODULES) mpirun --oversubscribe -np 6 ./procedure

.PHONY: clean_procedure run_procedure

# ------------------------------------------------------------------------------

clean: clean_example clean_circle clean_broadcast clean_procedure