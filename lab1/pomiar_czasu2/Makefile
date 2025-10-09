# kompilator c
CCOMP = gcc

# konsolidator
LOADER = gcc

# opcje optymalizacji:
# wersja do debugowania
# OPT = -g -DDEBUG
# wersja zoptymalizowana do mierzenia czasu
OPT = -O3

# pliki naglowkowe
INC = -I./inc

# biblioteki
LIB = -L./lib -lm

run: moj_program
	build/moj_program

# zaleznosci i komendy
moj_program: moj_program.o
	$(LOADER) $(OPT) $(LIB) build/moj_program.o -lpomiar_czasu -o build/moj_program

# jak uzyskac plik moj_program.o ?
moj_program.o: src/moj_program.c inc/pomiar_czasu.h libpomiar_czasu
	$(CCOMP) -c $(OPT) $(INC) src/moj_program.c -o build/moj_program.o
	


pomiar_czasu.o: src/pomiar_czasu.c inc/pomiar_czasu.h
	mkdir -p build
	$(CCOMP) -c $(OPT) $(INC) src/pomiar_czasu.c -o build/pomiar_czasu.o

libpomiar_czasu: pomiar_czasu.o
	mkdir -p lib
	ar rc lib/libpomiar_czasu.a build/pomiar_czasu.o

clean:
	rm -fr build
	rm -fr lib
