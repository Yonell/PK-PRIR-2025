

private: private.o
	gcc private.o -o private -fopenmp -lpthread
private.o: private.c
	gcc -c private.c -o private.o -fopenmp

conv: conv.o
	g++ conv.o -o conv -fopenmp -lpthread
conv.o: conv.cpp
	g++ -c -O3 conv.cpp -o conv.o -fopenmp


clean:
	rm -f *.o private conv