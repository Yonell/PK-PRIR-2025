#include "mpi.h"
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int token;
    if (rank == 0)
    {
        // Rank 0 dostaje token
        std::cin >> token;
        while (token >= 0) {
            // I broadcastuje liczby póki są nieujemne
            MPI_Bcast(&token, 1, MPI_INT, 0, MPI_COMM_WORLD);
            std::cout << "Process " << rank << " received token " << token << std::endl;
            std::cin >> token;
        }
        // Jeśli są nieujemne to broadcastuje i wychodzi
        MPI_Bcast(&token, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return MPI_Finalize();
    }
    else {
        while (true) {
            // inne ranki czekają na liczbę
            MPI_Bcast(&token, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (token < 0) {
                // Jeśli jest ujemna to wychodzą
                break;
            }
            // Jeśli nie, to wypisują i czekają dalej
            std::cout << "Process " << rank << " received token " << token << std::endl;
        }
        return MPI_Finalize();
    }
}