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
        std::cin >> token;
        while (token >= 0) {
            MPI_Bcast(&token, 1, MPI_INT, 0, MPI_COMM_WORLD);
            std::cout << "Process " << rank << " received token " << token << std::endl;
            std::cin >> token;
        }
        MPI_Bcast(&token, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return MPI_Finalize();
    }
    else {
        while (true) {
            MPI_Bcast(&token, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (token < 0) {
                break;
            }
            std::cout << "Process " << rank << " received token " << token << std::endl;
        }
        return MPI_Finalize();
    }
}