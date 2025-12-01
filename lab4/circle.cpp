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
            MPI_Send(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&token, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Process " << rank << " received token " << token << " from process " << size - 1 << std::endl;
            std::cin >> token;
        }
        MPI_Send(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        return MPI_Finalize();
    }
    else {
        int prev = rank - 1;
        int next = (rank + 1) % size;
        while (true) {
            MPI_Recv(&token, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (token < 0) {
                MPI_Send(&token, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
                break;
            }
            std::cout << "Process " << rank << " received token " << token << " from process " << prev << std::endl;
            MPI_Send(&token, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        }
        return MPI_Finalize();
    }
}