#include "mpi.h"
#include <iostream>

int main(int argc, char** argv) {
    // Inicjalizujemy MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int token;
    if (rank == 0)
    {
        // Rank 0 odbiera liczbę
        std::cin >> token;
        // Jeśli jest nieujemna
        while (token >= 0) {
            // To wysyła ją synchronicznie do następnego procesu
            MPI_Send(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            // I czeka, aż obiegnie kółko
            MPI_Recv(&token, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Process " << rank << " received token " << token << " from process " << size - 1 << std::endl;
            std::cin >> token;
        }
        // Ujemną też wysyła dalej
        MPI_Send(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        return MPI_Finalize();
    }
    else {
        int prev = rank - 1;
        int next = (rank + 1) % size;
        while (true) {
            // Inne ranki czekają na liczbę
            MPI_Recv(&token, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // I wysyłają dalej
            if (token < 0) {
                // Jeśli jest mniejsza niż 0 to wysyła dalej i wychodzi
                MPI_Send(&token, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
                break;
            }
            // Jeśli nie, to wysyła dalej i czeka na kolejną liczbę w następnym przejściu
            std::cout << "Process " << rank << " received token " << token << " from process " << prev << std::endl;
            MPI_Send(&token, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        }
        return MPI_Finalize();
    }
}