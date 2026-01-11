#include "mpi.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int CHUNK_SIZE = 10000 / size;

    srand(time(0));

    if (rank == 0) {
        //losowanie liczb
        std::vector<int> all_tokens(10000);
        for (int i = 0; i < 10000; ++i)
            all_tokens[i] = rand() % 101;

        // Liczę sumę kontrolną dla sprawdzenia potem
        int control_sum = std::accumulate(all_tokens.begin(), all_tokens.end(), 0);
        
        std::vector<int> local_tokens(CHUNK_SIZE);

        // Zapisuję czas startowy
        auto start_time = std::chrono::high_resolution_clock::now();

        // Proces 0 rozdziela tokeny po wątkach
        MPI_Scatter(all_tokens.data(), CHUNK_SIZE, MPI_INT, 
                    local_tokens.data(), CHUNK_SIZE, MPI_INT, 
                    0, MPI_COMM_WORLD);
        
        std::cout << "Process 0: Scattered data." << std::endl;

        // Proces sumuje swoją część
        int local_sum = 0;
        for (int val : local_tokens)
            local_sum += val;

        std::vector<int> gathered_sums(size);
        
        // I zbiera informacje o wyniku działania innych procesów
        MPI_Gather(&local_sum, 1, MPI_INT, 
                   gathered_sums.data(), 1, MPI_INT, 
                   0, MPI_COMM_WORLD);

        // Ostateczne sumowanie
        int total_sum = 0;
        for(int s : gathered_sums)
        {
            std::cout << "Main process: Received sum: " << s << std::endl;
            total_sum += s;
        }

        // Zapisanie czasu końcowego
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        // Wypisanie wartości na ekran
        std::cout << "Main process: Total sum is: " << total_sum << std::endl;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
        std::cout << "Control sum: " << control_sum << std::endl;
    }
    else {
        std::vector<int> local_tokens(CHUNK_SIZE);

        // Inne ranki dostają dane do zsumowania
        MPI_Scatter(nullptr, CHUNK_SIZE, MPI_INT, 
                    local_tokens.data(), CHUNK_SIZE, MPI_INT, 
                    0, MPI_COMM_WORLD);

        // Obliczają sumę
        int local_sum = 0;
        for (int val : local_tokens)
            local_sum += val;

        // I wysyłają ją do wątku głównego
        MPI_Gather(&local_sum, 1, MPI_INT, 
                   nullptr, 1, MPI_INT, 
                   0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}