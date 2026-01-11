#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm> 
#include <random>    
#include <thread>    
#include <chrono>    
#include <iterator>

enum MPI_TAG
{
    TAG_CALL_ELEVATOR = 100,
    TAG_PERMIT_BOARD,
    TAG_SELECT_FLOOR,
    TAG_NOTIFY_EXIT,
    TAG_DONE_FOR_DAY
};

struct PassengerRequest
{
    int rank;
    int floor;
};

void sleep_ms(int ms)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

void elevator_procedure(int size)
{
    int current_level = 0;
    int direction = 0; 
    int active_passengers = size - 1; 

    std::vector<PassengerRequest> waiting_queue;
    std::vector<PassengerRequest> inside_elevator;

    std::cout << "elevator: Started. Idle at floor 0\n";

    while (active_passengers > 0)
    {
        MPI_Status status;
        int flag = 0;
        bool state_changed = false;

        // Sprawdzanie requestów
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_CALL_ELEVATOR, MPI_COMM_WORLD, &flag, &status);
        while (flag)
        {
            int floor_call;
            MPI_Recv(&floor_call, 1, MPI_INT, status.MPI_SOURCE, TAG_CALL_ELEVATOR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            waiting_queue.push_back({status.MPI_SOURCE, floor_call});

            std::cout << "elevator: Call received from Rank " << status.MPI_SOURCE << " at floor " << floor_call << "\n";

            state_changed = true;

            MPI_Iprobe(MPI_ANY_SOURCE, TAG_CALL_ELEVATOR, MPI_COMM_WORLD, &flag, &status);
        }
        if (state_changed)
        {
            std::sort(waiting_queue.begin(), waiting_queue.end(), [](PassengerRequest a, PassengerRequest b){ return a.floor < b.floor; });
            std::cout << "elevator: requests: ";
            std::for_each(
                waiting_queue.begin(),
                waiting_queue.end(),
                [](auto e){ std::cout << e.floor << " "; }
            );
            std::cout << "\n";
        }

        // Sprawdzanie pasażerów, którzy zakończyli
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_DONE_FOR_DAY, MPI_COMM_WORLD, &flag, &status);
        if (flag)
        {
            int dummy;
            MPI_Recv(&dummy, 1, MPI_INT, status.MPI_SOURCE, TAG_DONE_FOR_DAY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            active_passengers--;
        }

        bool people_entered_or_left = false;

        // Wyrzucanie ludzi z windy
        auto it_out = std::remove_if(
            inside_elevator.begin(), 
            inside_elevator.end(), 
            [&](const PassengerRequest& p)
            {
                if (p.floor == current_level)
                {
                    int dummy = 0;
                    MPI_Send(&dummy, 1, MPI_INT, p.rank, TAG_NOTIFY_EXIT, MPI_COMM_WORLD);
                    std::cout << "elevator: Dropping off Rank " << p.rank << " at floor " << current_level << "\n";
                    people_entered_or_left = true;
                    return true; 
                }
                return false;
            });
        inside_elevator.erase(it_out, inside_elevator.end());

        // Zabieranie ludzi w tym samym kierunku
        auto it_in = std::remove_if(
            waiting_queue.begin(), 
            waiting_queue.end(), 
            [&](const PassengerRequest& p)
            {
                if (p.floor == current_level)
                {
                    MPI_Send(&direction, 1, MPI_INT, p.rank, TAG_PERMIT_BOARD, MPI_COMM_WORLD);
                    
                    int dest_floor;
                    MPI_Recv(&dest_floor, 1, MPI_INT, p.rank, TAG_SELECT_FLOOR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (dest_floor == -1)
                    {
                        return false;
                    }

                    inside_elevator.push_back({p.rank, dest_floor});
                    std::cout << "elevator: Rank " << p.rank << " entered at floor " << current_level << ", dest: " << dest_floor << "\n";
                    people_entered_or_left = true;
                    return true; 
                }
                return false;
            });
        std::sort(inside_elevator.begin(), inside_elevator.end(), [](PassengerRequest a, PassengerRequest b){ return a.floor < b.floor; });
        waiting_queue.erase(it_in, waiting_queue.end());

        // Sprawdzanie statusu windy
        if (inside_elevator.empty() && waiting_queue.empty())
        {
            if (direction != 0)
            {
                std::cout << "elevator: No requests. Becoming IDLE at floor " << current_level << "\n";
                direction = 0;
            }
        }
        else
        {
            int target_floor = -1;
            
            if (!inside_elevator.empty())
            {
                target_floor = inside_elevator.front().floor; 
            }
            else if (!waiting_queue.empty())
            {
                target_floor = waiting_queue.front().floor;
            }

            if (target_floor > current_level) direction = 1;
            else if (target_floor < current_level) direction = -1;
            else direction = 0; 
        }

        // Zmiana piętra
        if (direction != 0)
        {
            std::cout << "elevator: going from " << current_level << " to " << current_level + direction << "\n";
            sleep_ms(500); 
            current_level += direction;
        }
        else
        {
            sleep_ms(100);
        }
    }
    std::cout << "elevator: All passengers finished. Shutting down.\n";
}

void passenger_procedure(int rank)
{
    std::random_device rd;
    std::mt19937 gen(rd() + rank); 
    std::uniform_int_distribution<> floor_dist(0, 9); 
    std::uniform_int_distribution<> trips_dist(3, 6); 
    std::uniform_int_distribution<> wait_dist(500, 2000); 

    int trips = trips_dist(gen);
    int current_floor = floor_dist(gen);

    std::cout << "passenger " << rank << ": spawned at floor " << current_floor << ". Planning " << trips << " trips.\n";

    for (int i = 0; i < trips; ++i)
    {
        sleep_ms(wait_dist(gen));

        int dest_floor;
        do {
            dest_floor = floor_dist(gen);
        } while (dest_floor == current_floor);

        std::cout << "passenger " << rank << ": calling from " << current_floor << " to " << dest_floor << "\n";
        MPI_Send(&current_floor, 1, MPI_INT, 0, TAG_CALL_ELEVATOR, MPI_COMM_WORLD);

        int direction = -1;
        int wrong_direction = -1;
        int dummy = 0;
        do {
            MPI_Recv(&direction, 1, MPI_INT, 0, TAG_PERMIT_BOARD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (direction*(dest_floor-current_floor) <= 0)
            {
                std::cout << "passenger " << rank << ": Wrong direction, not entering\n";
                std::cout << "passenger " << rank << ": Wrong direction, waiting on " << current_floor << " to go to " << dest_floor << " floor\n";
                MPI_Send(&wrong_direction, 1, MPI_INT, 0, TAG_SELECT_FLOOR, MPI_COMM_WORLD);
            }
        } while (direction*(dest_floor-current_floor) < 0);

        MPI_Send(&dest_floor, 1, MPI_INT, 0, TAG_SELECT_FLOOR, MPI_COMM_WORLD);

        MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_NOTIFY_EXIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        current_floor = dest_floor;
        std::cout << "passenger " << rank << ": Arrived at floor " << current_floor << ".\n";
    }

    int dummy = 0;
    MPI_Send(&dummy, 1, MPI_INT, 0, TAG_DONE_FOR_DAY, MPI_COMM_WORLD);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    if (size < 2)
    {
        if (rank == 0) std::cerr << "Error: Run with at least 2 processes\n";
    }
    else
    {
        if (rank == 0)
        {
            elevator_procedure(size);
        }
        else
        {
            passenger_procedure(rank);
        }
    }

    MPI_Finalize();
    return 0;
}