
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <array>
#include <random>
#include <chrono>

template<long unsigned int height, long unsigned int width, int precision = 5>
void printMatrix(const std::array<std::array<double, width>, height>& p_arr)
{
    for (int i=0; i<height; i++)
    {
        for (int j=0; j<width; j++)
        {
            std::cout << std::fixed << std::setprecision(precision) << " " << p_arr[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<long unsigned int arrHeight, long unsigned int arrWidth, long unsigned int kerHeight, long unsigned int kerWidth>
std::array<std::array<double, arrWidth - kerWidth + 1>, arrHeight - kerHeight + 1> conv(
    const std::array<std::array<double, arrWidth>, arrHeight>& p_arr,
    const std::array<std::array<double, kerWidth>, kerHeight>& p_ker
)
{
    std::array<std::array<double, arrWidth - kerWidth + 1>, arrHeight - kerHeight + 1> l_result;

    for (int i=0; i<arrHeight - kerHeight + 1; i++)
    {
        for(int j=0; j<arrWidth - kerWidth + 1; j++)
        {
            l_result[i][j] = 0;
            for (int iKer=0; iKer<kerHeight; iKer++)
            {
                for (int jKer=0; jKer<kerWidth; jKer++)
                {
                    l_result[i][j] += p_ker[iKer][jKer]*p_arr[i+iKer][j+jKer];
                }
            }
        }
    }

    return l_result;
}

template<long unsigned int arrHeight, long unsigned int arrWidth, long unsigned int kerHeight, long unsigned int kerWidth>
std::array<std::array<double, arrWidth - kerWidth + 1>, arrHeight - kerHeight + 1> conv_par(
    const std::array<std::array<double, arrWidth>, arrHeight>& p_arr,
    const std::array<std::array<double, kerWidth>, kerHeight>& p_ker
)
{
    std::array<std::array<double, arrWidth - kerWidth + 1>, arrHeight - kerHeight + 1> l_result;

    #pragma omp parallel for 
    for (int i=0; i<arrHeight - kerHeight + 1; i++)
    {
        #pragma omp parallel for 
        for(int j=0; j<arrWidth - kerWidth + 1; j++)
        {
            l_result[i][j] = 0;
            for (int iKer=0; iKer<kerHeight; iKer++)
            {
                for (int jKer=0; jKer<kerWidth; jKer++)
                {
                    l_result[i][j] += p_ker[iKer][jKer]*p_arr[i+iKer][j+jKer];
                }
            }
        }
    }

    return l_result;
}

template<long unsigned int arrHeight, long unsigned int arrWidth>
std::array<std::array<double, arrWidth>, arrHeight> fillRandom(std::array<std::array<double, arrWidth>, arrHeight>& p_arr)
{
    std::random_device l_rd;
    std::mt19937 l_mt(l_rd());
    std::uniform_real_distribution<double> l_rng(0.0, 1.0);

    for (int i=0; i<arrHeight; i++)
    {
        for (int j=0; j<arrWidth; j++)
        {
            p_arr[i][j] = l_rng(l_mt);
        }
    }

    return p_arr;
}

template<long unsigned int arrHeight, long unsigned int arrWidth>
bool isSame(
    const std::array<std::array<double, arrWidth>, arrHeight>& p_arr1,
    const std::array<std::array<double, arrWidth>, arrHeight>& p_arr2
)
{
    for (int i=0; i<arrHeight; i++)
    {
        for (int j=0; j<arrWidth; j++)
        {
            if (p_arr1[i][j] != p_arr2[i][j])
            return false;
        }
    }
    return true;
}

template<long unsigned int arrHeight, long unsigned int arrWidth, long unsigned int kerHeight, long unsigned int kerWidth>
std::pair<std::chrono::duration<double, std::nano>, std::chrono::duration<double, std::nano>> experiment()
{
    std::array<std::array<double, arrWidth>, arrHeight> l_arr{};
    l_arr = fillRandom(l_arr);
    std::array<std::array<double, kerWidth>, kerHeight> l_ker{};
    l_ker = fillRandom(l_ker);

    
    auto l_start1 = std::chrono::high_resolution_clock::now();
    auto l_conv1 = conv(l_arr, l_ker);
    auto l_stop1 = std::chrono::high_resolution_clock::now();
    auto l_start2 = std::chrono::high_resolution_clock::now();
    auto l_conv2 = conv_par(l_arr, l_ker);
    auto l_stop2 = std::chrono::high_resolution_clock::now();

    std::cout << "Check if same: " << isSame(l_conv1, l_conv2) << std::endl;

    return {(l_stop1 - l_start1), (l_stop2 - l_start2)};
}

int main()
{


    auto l_time1 = experiment<5, 5, 2, 2>();
    auto l_time2 = experiment<15, 15, 5, 5>();
    auto l_time3 = experiment<500, 500, 15, 15>();
    auto l_time4 = experiment<1000, 1000, 15, 15>();
    auto l_time5 = experiment<2500, 2500, 30, 30>();

    std::cout << "Time1: " << l_time1.first.count() << "ns\n";
    std::cout << "Time1 parallel: " << l_time1.second.count() << "ns\n";
    std::cout << "Time2: " << l_time2.first.count() << "ns\n";
    std::cout << "Time2 parallel: " << l_time2.second.count() << "ns\n";
    std::cout << "Time3: " << l_time3.first.count() << "ns\n";
    std::cout << "Time3 parallel: " << l_time3.second.count() << "ns\n";
    std::cout << "Time4: " << l_time4.first.count() << "ns\n";
    std::cout << "Time4 parallel: " << l_time4.second.count() << "ns\n";
    std::cout << "Time5: " << l_time5.first.count() << "ns\n";
    std::cout << "Time5 parallel: " << l_time5.second.count() << "ns\n";


}