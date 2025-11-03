#include <omp.h>
#include <iostream>
#include <iomanip>

int main()
{
	int l_threadCount = 12;
	omp_set_num_threads(l_threadCount);
	int l_portionSize = 3;
	int size = 150000;

    auto l_startStaticSetPortionSize = omp_get_wtime();
	#pragma omp parallel for schedule(static, l_portionSize)
	for (int i=0; i<size; i++)
	{
		std::cout << "thread " << omp_get_thread_num() << " index " << i << "\n";
	}
    auto l_stopStaticSetPortionSize = omp_get_wtime();

    auto l_startStaticDefaultPortionSize = omp_get_wtime();
	#pragma omp parallel for schedule(static)
	for (int i=0; i<size; i++)
	{
		std::cout << "thread " << omp_get_thread_num() << " index " << i << "\n";
	}
    auto l_stopStaticDefaultPortionSize = omp_get_wtime();

    auto l_startDynamicSetPortionSize = omp_get_wtime();
	#pragma omp parallel for schedule(dynamic, l_portionSize)
	for (int i=0; i<size; i++)
	{
		std::cout << "thread " << omp_get_thread_num() << " index " << i << "\n";
	}
    auto l_stopDynamicSetPortionSize = omp_get_wtime();

    auto l_startDynamicDefaultPortionSize = omp_get_wtime();
	#pragma omp parallel for schedule(dynamic)
	for (int i=0; i<size; i++)
	{
		std::cout << "thread " << omp_get_thread_num() << " index " << i << "\n";
	}
    auto l_stopDynamicDefaultPortionSize = omp_get_wtime();

    std::cout << "\n";

    std::cout << "Wyniki\n";
    std::cout << std::left << std::setw(20) << "typ"
              << std::setw(20) << "portion_size"
              << "wynik" << "\n";
    std::cout << std::left << std::setw(20) << "Static" 
              << std::setw(20) << "Ustawiony" 
              << std::right << std::setw(20) << l_stopStaticSetPortionSize - l_startStaticSetPortionSize << "\n";
    std::cout << std::left << std::setw(20) << "Static" 
              << std::setw(20) << "Domyslny" 
              << std::right << std::setw(20) << l_stopStaticDefaultPortionSize - l_startStaticDefaultPortionSize << "\n";
    std::cout << std::left << std::setw(20) << "Dynamic" 
              << std::setw(20) << "Ustawiony" 
              << std::right << std::setw(20) << l_stopDynamicSetPortionSize - l_startDynamicSetPortionSize << "\n";
    std::cout << std::left << std::setw(20) << "Dynamic" 
              << std::setw(20) << "Domyslny" 
              << std::right << std::setw(20) << l_stopDynamicDefaultPortionSize - l_startDynamicDefaultPortionSize << "\n";

    std::cout << "\n";

    float l_resultWithout = 10000000000;
    float l_resultWith = 1000000000000;

    float base = 1.1;

    #pragma omp parallel for
    for (int i=0; i<1000; i++)
    {
        l_resultWithout /= base;
    }

    #pragma omp parallel for reduction(+:l_resultWith)
    for (int i=0; i<1000; i++)
    {
        l_resultWith /= base;
    }

    std::cout << "Wynik:\n";
    std::cout << std::left << std::setw(30) << "Klauzula" << std::setw(30) << "Wynik dodawania" << "\n";
    std::cout << std::left << std::setw(30) << "Wyłączona" << std::right << std::setw(30) << l_resultWithout << "\n";
    std::cout << std::left << std::setw(30) << "Włączona" << std::right << std::setw(30) << l_resultWith << "\n";


}