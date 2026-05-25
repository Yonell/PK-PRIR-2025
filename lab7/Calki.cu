#include <iostream>
#include <array>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

__device__ double d_function(double x)
{
	return 4/(1+x*x);
}

__host__ double h_function(double x)
{
	return 4/(1+x*x);
}

__global__ void gpuIntegral(double a, double b, int n, double *globalResult)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double h = (b - a) / n;

    if (index < n) {
        double x = a + (index + 0.5) * h;
        sdata[tid] = h * d_function(x);
    } else {
        sdata[tid] = 0.0;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(globalResult, sdata[0]);
    }
}

__host__ double cpuIntegral(double a, double b, int n)
{
	double h, sum = 0.0, x;
	h = (b - a) / n;
	for (int i = 0; i < n; i++) {
		x = a + (i + 0.5) * h;
		sum += h_function(x);
	}
	return h * sum;
}

template<std::pair<int, int> TestCase>
void runTest()
{
	int n = TestCase.first * TestCase.second;
	double a = 0.0;
	double b = 1.0;

	// CPU computation
	auto start_cpu = std::chrono::high_resolution_clock::now();
	double cpuResult = cpuIntegral(a, b, n);
	auto end_cpu = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> cpuDuration = end_cpu - start_cpu;

	// GPU computation
	double *d_result;
	cudaMalloc((void**)&d_result, sizeof(double));
	cudaMemset(d_result, 0, sizeof(double));
	int blockSize = TestCase.second;
	int numBlocks = (n + blockSize - 1) / blockSize;
	auto start_gpu = std::chrono::high_resolution_clock::now();
	gpuIntegral<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(a, b, n, d_result);
	cudaDeviceSynchronize();
	auto end_gpu = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> gpuDuration = end_gpu - start_gpu;

	double gpuResult;
	// We need to copy only the first element for final result
	cudaMemcpy(&gpuResult, d_result, sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_result);

	std::cout << "Test Case: n: " << std::setw(12) << n
			  << ", numBlocks: " << std::setw(12) << numBlocks
			  << ", blockSize: " << std::setw(12) << blockSize
	          << ", CPU time: " << std::setw(12) << std::fixed << std::setprecision(6) << cpuDuration.count() << " ms"
	          << ", GPU time: " << std::setw(12) << std::fixed << std::setprecision(6) << gpuDuration.count() << " ms"
	          << ", CPU result: " << std::setw(12) << std::fixed << std::setprecision(6) << cpuResult
	          << ", GPU result: " << std::setw(12) << std::fixed << std::setprecision(6) << gpuResult
	          << std::endl;
}

template<auto TestCases>
void runAllTests()
{
	auto lambdaUnroll = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (runTest<TestCases[Is]>(), ...);
    };

	lambdaUnroll(std::make_index_sequence<TestCases.size()>{});
}

int main(void) {
	constexpr std::array<std::pair<int, int>, 16> testCases = {
		std::make_pair(16, 16),
		std::make_pair(64, 16),
		std::make_pair(256, 16),
		std::make_pair(256, 32),
		std::make_pair(256, 64),
		std::make_pair(1024, 64),
		std::make_pair(4096, 64),
		std::make_pair(4096, 128),
		std::make_pair(4096, 256),
		std::make_pair(16384, 256),
		std::make_pair(32768, 256),
		std::make_pair(65536, 256),
		std::make_pair(131072, 256),
		std::make_pair(262144, 512),
		std::make_pair(524288, 512),
		std::make_pair(1048576, 1024)
	};
	runAllTests<testCases>();

	std::cout << "----------------------------------------" << std::endl;
	std::cout << "Tests for constant n but different block sizes:" << std::endl;

	constexpr std::array<std::pair<int, int>, 8> testCases2 = {
		std::make_pair(1048576, 8),
		std::make_pair(524288, 16),
		std::make_pair(262144, 32),
		std::make_pair(131072, 64),
		std::make_pair(65536, 128),
		std::make_pair(32768, 256),
		std::make_pair(16384, 512),
		std::make_pair(8192, 1024)

	};
	runAllTests<testCases2>();
	return 0;
}


