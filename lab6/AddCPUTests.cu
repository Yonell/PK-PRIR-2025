#include <chrono>
#include <array>
#include <iostream>
#define N 32

int* global_a;
int* global_b;
int* global_c;
int* global_d_a;
int* global_d_b;
int* global_d_c;

__global__ void add(int* a, int* b, int* c) 
{
	//identyfikator wątku
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void random(int* tab, long long wym )
{	
	int i;
	for(i=0;i<wym;i++)
		tab[i]=rand()%101;
}

template<long long size>
void addCPU(int* a, int* b, int* c)
{
	for (long long i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}

template<long long size>
double testCPU()
{
    random(global_a, size);
    random(global_b, size);
    auto start = std::chrono::high_resolution_clock::now();
    addCPU<size>(global_a, global_b, global_c);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

template<long long size>
double testGPUWithoutCopy()
{
    random(global_a, size);
    random(global_b, size);

    long long dataSize = size * sizeof(int);
    cudaMemcpy(global_d_a, global_a, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(global_d_b, global_b, dataSize, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    add<<<1, size>>>(global_d_a, global_d_b, global_d_c);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(global_c, global_d_c, dataSize, cudaMemcpyDeviceToHost);

    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

template<long long size>
double testGPUWithCopy()
{
    random(global_a, size);
    random(global_b, size);

    long long dataSize = size * sizeof(int);
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(global_d_a, global_a, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(global_d_b, global_b, dataSize, cudaMemcpyHostToDevice);
    add<<<1, size>>>(global_d_a, global_d_b, global_d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(global_c, global_d_c, dataSize, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

template<long long size, int times>
void testAll()
{
	double cpuTime = 0.0;
	double gpuTimeNoCopy = 0.0;
	double gpuTimeWithCopy = 0.0;
	for (int i = 0; i < times; i++)
	{
		cpuTime += testCPU<size>();
		gpuTimeNoCopy += testGPUWithoutCopy<size>();
		gpuTimeWithCopy += testGPUWithCopy<size>();
	}
	cpuTime /= times;
	gpuTimeNoCopy /= times;
	gpuTimeWithCopy /= times;
	std::cout << "Size: " << size 
		<< " | mean CPU Time: " << cpuTime << " ms"
		<< " | mean GPU Time (no copy): " << gpuTimeNoCopy << " ms"
		<< " | GPU Time (with copy): " << gpuTimeWithCopy << " ms"
		<< std::endl;
}

template<auto sizes>
void templateLambdaForTest()
{
	auto test = []<long long size>()
	{
		testAll<size, 5>();
	};

    auto loop_unroller = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (test.template operator()<sizes[Is]>(), ...);
    };

    loop_unroller(std::make_index_sequence<sizes.size()>{});
}

void allocateGlobalMemory(long long maxSize) {
    long long dataSize = maxSize * sizeof(int);
    global_a = new int[maxSize];
    global_b = new int[maxSize];
    global_c = new int[maxSize];
    cudaMalloc((void**)&global_d_a, dataSize);
    cudaMalloc((void**)&global_d_b, dataSize);
    cudaMalloc((void**)&global_d_c, dataSize);
}

void freeGlobalMemory() {
    delete[] global_a;
    delete[] global_b;
    delete[] global_c;
    cudaFree(global_d_a);
    cudaFree(global_d_b);
    cudaFree(global_d_c);
}

int main(void) {
    
    static constexpr std::array<long long, 26> sizes =
		{ 1, 32, 64, 128, 256, 512, 1024, 2048, 4096,
		  8192, 16384, 32768, 65536, 131072, 262144,
		  524288, 1048576, 2097152, 4194304, 8388608,
		  16777216, 33554432, 67108864, 134217728, 268435456,
	      536870912 };
    allocateGlobalMemory(sizes.back());

    templateLambdaForTest<sizes>();

    freeGlobalMemory();
    return 0;
}


