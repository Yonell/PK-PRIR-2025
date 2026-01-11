#include<stdio.h>

#define N 30
#define BLOCK_SIZE 8
#define RADIUS 3

__global__ void stencil_1d(int *in, int *out) {
 	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
 	int lindex = threadIdx.x + RADIUS;
 	// Read input elements into shared memory
 	temp[lindex] = in[gindex];
 	if (threadIdx.x < RADIUS) 
 	{
		temp[lindex - RADIUS] = in[gindex - RADIUS];
		temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
 	}
 	// Synchronizacja wątków (bariera) - sprawdź co sie stanie bez tego
	__syncthreads();
	// Apply the stencil
 	int result = 0;
 	for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
 		result += temp[lindex + offset];
 	// Store the result
 	out[gindex] = result;
}

void random (int *tab, int wym )
{	
	int i;
	for(i=0;i<wym;i++)
		tab[i]=rand()%101;
}


int main(void) {
	int *in, *out; // host copies of in, out;
	int *d_in, *d_out; // device copies of in and out
	int size = N * sizeof(int);
	int i;
	srand(time(NULL));
	// Allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_in, size);
	cudaMalloc((void **)&d_out, size);	
	
	// Alloc space for host copies of a, b, c and setup input values
	in = (int *)malloc(size); random(in, N);
	out = (int *)malloc(size); 

	// Copy inputs to device
	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU
	stencil_1d<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(d_in, d_out);
	// Copy result back to host
	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

	for(i=0;i<N;i++)
	{
		printf("in[%d](%d) = out[%d](%d)\n",i,in[i],i,out[i]);
	}
	// Cleanup
	//printf("%d+%d=%d\n",a,b,c);
	free(in); free(out);
	cudaFree(d_in); cudaFree(d_out);
	return 0;
}


