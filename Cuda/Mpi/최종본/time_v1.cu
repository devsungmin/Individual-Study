#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

MPI_Status status;

__global__ void matrixMul(float* MatA, float* MatB, float* MatC, int arr_size, int start_range, int end_range)
{
	int i = threadIdx.x;
	int j = blockIdx.x;	
	
	if(start_range<=j && j<end_range)
	{
		for(int x=0 ;x<arr_size ; x++)
		{	
			MatC[arr_size*j + i] += MatA[arr_size*j + x] * MatB[arr_size * x + i];
		}
	}	
}

int main(int argc, char** argv)
{
	int n = 1024;
	int offset=0;
	int before_offset=0;
	int size, myrank;

	float* host_MatA;
	float* host_MatB;
	float* host_MatC;
	float* host_tmp;	

	float* dev_MatA;
	float* dev_MatB;
	float* dev_MatC;

	size_t bytes = n * n * sizeof(float);
	
	clock_t start, end;
	float result = 0;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	host_MatA = (float*)malloc(bytes);
	host_MatB = (float*)malloc(bytes);
	host_MatC = (float*)malloc(bytes);
	host_tmp = (float*)malloc(bytes);

	for(int i = 0; i < n; i++)
        {
        	for(int j = 0; j < n; j++)
                {
                         host_MatA[i * n + j] = 1;//rand() % 32;
                         host_MatB[i * n + j] = 1;//rand() % 32;
			 host_MatC[i * n + j] = 0;
			 host_tmp[i * n + j] = 0;
                }
        }

	cudaMalloc((void**)&dev_MatA, bytes);
	cudaMalloc((void**)&dev_MatB, bytes);
	cudaMalloc((void**)&dev_MatC, bytes);
       	
	start = clock();	

	if(myrank == 0)
        {
                int start_range = (n/size)*(myrank);
                int end_range = ((myrank+1)*(n/size));

		for(int i=1; i<size; i++)
		{
			MPI_Send(host_MatA, n*n, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
	        	MPI_Send(host_MatB, n*n, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
                        MPI_Send(host_MatC, n*n, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
		}

		cudaMemcpy(dev_MatA, host_MatA, bytes, cudaMemcpyHostToDevice);
                cudaMemcpy(dev_MatB, host_MatB, bytes, cudaMemcpyHostToDevice);
                cudaMemcpy(dev_MatC, host_MatC, bytes, cudaMemcpyHostToDevice);
				
		matrixMul<<<n, n>>>(dev_MatA, dev_MatB, dev_MatC, n, start_range, end_range);
		cudaDeviceSynchronize();
		cudaMemcpy(host_MatC, dev_MatC, bytes, cudaMemcpyDeviceToHost);
		
		offset = (int)n/size;
		for(int i=1; i<size ; i++)
		{
			MPI_Recv(host_tmp, n*n, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
			before_offset = offset;
			offset+=(n/size);
	                for(int i = before_offset; i < offset; i++)
        	        {
                	         for(int j = 0; j < n; j++)
                       		 {
                                	host_MatC[i * n + j] = host_MatC[i * n + j] + host_tmp[i * n + j];
                        	 }
                	}
		}
        }
	else if(myrank > 0)
        {
                int start_range = (n/size)*(myrank);
                int end_range = ((myrank+1)*(n/size));

	        float* slave_MatA = (float*)malloc(bytes);
        	float* slave_MatB = (float*)malloc(bytes);
	        float* slave_MatC = (float*)malloc(bytes);
	
                MPI_Recv(slave_MatA, n*n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(slave_MatB, n*n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(slave_MatC, n*n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
		
                cudaMemcpy(dev_MatA, slave_MatA, bytes, cudaMemcpyHostToDevice);
                cudaMemcpy(dev_MatB, slave_MatB, bytes, cudaMemcpyHostToDevice);
                cudaMemcpy(dev_MatC, slave_MatC, bytes, cudaMemcpyHostToDevice);

		matrixMul<<<n, n>>>(dev_MatA, dev_MatB, dev_MatC, n, start_range, end_range);
		cudaDeviceSynchronize();
		cudaMemcpy(slave_MatC, dev_MatC, bytes, cudaMemcpyDeviceToHost);
				
		MPI_Send(slave_MatC, n*n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
	        free(slave_MatA);
       		free(slave_MatB);
        	free(slave_MatC);
        }
	cudaDeviceSynchronize();
	end = clock();
	result = (float)(end - start)/CLOCKS_PER_SEC;	
	/*
	if(myrank == 0)
	{
		for(int i = 0; i < n*n; i++)
        	{
                	if(i%n == 0) printf("\n");
                	printf("[%d]%.1f ",i, host_MatC[i]);
        	}
	}
	*/
	printf("rank : %d  time : %.4f\n", myrank, result);
	
	free(host_MatA);
	free(host_MatB);
	free(host_MatC);
	free(host_tmp);

	cudaFree(dev_MatA);
	cudaFree(dev_MatB);
	cudaFree(dev_MatC);
	
	MPI_Finalize();
	return 0;
}
