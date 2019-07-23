#include <stdio.h>
#include <cuda.h>
#include "mpi.h"

#define ROW 32 //가로
#define COL 32 //세로
#define stride 2
#define kernal_size 4

MPI_Status status;

int size, rank; //MPI환경을 위한 변수 설정

__global__ void Average_Pooling(float* dev_a,float* dev_b, float* dev_c, int arr_size,int start_range, int end_range) {
	int i = threadIdx.x;
	int j = blockIdx.x;

	if(start_range <= j && j < end_range) {
		for(int x=0; x<arr_size; x++) {
			dev_c[arr_size * j + i] += dev_a[arr_size * j + x] * dev_b[arr_size * x + i];
		}
	}	
}

int main(int argc, char **argv) {
	int arr_size = ROW*COL; // 32 * 32 = 1024	
	/*행렬 선언 및 초기화*/
	float*  a[arr_size]; 
	float*  b[arr_size];
	float*  c[arr_size];
	/*gpu 행렬*/
	float* dev_a[arr_size];
	float* dev_b[arr_size];
	float* dev_c[arr_size];

	int offset=0;
	int before_offset=0;
	
	size_t bytes = arr_size * arr_size * sizeof(float);
	
	/*MPI 환경 초기화*/
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	a = (float*)malloc(bytes);
	b = (float*)malloc(bytes);
	c = (float*)malloc(bytes);

	for(int i=0;i<arr_size; i++) {
		/*0 ~ 1023까지 정적으로 고정 되어 들어감*/
		a[i] = static_cast<float>;
		b[i] = static_cast<float>;
		c[i] = 0;
	}

	cudaMalloc((void**)&dev_a, bytes);
	cudaMalloc((void**)&dev_b, bytes);
	cudaMalloc((void**)&dev_c, bytes);

	cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess) {
		fprintf("cudaDeviceSynchronize returned error code %d after launching a kernel!!\n", cudaStatus);
		return -1;
	}

	if(rank == 0) {
		//MPI_Recv
		//MPI_Recv(buf_host, size, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
	}
	else if(rank > 0) {
		//MPI_Send
		//MPI_Send(buf_host, size, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
	}

	free(a);
	free(b);
	free(c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	MPI_Finalze(); //MPI환경 초기화
	return 0;
}
