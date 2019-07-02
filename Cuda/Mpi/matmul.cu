#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "mpi.h"

// 행렬 곱셈 커널 함수를 콜할 호스트 함수
cudaError_t multiWithCuda(float* c, float* a, float* b, unsigned int size);

// cuda 행렬 곱
__global__ void multiKernel(float* c, float* a, float* b, unsigned int size)
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    
    // 행렬의 곱셈 -> thread의 인덱스 값에 접근하여 계
    for(int x=0; x<size; x++)
        c[size*j + i] += a[size*j + x] * b[size*x + i];
}

int main(int argc, char **argv)
{
    int rank,size;
    const int arraySize = 32;
    MPI_Init(&argc, &argv); //mpi초기화
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 행렬 a,b,c 를 만든다.
    float a[arraySize*arraySize] = {0}; //1024
    float b[arraySize*arraySize] = {0}; //1024
    float c[arraySize*arraySize] = {0}; //1024

    // 알맞은 값으로 초기화 한다. -> 0부터 1024까지의 값이 들어
    for(int i=0; i<arraySize*arraySize; i++)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i);
    }

    // 작업할 함수를 콜한다.
    cudaError_t cudaStatus = multiWithCuda(c, a, b, arraySize);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "multiWithCuda failed!");
	MPI_Finalize();
        return -1;
    }

    // 결과를 출력한다.
    for(int i=0; i<arraySize*arraySize; i++)
    {
        if(i % arraySize == 0) { 
	printf("\n");
	MPI_Send(&c[i], 0, MPI_FLOAT,1,0,MPI_COMM_WORLD);
	printf("%8.1f ",c[i]);
	}
	else {
	MPI_Recv(&c[i],1,MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	printf("%8.1f ",c[i]);
	}
    }
    printf("\n");

    // 모든 작업이 완료되었으므로
    // device 를 reset 한다.
    cudaStatus = cudaDeviceReset();
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr,"cudaDeviceReset, failed!");
	MPI_Finalize();	
        return 1;
    }
    MPI_Finalize();
    return 0;
}

// 커널함수 호출하는 헬퍼 함수 multiWithCuda를 정의하자
cudaError_t multiWithCuda(float* c, float* a, float* b, unsigned int size)
{
    // gpu에 할당한 메모리 주소값을 저장할 변수를 선언한다.
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "CudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // GPU에 메모리를 할당한다.
    // 행렬 크기만큼 할당한다.
    cudaStatus = cudaMalloc((void**)&dev_c, size*size*sizeof(float));
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr,"cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_a, size*size*sizeof(float));
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr,"cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_b, size*size*sizeof(float));
    if(cudaStatus != cudaSuccess)
    {

        fprintf(stderr,"cudaMalloc failed!");
        goto Error;
    }

    // 호스트 메모리에 있는 값을 디바이스 메모리에 복사한다.
    cudaStatus = cudaMemcpy(dev_a, a, size*size*sizeof(float), cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
     cudaStatus = cudaMemcpy(dev_b, b, size*size*sizeof(float), cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // 커널 함수를 실행한다.
    multiKernel<<<size, size>>>(dev_c, dev_a, dev_b, size);

    // 커널 함수 실행후 에러가 있는지 확인
    cudaStatus = cudaGetLastError();
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "multiKernel launch failed : %s\n", cudaGetErrorString(cudaStatus));
	goto Error;
    }


    // 커널이 모두 종료되었는지 확인
    cudaStatus  = cudaDeviceSynchronize();
    if(cudaStatus!= cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // 결과를 호스트 메모리에 복사
    cudaStatus = cudaMemcpy(c, dev_c, size*size*sizeof(float), cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
// gpu에 할당한 메모리를 반환
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    //mpi 해제
    MPI_Finalize();
    return cudaStatus;
}
