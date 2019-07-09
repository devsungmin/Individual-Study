#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "mpi.h"
#include <stdlib.h> //rand사용을 위해 추가
#include <time.h> //srand사용을 위해 추가
// 행렬 곱셈 커널 함수를 콜할 호스트 함수
cudaError_t multiWithCuda(float* c, float* a, float* b, unsigned int size);

// cuda 행렬 곱
__global__ void multiKernel(float* c, float* a, float* b, unsigned int size)
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    
    // 행렬의 곱셈 -> thread의 인덱스 값에 접근하여 계산
    for(int x=0; x<size; x++)
        c[size*j + i] += a[size*j + x] * b[size*x + i];
}

int main(int argc, char **argv)
{
    int rank,size;
    const int arraySize = 5;
    srand(time(NULL));
    /*MPI환경 초기화*/
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
   
    
    // 행렬 a,b,c을 1차원 배열로 하여 32 * 32 사이즈로 구성
    float a[arraySize*arraySize] = {0}; //행렬 크기는 1024
    float b[arraySize*arraySize] = {0}; //행렬 크기는 1024
    float c[arraySize*arraySize] = {0}; //행렬 크기는 1024

    //0부터 1024까지의 값이 들어감
    for(int i=0; i<arraySize*arraySize; i++)
    {
       	a[i]=2;
	b[i]=2;
	//a[i] = static_cast<float>(i);
        //b[i] = static_cast<float>(i);
	//a[i] = rand() % 1024;
	//b[i] = rand() % 1024;
    }


    // 작업할 함수를 콜한다.
    cudaError_t cudaStatus = multiWithCuda(c, a, b, arraySize);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "multiWithCuda failed!");
//	MPI_Finalize(); //MPI환경 제거
	return -1;
    }

    	/*send, recv을 활용하여 결과값을 출력*/
    	//node0은 master이고 node1은 slave가 되는 경우를 생각
    	// 단 slave는 여러개가 되어 다대다 연결이 가능하게 해야함
    	for(int i=0; i<arraySize*arraySize; i++) 
	{
		if(rank == 0)
		{
			//MPI_Send(address, count, datatype, destination, tag, comm)
                       	MPI_Send(&c[i], 0, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
   	   		printf("ping c[%d] = %8.1f\n",i,c[i]);
			MPI_Recv(&c[i], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		else
		{
                         //MPI_Recv(address, maxcount, datatype, source, tag, comm, status)
                         MPI_Recv(&c[i], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			 printf("c[%d] = %8.1f \n",i,c[i]);
			 MPI_Send(&c[i], 0, MPI_FLOAT,1,0,MPI_COMM_WORLD);
		}
    		// 모든 작업이 완료되었으므로
    		// device 를 reset 한다.
    		cudaStatus = cudaDeviceReset();
    		if(cudaStatus != cudaSuccess)
    		{ 
       		 	fprintf(stderr,"cudaDeviceReset, failed!");
			//MPI_Finalize();	
       		 return 1;
    		} 
		//MPI_Finalize();
	}
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

	cudaStatus = cudaMemcpy(dev_c, c, size*size*sizeof(float), cudaMemcpyHostToDevice);
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
//    MPI_Finalize();
    return cudaStatus;
}
