#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<algorithm>
#include<time.h>
#include<cuda.h>
#include<mpi.h>

using namespace std;
MPI_Status status;

void Init_input(float* input, int input_h_size, int input_w_size)
{
   srand(time(NULL));
   for (int h = 0; h < input_h_size; h++)
   {
      for (int w = 0; w < input_w_size; w++)
      {
         input[(h * input_w_size) + w] = rand() % 10;
      }
   }
}

__global__ void Avg_pooling(int pooled_h, int pooled_w, int pool_h_stride, int pool_w_stride, int pool_h_size, int input_h_size, int pool_w_size, int input_w_size, int sum, float avg, float* gpu_input, float* gpu_output_data)
{
   int i = blockIdx.x;
   int j = blockIdx.y;
   int w_start = i * pool_w_stride;
   int h_start = j * pool_h_stride;
   int w_end = min(w_start + pool_w_size, input_w_size);
   int h_end = min(h_start + pool_h_size, input_h_size);

   w_start = max(w_start, 0);
   h_start = max(h_start, 0);
             
   sum=0;
   avg=0;

   int pool_index = (j * pooled_w) + i;
   for (int h = h_start; h < h_end; h++)
   {
      for (int w = w_start; w < w_end; w++)
      {
         int index = (h * input_w_size) + w;
         sum += gpu_input[index];
	   }
      avg = (float)sum / (pool_h_size * pool_w_size);
      gpu_output_data[pool_index] = avg;
   }
}

void print(float* data, int h_size, int w_size)
{
      for (int h = 0; h < h_size; h++)
      {
         for (int w = 0; w < w_size; w++)
         {
            printf("%.2f  ", data[(h * w_size) + w]);
         }
         cout << endl;
      }
      cout << endl;
      cout << endl;
}

int main(int argc, char** argv)
{
   int sum ;
   float avg;

   int input_h_size = 6;
   int input_w_size = 6;
   /*pool => window size*/
   int pool_w_size = 3;
   int pool_h_size = 3;
   int pool_w_stride = 3;
   int pool_h_stride = 3;

   int size;
   int rank;

   /*pooling 된 행렬들*/
   int pooled_h = ((input_h_size - pool_h_size) / pool_h_stride) + 1;
   int pooled_w = ((input_w_size - pool_w_size) / pool_w_stride) + 1;

   float* input = new float[input_h_size * input_w_size * sizeof(float)];
   float* cpu_output_data = new float[input_h_size * input_w_size* sizeof(float)];

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   Init_input(input, input_h_size, input_w_size);
   print(input, input_h_size, input_w_size);

   float* gpu_input;
   float* gpu_output_data; 

   cudaMalloc((void**)&gpu_input, input_h_size*input_w_size* sizeof(float));
   cudaMalloc((void**)&gpu_output_data, input_h_size*input_w_size* sizeof(float));

   //cudaMemcpy(gpu_input, input, input_h_size*input_w_size* sizeof(float), cudaMemcpyHostToDevice);	

   dim3 dimGrid(input_h_size, input_w_size);
   dim3 dimBlock(1, 1);
   
   if(rank == 0)
   {
      int start_range = ((input_h_size*input_w_size)/size)*rank;
      int end_range = ((rank+1)*((input_h_size*input_w_size)/size));

      for(int i=1; i<size; i++)
      {
         MPI_Send(input, input_h_size * input_w_size, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
      }
      cudaMemcpy(gpu_input, input, input_h_size*input_w_size* sizeof(float), cudaMemcpyHostToDevice);	
      Avg_pooling<<< dimGrid, dimBlock >>>(pooled_h, pooled_w, pool_h_stride, pool_w_stride, pool_h_size, input_h_size, pool_w_size, input_w_size, sum, avg, gpu_input, gpu_output_data);
      cudaMemcpy(cpu_output_data, gpu_output_data, input_h_size*input_w_size* sizeof(float), cudaMemcpyDeviceToHost);
   }
   else if(rank > 0)
  {
   int start_range = ((input_h_size*input_w_size)/size)*rank;
   int end_range = ((rank+1)*((input_h_size*input_w_size)/size));

    MPI_Recv(input, input_h_size * input_w_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);

    cudaMemcpy(gpu_input, input, input_h_size*input_w_size* sizeof(float), cudaMemcpyHostToDevice);	
    Avg_pooling<<< dimGrid, dimBlock >>>(pooled_h, pooled_w, pool_h_stride, pool_w_stride, pool_h_size, input_h_size, pool_w_size, input_w_size, sum, avg, gpu_input, gpu_output_data);
    cudaMemcpy(cpu_output_data, gpu_output_data, input_h_size*input_w_size* sizeof(float), cudaMemcpyDeviceToHost); 
  }
   
   //cudaDeviceSynchronize();

   //printf("====GPU Pooling Result value=====\n");
   print(cpu_output_data, pooled_h, pooled_w);

   
   cudaFree(gpu_input);
   cudaFree(gpu_output_data);

   delete input;
   delete cpu_output_data;

   return 0;
}