#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<algorithm>
#include<time.h>

using namespace std;

void Init_input(float* input, int input_channel, int input_h_size, int input_w_size)
{
   srand(time(NULL));

   for (int c = 0; c < input_channel; c++)
   {
      for (int h = 0; h < input_h_size; h++)
      {
         for (int w = 0; w < input_w_size; w++)
         {
            input[(c * input_h_size * input_w_size) + (h * input_w_size) + w] = rand() % 10;
         }
      }
   }
}

void print(float* data, int ch, int h_size, int w_size)
{
   for (int c = 0; c < ch; c++)
   {
      for (int h = 0; h < h_size; h++)
      {
         for (int w = 0; w < w_size; w++)
         {
            printf("%.2f ", data[(c * h_size * w_size) + (h * w_size) + w]);
         }
         cout << endl;
      }
      cout << endl;
      cout << endl;
   }
}

int main(int argc, char** argv)
{
   int sum;
   float avg;

   int input_h_size = 6;
   int input_w_size = 6;
   int input_channel = 1;

   int pool_w_size = 3;
   int pool_h_size = 3;
   int pool_w_stride = 3;
   int pool_h_stride = 3;

   int pooled_h = ((input_h_size - pool_h_size) / pool_h_stride) + 1;
   int pooled_w = ((input_w_size - pool_w_size) / pool_w_stride) + 1;

   float* input = new float[input_channel * input_h_size * input_w_size];

   Init_input(input, input_channel, input_h_size, input_w_size);
   print(input, input_channel, input_h_size, input_w_size);

   float* cpu_output_data = new float[input_channel * input_h_size * input_w_size];

   for (int c = 0; c < input_channel; c++)
   {
      for (int ph = 0; ph < pooled_h; ph++)
      {
         for (int pw = 0; pw < pooled_w; pw++)
         {
            int h_start = ph * pool_h_stride;
            int w_start = pw * pool_w_stride;
            int h_end = min(h_start + pool_h_size, input_h_size);
            int w_end = min(w_start + pool_w_size, input_w_size);
            h_start = max(h_start, 0);
            w_start = max(w_start, 0);
            sum = 0;
            avg = 0;

            int pool_index = (c * pooled_h * pooled_w) + (ph * pooled_w) + pw;
            for (int h = h_start; h < h_end; h++)
            {
               for (int w = w_start; w < w_end; w++)
               {
                  int index = (c * input_h_size * input_w_size) + (h * input_w_size) + w;
                  sum += input[index];
               }
            }
            avg = (float)sum / (pool_h_size * pool_w_size);

            cpu_output_data[pool_index] = avg;
         }
      }
   }

   print(cpu_output_data, input_channel, pooled_h, pooled_w);
   
   delete input;
   delete cpu_output_data;

   return 0;
}
