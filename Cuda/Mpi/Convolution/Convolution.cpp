#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>

using namespace std;

/*함수 초기화 부분*/
void Init_input(int input_w_size, int input_h_size, int input_channel, float* input)
{
    srand(time(NULL));

    for (int c = 0; c < input_channel; c++)
    {
        for (int i = 0; i < input_h_size; i++)
        {
            for (int j = 0; j < input_w_size; j++)
            {
                input[(c * input_h_size * input_w_size) + (i * input_w_size)] = rand() % 100;
            }
        }
    }
}

/*함수 출력 부분*/
void print(float* data, int ch, int h_size, int w_size)
{
   for (int c = 0; c < ch; c++)
   {
      for (int h = 0; h < h_size; h++)
      {
         for (int w = 0; w < w_size; w++)
         {
            printf("%.2f  ", data[(c * h_size * w_size) + (h * w_size) + w]);
         }
         cout << endl;
      }
      cout << endl;
      cout << endl;
   }
}


// /*컨볼루션 함수*/
// void convolution(int input_w_size,int input_h_size,int input_channel,int filter_h_size, int filter_w_size)
// {

// }

int main()
{
    int input_w_size = 6; //가로 사이즈
    int input_h_size = 6; //세로 사이즈
    int input_channel = 1; //채널 사이즈

    int filter_w_size = 3;
    int filter_h_size = 3;

    int stride_w_size = 3;
    int stride_h_size = 3;

    float input_size = input_w_size * input_h_size;

    float* input = (float*)malloc(sizeof(float) * input_size);

    /*초기 함수 랜덤 설정*/
    Init_input(input_w_size, input_h_size, input_channel, input);
    print(input, input_channel, input_h_size, input_w_size);
    //convolution(input_w_size,input_h_size,input_channel,filter_h_size, filter_w_size);

    free(input);

    return 0;
}