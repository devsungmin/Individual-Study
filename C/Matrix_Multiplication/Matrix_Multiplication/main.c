#include <stdio.h>
#include <stdlib.h>

/*함수 정의*/
int mutrix(); //행렬 곱셉

/*메인 함수*/
int main() {
	mutrix();
	system("pause");
	return 0;
}

/*행렬 곱셈*/
int mutrix() {
	//변수 초기화
	int i, j, k;

	//행렬 초기화
	int A[4][4] = { {1,1,1,1},
					{1,1,1,1},
					{1,1,1,1},
					{1,1,1,1} };

	int B[4][4] = { {1,1,1,1},
					{1,1,1,1},
					{1,1,1,1},
					{1,1,1,1} };
	int C[4][4] = { 0 };

	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			for (k = 0; k < 4; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}

	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4;j++){
			printf("c[%d][%d] = %d\t",i,j,C[i][j]);
		}
		printf("\n");
	}
}