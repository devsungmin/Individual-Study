print("이 프로그램은 구구단 프로그램입니다,")

dan = int(input("몇단을 출력 할까요 ?"))

for i in range(1,10,1) :
    print("%d * %d = %d" %(dan,i,dan*i))