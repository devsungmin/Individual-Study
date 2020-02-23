# 피보나치 수열
def fibo(n) :
    if n < 2 :
    	return n
    return fibo(n-1) + fibo(n-2)

def main() :
    n = int(input("input insert : "))
    print("n = %d fibo : %d" %(n,fibo(n)))

    return 0

if __name__ == "__main__" :
	main()
