package dev.sungmin;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        System.out.println("이 프로그램은 사칙연산을 해주는 프로그램입니다!!");
        Scanner scan = new Scanner(System.in);
        System.out.println("첫번째 숫자를 입력하시오 : ");
        int num = scan.nextInt();
        System.out.println("두번째 숫자를 입력하시오 : ");
        int num2 = scan.nextInt();
        System.out.println("어떠한 연산을 할지 입력 하시오 : (ex. + * - /)");
        String Op = scan.next();

        if(Op.equals("+")) {
            int sum = num + num2;
            System.out.println("덧셈을 입력 하셨습니다 : " + sum);
        } else if (Op.equals("-")) {
            int sub = num - num2;
            System.out.println("뺄셈을 입력 하셨습니다 : " + sub);
        } else if (Op.equals("*")) {
            int mul = num*num2;
            System.out.println("곱셉을 입력 하셨습니다 : "+ mul);
        } else if (Op.equals("/")) {
            int div = num / num2;
            System.out.println("나눗셈을 입력 하셨습니다 : " + div);
        }
    }
}
