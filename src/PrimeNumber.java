import java.util.ArrayList;
import java.util.List;

public class PrimeNumber {
    // 查找小于param的质数
    public static void findPrimeLessParam(int param) {
        for (int i = 2; i < param; i++) {
            boolean isPrime = true;// 注意两个循环的起始值均为2，都为2时，不进入第二个循环，即默认2为素数
            for (int j = 2; j < i; j++) {
                if (i % j == 0) {
                    isPrime = false;
                    break;
                }
            }
            if (isPrime) {
                System.out.print(i + " ");
            }
        }
        System.out.println();
    }

    public int[] prime(int n) {
        boolean arr[] = new boolean[n + 1];
        int max = (int) Math.sqrt(n);
        for (int i = 2; i <= max; i++) {
            if (!arr[i]) {
                for (int j = i + i; j <= n; j += i) {
                    arr[j] = true;
                }
            }

        }
        List<Integer> res = new ArrayList<>();
        for (int i = 2; i <= n; i++) {
            if (!arr[i]) {
                res.add(i);
            }
        }
        int resarr[] = new int[res.size()];
        int i = 0;
        for (int num : res) {
            resarr[i++] = num;
        }

        return resarr;
    }

    public static void main(String[] arg) {
        findPrimeLessParam(200);
        int arr[] = new PrimeNumber().prime(200);
        for (int i : arr) {
            System.out.print(i + " ");
        }
        System.out.println();
    }

}
