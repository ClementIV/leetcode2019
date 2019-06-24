import java.util.ArrayDeque;
import java.util.Deque;

public class SortStack {
    // poj 3250
    private void poj3250(long[] a, int n) {
        Deque<Integer> deque = new ArrayDeque<Integer>();
        long ans = 0, t = 0;
        for (int i = 0; i <= n; i++) {
            while (!deque.isEmpty() && a[deque.peekLast()] <= a[i]) {//如果栈不为空并且栈顶元素不大于入栈元素，则将栈顶元素出栈
                t = deque.pollLast(); //栈定元素
                // 这时候也就找到了第一个比栈顶元素大的元素
                // 计算这之间牛的个数，为下标之差 -1；
                ans += (i - t - 1);
            }
            deque.addLast(i); //当所有破坏栈的单调性的元素都出栈后，将当前元素入栈

        }

        System.out.println(ans);
    }

    //poj 2559
    private static void poj2559(long[] a,int n){
        Deque<Integer> deque = new ArrayDeque<Integer>();
        long ans = 0;
        int t =0;
        for(int i=0;i<=n;i++){
            if(deque.isEmpty()||a[deque.peekLast()]<=a[i]){//单减栈，当栈是空或者栈顶元素小于等于时入栈
                deque.addLast(i);
            }else {
                while(!deque.isEmpty()&&a[deque.peekLast()]>a[i]){// 寻找到第一个小于的数
                    t = deque.pollLast();//左侧等于的数
                    ans = Math.max(ans,(i-t)*a[t]);
                }
                deque.addLast(t);//延伸到最左侧时的数
                a[t] = a[i];//有效高度
            }
        }

        System.out.println(ans);
    }

    private static int  poj3494(int []a ,int n){
        Deque<Integer> deque = new ArrayDeque<Integer>();
        int ans = 0,t =0;

        for(int i=0;i<=n;i++){
            if(deque.isEmpty()||a[deque.peekLast()]<= a[i]){
                deque.addLast(i);
            }else {
                while(!deque.isEmpty()&&a[deque.peekLast()]>a[i]){
                    t = deque.pollLast();
                    ans = Math.max(ans,(i-t)*a[t]);
                }
                deque.addLast(t);
                a[t] = a[i];
            }
        }

        return  ans;
    }


}
