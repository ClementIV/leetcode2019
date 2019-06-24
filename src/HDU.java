import java.util.Arrays;
import java.util.Scanner;

public class HDU {

    /**
     * hdu2602 基本01背包
     * @param args
     */
    public static  void hdu2602(String[] args){
        Scanner in = new Scanner(System.in);
        int t = in.nextInt();
        int n,m;
        while(t-- >0){
            n = in.nextInt();m = in.nextInt();
            int w[] = new int[n];
            int v[] = new int[n];
            for(int i=0;i<n;i++) w[i] = in.nextInt();
            for(int i=0;i<n;i++) v[i] = in.nextInt();
            System.out.println(hdu2602(w,v,n,m));
        }

    }

    private static int hdu2602(int[] w,int[] v,int n,int m){
        int[] dp = new int[m+1];
        for(int i=0;i<n;i++){
            for(int j = m;j>=v[i];j--){
                dp[j] =Math.max(dp[j-v[i]]+w[i],dp[j]);
            }
        }
        return dp[m];
    }


    /**
     * hdu 2546 01背包变种
     * @param args
     */
    public static void hdu2546(String[] args){
        Scanner in = new Scanner(System.in);
        int n,m;

        while( (n = in.nextInt()) > 0){
            int[] v = new int[n];
            for(int i=0;i<n;i++) v[i] = in.nextInt();
            m = in.nextInt();
            if(m<5) continue;

            m -=5;
            Arrays.sort(v);
            int t=hdu2546(v,n-1,m);
            System.out.println(m - t+5-v[n-1]);
        }
    }
    private static int hdu2546(int[] v,int n,int m){
        int dp[] = new int[m+1];
        for(int i=0;i<n;i++){
            for(int j=m;j>=v[i];j--){
                dp[j] = Math.max(dp[j-v[i]]+v[i],dp[j]);
            }
        }

        return  dp[m];
    }


    /**
     * hdu1114 完全背包
     */
    static int INF = Integer.MAX_VALUE;
    public static void  hdu1114(String[] args){
        Scanner in = new Scanner(System.in);
        int t ,min,max;
        t = in.nextInt();
        while(t-->0){
            min = in.nextInt();
            max = in.nextInt();
            int m = max -min;

            int n = in.nextInt();
            int v,w;
            long[] dp = new long[m+1];
            for(int i=1;i<=m;i++) dp[i] = INF;

            for(int i=0;i<n;i++){
                w = in.nextInt();
                v = in.nextInt();
                for(int j=v;j<=m;j++){
                    dp[j] = Math.min(dp[j-v]+w,dp[j]);
                }
            }

            if(INF == dp[m]) System.out.println("This is impossible.");
            else System.out.println("The minimum amount of money in the piggy-bank is "+dp[m]+".");
        }

    }

    /**
            hdu1059 多重背包
    **/

//import java.util.Scanner;
//
//    public class Main {
//
//        static int MAX = 120010;
//        static int[] dp= new int[MAX];
//        static int[] v = new int[7];
//        private static void completePack(int val,int m){
//            for(int i=val;i<=m;i++){
//                dp[i] = dp[i-val]|dp[i];
//            }
//        }
//
//        private static void zeroOnePack(int val,int m){
//            for(int i=m;i>=val;i--){
//                dp[i] = dp[i-val]|dp[i];
//            }
//        }
//
//        private static void mutilPack(int val,int num,int m){
//            if(val*num>=m) {
//                completePack(val,m);
//            } else {
//                int k =1;
//                while(k<num){
//                    zeroOnePack(val*k,m);
//                    num -=k;
//                    k <<=1;
//                }
//                zeroOnePack(val*num,m);
//            }
//        }
//        public static void  main(String[] args){
//            Scanner in = new Scanner(System.in);
//            int T= 0;
//            while(in.hasNext()){
//                boolean flag = true;
//                for(int i=1;i<7;i++) {
//                    v[i] = in.nextInt();
//                    if(v[i]!=0) {flag = false;}
//                }
//                T++;
//
//                if(flag) return;
//
//                int m=0;
//                for(int i=1;i<7;i++){
//                    v[i]%=30;//优化代码 ！！！ 1--6的公倍数考虑
//                    m+= v[i]*i;
//                }
//
//                if(m%2==1){
//                    System.out.print("Collection #"+T+":\nCan't be divided.\n\n");
//                }else {
//                    dp[0] = 1;
//                    for(int i=1;i<=m+1;i++) dp[i]=0;
//
//                    m=m/2;
//                    for(int i = 1; i < 7; i++)
//                        mutilPack(i, v[i],m);
//                    if(dp[m]==0){
//                        System.out.print("Collection #"+T+":\nCan't be divided.\n\n");
//                    }else {
//                        System.out.print("Collection #"+T+":\nCan be divided.\n\n");
//                    }
//                }
//
//
//            }
//
//        }
//
//
//
//    }

    /**
     * hdu2844 多重背包，混合背包
     */
    /**


    static int MAX = 100010;
    static int[] dp = new int[MAX];

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNext()) {
            int n = in.nextInt(), m = in.nextInt();
            if (n == 0 && m == 0) return;
            int val[] = new int[n];
            int num[] = new int[n];
            for (int i = 0; i < n; i++) val[i] = in.nextInt();
            for (int i = 0; i < n; i++) num[i] = in.nextInt();
            for (int i = 1; i <= m + 1; i++) dp[i] = 0;
            dp[0] = 1;
            for (int i = 0; i < n; i++) {
                int k = 1;
                while (k < num[i]) {
                    for (int j = m; j >= k * val[i]; j--) {
                        dp[j] = dp[j - k * val[i]] | dp[j];
                    }
                    num[i] -= k;
                    k <<= 1;
                }
                for (int j = m; j >= num[i] * val[i]; j--) {
                    dp[j] = dp[j - num[i] * val[i]] | dp[j];
                }

            }
            int sum = 0;
            for (int i = 1; i <= m; i++) {
                if (dp[i] == 1) sum++;
            }
            System.out.println(sum);
        }

    }


     */
    /**
    private void hdu2844(){
        static int MAX = 100010;
        static int[] dp= new int[MAX];
        private static void completePack(int val,int m){
            for(int i=val;i<=m;i++){
                dp[i] = dp[i-val]|dp[i];
            }
        }

        private static void zeroOnePack(int val,int m){
            for(int i=m;i>=val;i--){
                dp[i] = dp[i-val]|dp[i];
            }
        }

        private static void mutilPack(int val,int num,int m){
            if(val*num>=m) {
                completePack(val,m);
            } else {
                int k =1;
                while(k<num){
                    zeroOnePack(val*k,m);
                    num -=k;
                    k <<=1;
                }
                zeroOnePack(val*num,m);
            }
        }
        public static void  main(String[] args){
            Scanner in = new Scanner(System.in);
            while(in.hasNext()){
                int n = in.nextInt(),m = in.nextInt();
                if(n==0&&m==0) return;
                int[] v = new int[n];
                int[] num = new int[n];
                for(int i=0;i<n;i++) v[i] = in.nextInt();
                for(int i=0;i<n;i++) num[i] = in.nextInt();
                for(int i=1;i<=m+1;i++) dp[i] = 0;
                dp[0] = 1;
                for(int i=0;i<n;i++) mutilPack(v[i],num[i],m);

                int sum=0;
                for(int i =m;i>=1;i--) {
                    if(dp[i]==1){
                        sum++;
                    }
                }
                System.out.println(sum);
            }

        }
    }
    */

    /**
     * hdu2159多重背包
     * @param args
     */
    public static void  hdu2159(String[] args){
        Scanner in = new Scanner(System.in);
        while(in.hasNext()){
            int n = in.nextInt(),m = in.nextInt();
            int k = in.nextInt(),s = in.nextInt();
            int dp[][] = new int[m+1][s+1];
            for(int i=0;i<k;i++) {
                int w=in.nextInt(),v = in.nextInt();
                for(int j =v;j<=m;j++){
                    for(int z=1;z<=s;z++){
                        dp[j][z] = Math.max(dp[j-v][z-1]+w,dp[j][z]);
                    }
                }
            }
            int j=1;
            for(j=1;j<=m;j++){
                if(dp[j][s]>=n){
                    System.out.println(m-j);
                    break;
                }
            }
            if(j>m) System.out.println(-1);

        }

    }

    /**
     * hdu 2159
     */
    /**
     *
     *
     *

    static int MAX = 110;
    static int[][] dp = new int[MAX][MAX];

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNext()) {
            int n = in.nextInt(), m = in.nextInt();
            int k = in.nextInt(), s = in.nextInt();
            for(int i=1;i<=m;i++){
                for(int j =1;j<=s;j++){
                    dp[i][j] = 0;
                }
            }
            for(int t=0;t<k;t++){
                int val = in.nextInt();
                int w = in.nextInt();
                for(int i=w;i<=m;i++){
                    for(int j=1;j<=s;j++){
                        dp[i][j] = Math.max(dp[i-w][j-1]+val,dp[i][j]);
                    }
                }
            }

            int i=1;
            for(i=1;i<=m;i++){
                if(dp[i][s]>=n) {
                    System.out.println(m-i);
                    break;
                }
            }
            if(i>m) System.out.println(-1);
        }

    }
    **/


    static int MAX = 110;
    static int[][] dp = new int[MAX][MAX];

    /**

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNext()) {
            int n = in.nextInt();//n组物品
            int T= in.nextInt();//总计背包容量T,T分钟
            for(int i=0;i<=n;i++)
                for(int j=0;j<=T;j++)
                    dp[i][j] = 0;


            for(int i=1;i<=n;i++){//枚举分组

                int m= in.nextInt(),s = in.nextInt();
                int[] c =new int[m];//每组数量
                int[] g = new int[m];
                for(int j=0;j<m;j++) {
                    c[j] = in.nextInt();
                    g[j] = in.nextInt();
                }
                if(s==2){//可以选0个或者多个 基本01背包问题
                    for(int j=0;j<MAX;j++){
                        dp[i][j] = dp[i-1][j];
                    }
                    for(int k=0;k<m;k++){
                        for(int j=T;j>=c[k];j--){//枚举体积
                            dp[i][j] = Math.max(dp[i][j-c[k]]+g[k],dp[i][j]);
                        }
                    }

                }else if(s==1){
                    for(int j=0;j<MAX;j++){
                        dp[i][j] = dp[i-1][j];
                    }

                    for(int k=0;k<m;k++) {
                        for(int j=T;j>=c[k];j--){//至多取1，分组01背包
                            dp[i][j] = Math.max(dp[i-1][j-c[k]]+g[k],dp[i][j]);
                        }
                    }
                }else {

                    //至少选一件
                    for(int j=0;j<MAX;j++){
                        dp[i][j] = -10000;
                    }
                    //选择1件或者件
                    for(int k=0;k<m;k++){
                        for(int j=T;j>=c[k];j--){//枚举体积
                            int temp = Math.max(dp[i][j],dp[i-1][j-c[k]]+g[k]);
                            dp[i][j] = Math.max(dp[i][j],dp[i][j-c[k]]+g[k]);
                            dp[i][j] = Math.max(dp[i][j],temp);
                        }
                    }
                }

            }
            dp[n][T] = Math.max(dp[n][T],-1);

            System.out.println(dp[n][T]);
        }

    }
     */

    /**
     * hdu1712 01分组背包
     */
    /**
    static int MAX = 110;
    static int[][] dp = new int[MAX][MAX];

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNext()) {
            int n = in.nextInt();//n组物品
            int m= in.nextInt();//总计背包容量T,T分钟
            if(n==0&&m==0) return;

            for(int i=0;i<=n;i++){
                for(int j=0;j<m;j++){
                    dp[i][j] = 0;
                }
            }

            for(int i=1;i<=n;i++){
                for(int j=0;j<MAX;j++){
                    dp[i][j] = dp[i-1][j];
                }
                for(int k=1;k<=m;k++) {
                    int v= in.nextInt();
                    for(int j=m;j>=k;j--){//至多取1，分组01背包
                        dp[i][j] = Math.max(dp[i-1][j-k]+v,dp[i][j]);
                    }
                }
            }

            System.out.println(dp[n][m]);

        }

    }*/

    /**
     * HDU3033 1711
     */

//import java.util.*;
//
//    public class Main {
//        static int MAXN = 110,MAXK=12,MAXM=10010;
//        static int a[] = new int[MAXN];
//        static int b[] = new int[MAXN];
//        static int c[] = new int[MAXN];
//        static int dp[][] = new int[MAXK][MAXM];
//        public static void main(String[] args){
//            Scanner in = new Scanner(System.in);
//            while(in.hasNext()){
//                int n = in.nextInt(),m = in.nextInt(),s = in.nextInt();
//                for(int i=1;i<=s;i++){
//                    for(int j=1;j<=m;j++){
//                        dp[i][j] = -999999;
//                    }
//                }
//                for(int i=0;i<MAXN;i++) {
//                    a[i] = 0;
//                    b[i] = 0;
//                    c[i] = 0;
//                }
//                for(int i=1;i<=n;i++) {
//                    a[i] = in.nextInt();
//                    b[i] = in.nextInt();
//                    c[i] = in.nextInt();
//                }
//                for(int i=1;i<=s;i++){
//                    for(int j=1;j<=n;j++){
//                        for(int k=m;k>=1;k--){
//                            if(k>=b[j]&&a[j]==i){
//                                int temp = Math.max(dp[i][k],dp[i-1][k-b[j]]+c[j]);
//                                dp[i][k] = Math.max(dp[i][k],dp[i][k-b[j]]+c[j]);
//                                dp[i][k] = Math.max(dp[i][k],temp);
//                            }
//                        }
//                    }
//                }
//                if(dp[s][m]<0) System.out.println("Impossible");
//                else System.out.println(dp[s][m]);
//            }
//        }
//
//
//    }

    /**
     * hdu 1711
     * @param args
     */
    public static void hdu1711(String[] args){
        Scanner in = new Scanner(System.in);
        int t = in.nextInt();
        while(t-->0){
            int n = in.nextInt(),m = in.nextInt();
            int []a = new int[n];
            int []b = new int[m];
            for(int i=0;i<n;i++) a[i] = in.nextInt();
            for(int j=0;j<m;j++) b[j] = in.nextInt();

            int[] f = getNext(b);
            int ans =-1;
            int i=0,j=0;
            while(i<n){
                if(b[j]==a[i]){
                    i++;j++;
                    if(j==m){
                        ans = i-j;
                        break;
                    }
                }else{
                    if(j==0) i++;
                    else j = f[j-1]+1;
                }
            }
            if(ans==-1)System.out.println(ans);
            else System.out.println(ans+1);


        }
    }
    private  static int[] getNext(int[] b){
        int n = b.length;
        int f[] = new int[n];
        f[0] = -1;
        for(int i=1;i<n;i++){
            int j = f[i-1];
            while(j>=0&&b[j+1]!=b[i]){
                j = f[j];
            }
            if(b[j+1]==b[i]){
                f[i] = j+1;
            }else {
                f[i] = -1;
            }
        }

        return f;

    }

    /**
     * POJ 3206 多重背包
     */
}
