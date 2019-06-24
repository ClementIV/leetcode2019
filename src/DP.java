import java.util.*;

public class DP {
    public int memosizeCutRod(int[] p,int n){
        int [] r = new int[n+1];
        for(int i=0;i<=n;i++){r[i]= -999; }
        return memosizeCutRodAux( p, n,r);
    }

    private int  memosizeCutRodAux(int[] p, int n, int[] r) {
        if(r[n]>=0) return r[n];
        int q= 0;
        if(n>0) q = -999;
        for(int i=1;i<=n;i++) q =Math.max(q,p[i]+memosizeCutRodAux(p,n-i,r));
        r[n] = q;
        return q;
    }

    public int[][] BottomToUpCutRod(int[]p,int n,int c){
        int[] r = new int[n+1];
        int[] s = new int[n+1];
        for(int i=1;i<=n;i++){
          int  q=p[i];
          s[i] = i;
          for(int j=1;j<=i;j++){
             if(q<p[j]+r[i-j]-c){
                 q = p[j]+r[i-j]-c;
                 s[i]=j;
             }
          }
          r[i] = q;
        }
        return new int[][]{r,s};
    }


    public static void main(String[] args){
        String s = "ABAC",t = "CAB";
        int[][] ans = new DP().LCS(s,t);

        System.out.println(ans[4][3]);
        new DP().printLCS(ans,s,4,3);
        System.out.println();


    }

    public int pack(int[] p,int n){
        int dp[][] = new int[p.length][n+1];
        int count[] = new int[p.length];
        for(int i=1;i<p.length;i++){
            for(int v=i;v<=n;v++){
                int k=0;
                for(;k*i<=v;k++)
                    dp[i][v] = Math.max(dp[i-1][v],dp[i-1][v-k*i]+k*p[i]);
                count[i] = k-1;
            }
        }
        int ans= dp[p.length-1][n];
        int v =n;
        for(int i=p.length-1;i>=1;i--){
            if(v>0&count[i]>0&&ans==p[i]*count[i]+dp[i-1][v-count[i]*i]){
                System.out.println(i+" "+count[i]);
                ans -=p[i]*count[i];
                v -= count[i]*i;
            }
        }
        return dp[p.length-1][n];
    }

    public int[][] LCS(String s,String t){
        int m = s.length(),n = t.length();
        int[][] c=new int[m+1][n+1];
        for(int i=1;i<=m;i++){
            for(int j=1;j<=n;j++){
                if(s.charAt(i-1)==t.charAt(j-1)){
                    c[i][j] = c[i-1][j-1]+1;
                }else{
                    c[i][j] = Math.max(c[i-1][j],c[i][j-1]);
                }
            }
        }
        return c;
    }
    public void printLCS(int[][] b,String s,int i,int j){
        if(i==0||j==0) return;
        if(b[i][j]==0){
            printLCS(b,s,i-1,j-1);
            System.out.print(s.charAt(i-1));
        }else if(b[i][j]==1){
            printLCS(b,s,i-1,j);
        }else printLCS(b,s,i,j-1);
    }
    public String longestPalindrome(String s) {
        int n = s.length();
        int dp[][] = new int[n][n];
        for(int j=0;j<n;j++){
            dp[j][j]=1;
            for(int i=j-1;i>=0;i--){
                if(s.charAt(i)==s.charAt(j))
                    dp[i][j]=dp[i+1][j-1]+2;
                else
                    dp[i][j]=Math.max(dp[i+1][j],dp[i][j-1]);
            }
        }

        return getMaxString(dp,s,0,n-1);
    }
    public String getMaxString(int[][]dp,String s,int i,int j){
        if(i==j) return s.charAt(i)+"";
        if(i+1==j&&s.charAt(i)==s.charAt(j)) return s.substring(i,j+1);
        String ans ="";
        if(dp[i][j]== dp[i][j-1]){
            ans = getMaxString(dp,s,i,j-1);
        }else if(dp[i][j]== dp[i+1][j]){
            ans = getMaxString(dp,s,i+1,j);
        }else if(dp[i][j] ==dp[i+1][j-1]+2){
            ans = s.charAt(i)+getMaxString(dp,s,i+1,j-1)+s.charAt(j);
        }

        return  ans;

    }


//    static int MAXN = 110;
//    static int MAXM = 10010;
//    static int MAXK = 12;
//    static int[][] dp = new int[MAXK][MAXM];
//    static Map<Integer, List<Integer>> val = new HashMap<>();
//    static Map<Integer, List<Integer>> weight = new HashMap<>();
//    public static void main(String[] args) {
//        Scanner in = new Scanner(System.in);
//        while (in.hasNext()) {
//            int n = in.nextInt(),m = in.nextInt(),k= in.nextInt();
//
//            for(int i=1;i<=k;i++){
//                for(int j=1;j<=m;j++){
//                    dp[i][j] = -999999;
//                }
//            }
//            val.clear();
//            weight.clear();
//
//
//            for(int i=1;i<=n;i++){
//                int a = in.nextInt(),b = in.nextInt(),c = in.nextInt();
//                if(val.containsKey(a)){
//                    val.get(a).add(c);
//                    val.put(a,val.get(a));
//                    weight.get(a).add(b);
//                    weight.put(a,weight.get(a));
//                }else {
//                    List<Integer> temp = new ArrayList<>();temp.add(c);
//                    val.put(a,temp);
//                    List<Integer> temp1 = new ArrayList<>();temp1.add(b);
//                    weight.put(a,temp1);
//                }
//
//            }
//
//            for(int i=1;i<=k;i++){
//
//                List<Integer> sn = weight.get(i);
//                for(int j = 0;j<sn.size();j++) {
//                    int v = val.get(i).get(j);
//                    int w = sn.get(j);
//                    for(int p=m;p>=w;p--){
//                        int temp = Math.max(dp[i][p],dp[i-1][p-w]+v);
//                        dp[i][p] = Math.max(dp[i][p],dp[i][p-w]+v);
//                        dp[i][p] = Math.max(dp[i][p],temp);
//                    }
//                }
//            }
//            if(dp[k][m]<0)
//                System.out.println("Impossible");
//            else
//                System.out.println(dp[k][m]);
//        }
//
//
//    }


}
