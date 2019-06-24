import sun.java2d.cmm.lcms.LcmsServiceProvider;

import javax.xml.stream.events.StartDocument;
import java.util.*;

public class COM {
    public int twoCitySchedCost(int[][] costs) {
        Arrays.sort(costs, (o1, o2) -> (o2[1] - o2[0]) - (o1[1] - o1[0]));
        int max = 0;
        for (int i = 0; i < costs.length / 2; i++) {
            max += costs[i][0];
        }
        for (int i = costs.length / 2; i < costs.length; i++)
            max += costs[i][1];
        return max;
    }

    public int[][] allCellsDistOrder(int R, int C, int r0, int c0) {
        int[][] arr = new int[R * C][2];
        int index = 0;
        for (int i = 0; i < R; i++)
            for (int j = 0; j < C; j++) {
                arr[index][0] = i;
                arr[index++][1] = j;
            }
        Arrays.sort(arr, (o1, o2) -> Math.abs(o1[1] - c0) + Math.abs(o1[0] - r0) - (Math.abs(o2[1] - c0) + Math.abs(o2[0] - r0)));
        return arr;
    }

    //lintcode 943 区间和

    public  int[] getNext(String target){
        int n = target.length();
        int f[] = new int[n];
        f[0] = -1;
        for(int i=1;i<n;i++){
            int j= f[i-1];
            while(j>=0&&target.charAt(j+1)!=target.charAt(i)){
                j = f[j];
            }
            if(target.charAt(i)==target.charAt(j+1)) f[i] = j+1;
            else f[i] = -1;
        }

        return f;
    }

    /**
     * 1223. 环绕字符串中的唯一子串
     * 中文English
     * 字符串s是由字符串"abcdefghijklmnopqrstuvwxyz"无限重复环绕形成的，所以s看起来像是这样："...zabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcd...."。
     *
     * 现在有另一个字符串p。你的任务是找出p中有多少互不相同的非空子串出现在s中。确切的说，你得到字符串p作为输入，你需要输出关于p非空子串的数目，这些子串互不相同，并且能在s中被找到。
     *
     * 样例
     * 样例1：
     *
     * 输入："a"
     * 输出：1
     * 说明：字符串"a"的所有非空子串中，只有"a"出现在字符串s中。
     * 样例2：
     *
     * 输入："cac"
     * 输出：2
     * 说明：字符串"cac"的所有非空子串中，"a"和"c"出现在字符串s中。
     * 样例3：
     *
     * 输入："zab"
     * 输出：6
     * 说明：有六个"zab"的非空子串"z"、"a"、"b"、"za"、"ab"、"zab"出现在字符串s中。
     * 注意事项
     * p只包含小写字母，并且它的长度可能会超过10000。
     * @param p
     * @return
     */
    public int findSubstringInWraproundString(String p) {
        int[] dp = new int[26];
        int n = p.length(),pos = 0;
        for(int i=0;i<n;i++){
            if(i>0&&(p.charAt(i)-p.charAt(i-1)==1 || p.charAt(i)=='a'&&p.charAt(i-1)=='z')){
                pos++;
            }else pos =1;
            int c =p.charAt(i)-'a';
            dp[c] = Math.max(dp[c],pos);
        }

        int ans =0;
        for(int i=0;i<26;i++) ans += dp[i];

        return ans;
    }
    public static void main(String[] args){
        //int[] f = getNext("abcabcabcabc");
        //for(int i=0;i<f.length;i++) System.out.print(f[i]+" ");
        System.out.println(9%-4);
    }
    public int[][] kClosest(int[][] points, int K) {
        Arrays.sort(points, (o1, o2) -> o1[0]*o1[0]+o1[1]*o1[1] -(o2[0]*o2[0]+o2[1]*o2[1]));
        int[][] ans = new int[K][2];
        for(int i=0;i<K;i++) ans[i]=points[i];

        return ans;
    }
    public int largestPerimeter(int[] A) {
        Arrays.sort(A);

        for(int i=A.length-1;i>=2;i--){
            int c = A[i];
            if(A[i-1]+A[i-2]>c) return A[i]+A[i-1]+A[i-2];
        }
        return 0;
    }

    public int subarraysDivByK(int[] A, int K) {
        int n = A.length;
        int f[] = new int[n+1];
        for(int i=1;i<=n;i++) f[i] = f[i-1]+A[i-1];
        int modK[] = new int[K];
        for(int i=0;i<=n;i++){ modK[(f[i]%K+K)%K]++; }

        int ans =0;
        for(int v:modK) ans +=v*(v-1)/2;

        return ans;
    }
    //372超级次方
    public int superPow(int a ,int[] b){
        int c = 1337;
        int exp = 0;
        int phi = euler(c);
        //同余
        for(int i=0;i<b.length;i++) exp = (exp*10+b[i])%1140;

        return qucikPow(a%1337,exp,1337);
    }
    //快速幂计算
    public int qucikPow(int a,int b,int c){
        int ans =1;
        while(b>0){
            if((b&1)==1){
                ans = (ans*a)%c;
            }
            b >>=1;
            a = (a*a)%c;
        }
        return ans;
    }
    //欧拉函数
    public int euler(int n){
        int ret= n;
        for(int i=2;i*i<n;i++){
            if(n%i==0){//n的质因数
                ret = ret/n*(n-1); //欧拉函数公式
                while(n%i==0){//去掉质因数i
                    n/=i;
                }
            }
        }
        if(n>1){//n本来就是质数 f(n) = n-1;
            ret = ret/n*(n-1);
        }
        return ret;
    }
    public boolean isRobotBounded(String instructions) {
        int flag =0;int[] p = new int[2];
        for(int k=0;k<4;k++){
            for(int i=0;i<instructions.length();i++){
              char c= instructions.charAt(i);
              if(c=='G'){
                 go(p,flag);
              }else if(c=='L') flag=(flag+1)%4;
              else flag=(flag-1 +4)%4;
            }
        }

        if(p[0]==0&&p[1]==0) return true;

        return false;
    }
    private void go(int[] p,int flag){
        if(flag==0) p[1] ++;
        else if(flag==1) p[0] --;
        else if(flag==2) p[1] --;
        else p[0]++;
    }

    public int[] gardenNoAdj(int N, int[][] paths) {
        int[][] graph = new int[N+1][4];
        for(int[] e :paths){
            for(int i=0;i<4;i++){
                if(graph[e[0]][i]==0){
                    graph[e[0]][i]= e[1];
                    break;
                }
            }
            for(int i=0;i<4;i++){
                if(graph[e[1]][i]==0){
                    graph[e[1]][i]= e[0];
                    break;
                }
            }
        }


        int []ans = new int[N+1];
        dfs(graph,ans,N);

        return Arrays.copyOfRange(ans,1,N+1);
    }

//    public  int[] getNext(String target){
//        int n = target.length();
//        int f[] = new int[n];
//        f[0] = -1;
//        for(int i=1;i<n;i++){
//            int j= f[i-1];
//            while(j>=0&&target.charAt(j+1)!=target.charAt(i)){
//                j = f[j];
//            }
//            if(target.charAt(i)==target.charAt(j+1)) f[i] = j+1;
//            else f[i] = -1;
//        }
//
//        return f;
//    }

    public String longestDupSubstring(String s) {
        int k =2;
        String temp = helper(s,k);
        if(temp.length()<=0) return "";
        int f[] = getNext(temp);
        for(int i=1;i<f.length;i++) if(f[i]<0) return "";
        return temp.substring(0,f[f.length-1]);
    }
    public String helper(String s,int k){
        if(k>s.length()||s.length()==0){
            return "";
        }
        int count[] = new int[26];
        for(int i=0;i<s.length();i++){
            count[s.charAt(i)-'a']++;
        }
        String max = "";
        int maxCount = -1;
        int start =0;
        for(int i=0;i<s.length();i++){
            if(count[s.charAt(i)-'a']<k){
                String temp =helper(s.substring(start,i),k);
                if(temp.length()>max.length()){
                    max = temp;
                    maxCount = temp.length();
                }

                start = i+1;
            }
        }
        if(start>0&&start<s.length()){
            String temp =helper(s.substring(start,s.length()),k);
            if(temp.length()>max.length()){
                max = temp;
                maxCount = temp.length();
            }
        }
        return maxCount==-1? s:max;
    }
    public int longestSubstring(String s, int k) {
        if(k>s.length()||s.length()==0){
            return 0;
        }
        int count[] = new int[26];
        for(int i=0;i<s.length();i++){
            count[s.charAt(i)-'a']++;
        }
        int max = -1,start =0;
        for(int i=0;i<s.length();i++){
            if(count[s.charAt(i)-'a']<k){
                max = Math.max(longestSubstring(s.substring(start,i),k),max);
                start = i+1;
            }
        }
        if(start>0&&start<s.length()){
            max = Math.max(longestSubstring(s.substring(start,s.length()),k),max);
        }
        return max==-1?s.length():max;
    }
    private void dfs(int[][] graph, int[] ans,int n) {
        for(int i=1;i<=n;i++){
            boolean v[] = new boolean[5];
            for(int j=0;j<4;j++){
                if(graph[i][j]>0&&ans[graph[i][j]]>0){v[ans[graph[i][j]]] = true; }
            }
            for(int j =1;j<5;j++){
                if(!v[j]){
                    ans[i] = j;
                    break;
                }

            }
        }
    }
    private int getMax(int[] a, int i, int i1){
        int max= a[i];
        while(i<i1){
            max = Math.max(max,a[i]);
            i++;
        }
        return max;
    }
    public int maxSumAfterPartitioning(int[] A, int K) {
        int n = A.length;
        int dp[] = new int[n+1];
        for(int i=1;i<=n;i++){
            dp[i] = dp[i-1]+A[i-1];
            for(int j=1;j<i&&j<K;j++){
                int m = getMax(A,i-j-1,i);
                dp[i] = Math.max(dp[i],dp[i-j-1]+m*(j+1));
            }
        }
        return dp[n];
    }

    public boolean isMonotonic(int[] A) {
        if(A.length<=1) return true;
        int f[] = new int[A.length];
        for(int i=1;i<A.length;i++){
            f[i] = A[i]-A[i-1];
        }
        int countA =0 ,countB = 0;
        for(int i=1;i<A.length;i++){
            if(f[i]>0) countA++;
            else if(f[i]<0) countB++;
        }
        if(countA>0&&countB>0) return false;
        return true;
    }
    public TreeNode increasingBST(TreeNode root) {
        return  increasingBST(root,null);
    }
    public TreeNode increasingBST(TreeNode root,TreeNode next){
        if(root==null) return null;

        if(root.right==null) root.right = next;
        else root.right = increasingBST(root.right,next);

        if(root.left==null) return root;
        next = root;
        TreeNode ans = increasingBST(root.left,next);
        root.left = null;

        return ans;
    }

    public int subarrayBitwiseORs(int[] A) {
        Set<Integer> set = new HashSet<>();//最后的结果
        Set<Integer> pre = new HashSet<>();//上一组的或操作结果集合
        //dp[i][j]代表从A[i] | A[i+1] | ... | A[j]
        //dp[][j - 1]代表最后一个元素在原数组中的索引为j - 1的子数组
        //dp[i][j]是dp[i][j - 1] | A[j]，但是dp[i - k][j]和dp[i][j]可能有重复，只需要计算其中一个即可
        //这里我们可以使用set记录每一组dp[][j]的结果，每次只要A[i]*set(k)再加上一个A[i]即可
        for(int i = 0; i < A.length; i++) {
            Set<Integer> cur = new HashSet<>();//当前组的或操作结果集合
            set.add(A[i]);
            cur.add(A[i]);
            for(int num : pre) {
                int tmp = (A[i] | num);
                set.add(tmp);
                cur.add(tmp);
            }
            pre = cur;
        }
        return set.size();
    }
    public String orderlyQueue(String S, int K) {

       if(K>1) {
           char[] chars = S.toCharArray();
           Arrays.sort(chars);
           return String.valueOf(chars);
       }
        String minS=S;
        for (int i=0;i<S.length();++i){
            S=S.substring(1)+S.substring(0,1);
            if(S.compareTo(minS)<0) minS = S;
        }
        return minS;

    }
    public String removeOuterParentheses(String S){
        Deque<Character> deque = new ArrayDeque<Character>();
        int count=0;
        String ans = "";
        for(int i=0;i<S.length();i++){
            char c = S.charAt(i);
            if(c=='('){
                deque.addLast(c);
                count ++;
            }
            else{
                deque.addLast(c);
                count --;
                if(count==0){
                    deque.pollFirst();
                    deque.pollLast();
                    while(!deque.isEmpty())
                        ans+=deque.pollFirst();
                }
            }
        }
        while(!deque.isEmpty())
            ans+=deque.pollFirst();

        return ans;
    }

    public int sumRootToLeaf(TreeNode root) {
        int c = 1000000007;
        return sumRootToLeaf(root,0,c);
    }
    public int sumRootToLeaf(TreeNode root,int n,int c){
        if(root==null) return 0;
        if(root.left==null&& root.right==null) return  (n%c*2+root.val)%c;

        n = (n%c*2+root.val)%c;
        return (sumRootToLeaf(root.left,n,c)+sumRootToLeaf(root.right,n,c))%c;
    }
    public List<Boolean> camelMatch(String[] queries, String pattern) {
        List<String> pas = new ArrayList<>();
        int start =-1;

        for(int i=0;i<pattern.length();i++){
            if(pattern.charAt(i)>='A'&&pattern.charAt(i)<='Z'){
                if(start>=0) pas.add(pattern.substring(start,i));
                start = i;
            }
        }
        if(start>=0) pas.add(pattern.substring(start));
        List<Boolean> ans = new ArrayList<>();

        if(pas.size()==0) {
            for(int i=0;i<queries.length;i++){
                ans.add(Boolean.FALSE);
            }
            return ans;
        }


        for(int i=0;i<queries.length;i++){
            char chars[] = new char[pas.size()];
            List<String> temp = new ArrayList<>();

            int count =0,flag= 0,st = -1;
            for(int j =0;j<queries[i].length();j++){
                if(queries[i].charAt(j)>='A'&&queries[i].charAt(j)<='Z'){
                    if(count>=pas.size()|| pas.get(count).charAt(0)!=queries[i].charAt(j)){
                        ans.add(Boolean.FALSE);
                        flag =1;
                        break;
                    }else {
                        count ++;
                        if(st>=0) temp.add(queries[i].substring(st,j));
                        st = j;
                    }
                }

            }
            if(st>=0) temp.add(queries[i].substring(st));
            if(flag==1) continue;
            if(count<pas.size()){
                ans.add(Boolean.FALSE);
                continue;
            }
            flag = 0;
            for(int j=0;j<pas.size();j++){
                if(!containsString(temp.get(j),pas.get(j))){
                    flag=1;
                    ans.add(Boolean.FALSE);
                    break;
                }
            }
            if(flag==0)
                ans.add(Boolean.TRUE);
        }

        return  ans;
    }
    private  boolean containsString(String a, String b){
        if(a.length()<b.length()) return false;
        int[] n1 =new int[26];
        int[] n2 = new int[26];
        for(int i=1;i<a.length();i++){
            n1[a.charAt(i)-'a']++;
        }
        for(int i=1;i<b.length();i++){
            n2[b.charAt(i)-'a']++;
        }
        for(int i=0;i<26;i++)
            if(n1[i]<n2[i]) return false;
            return true;
    }
    public int maxSumTwoNoOverlap(int[] A, int L, int M) {
        int max=0,n=A.length;
        for(int i=0;i<=n;i++){
            max =Math.max(max,Math.max(maxSum(A,L,0,i)+maxSum(A,M,i,n),maxSum(A,M,0,i)+maxSum(A,L,i,n)));
        }
        return max;
    }
    public int maxSum(int []A,int len,int i,int j){
        int n = j-i;
        if(n<len) return 0;
        int max=0,sum =0;
        for(int k=i;k<i+len;k++){sum+=A[k];}
        max = sum;
        for(int k=i+len;k<j;k++){
            sum = sum-A[k-len]+A[k];
            if(sum >max) max = sum;
        }
        return  max;
    }

    public String strWithout3a3b(int A, int B) {
        StringBuilder res = new StringBuilder(A + B);
        char a = 'a', b = 'b';
        int i = A, j = B;
        if (B > A) { a = 'b'; b = 'a'; i = B; j = A; }
        while (i-- > 0) {
            res.append(a);
            if (i > j) { res.append(a); --i; }
            if (j-- > 0) res.append(b);
        }
        return res.toString();
    }

    /*
    import java.util.*;

public class Main {
    public static void main(String[] arg) {
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        int A[] = new int[n];
        for(int i=0;i<n;i++) A[i] = in.nextInt();
        System.out.println(maxScore(A));
    }
    public static int maxScore(int[] A){
        int mmax = A[0],mmax_index =0;
        int res =0;
        for(int i=1; i<A.length;i++){
            res = Math.max(res,mmax+A[i]+mmax_index - i);
            if(A[i]+i>mmax+mmax_index){
                mmax = A[i];

                mmax_index = i;
            }
        }
        return res;
    }
*/
    //138周赛
    public int heightChecker(int[] heights) {
        int[] array = Arrays.copyOf(heights,heights.length);
        Arrays.sort(array);
        int ans =0;
        for(int i=0;i<heights.length;i++){
            if(heights[i]!=array[i]){
                ans++;
            }
        }
        return ans;
    }
    public int maxSatisfied(int[] customers, int[] grumpy, int X) {
        int max = 0,start =0;
        int n = grumpy.length;
        int ans =0;
        if(X<n){
            int temp =0;
            for(int i=0;i<X;i++) {
                if(grumpy[i]==1)
                    temp+=customers[i];
            }
            max = temp;
            for(int i=X;i<n;i++){
                if(grumpy[i-X]==1){temp -= customers[i-X];}
                if(grumpy[i]==1){temp += customers[i];}
                if(temp>max){
                    max = temp;
                    start= i-X+1;
                }
            }
            for(int i=start;i<start+X;i++) grumpy[i] = 0;
        }
        for(int i=0;i<n;i++){
            if(grumpy[i]==0)
                ans+=customers[i];
        }
        return ans;
    }

    public void duplicateZeros(int[] arr) {
        int i=0,k=0;
        int[] arrcopy = Arrays.copyOf(arr,arr.length);
        for(i=0,k=0;i<arr.length;i++,k++){
            if(arrcopy[k]==0){
                arr[i] =0;
                if(i<arr.length-1){
                    arr[++i] =0;
                }
            }else {
                arr[i] = arrcopy[k];
            }
        }
    }

    public int largestValsFromLabels(int[] values, int[] labels, int num_wanted, int use_limit) {
        int n = values.length;
        List<Integer>v =getItem(values,labels,use_limit);

        int dp[] = new int[num_wanted+1];
        for(int i=0;i<v.size();i++){
            for(int j=num_wanted;j>=1;j--){
                dp[j] = Math.max(dp[j-1]+v.get(i),dp[j]);
            }
        }

        return dp[num_wanted];
    }

    private List<Integer> getItem(int[] values, int[] labels,int use_limit){
        int n = values.length;
        List<Integer> res = new ArrayList<>();
        int[][] ans = new int[n][2];
        for(int i=0;i<n;i++){
            ans[i][0] = values[i];
            ans[i][1] = labels[i];
        }
        Arrays.sort(ans, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if(o1[1]==o2[1]) {
                    return o2[0] - o1[0];
                }else {
                    return o1[1] - o2[1];
                }
            }
        });

        int count =0;
        for(int i=0;i<n;i++){
            if(i>0&&ans[i][1]==ans[i-1][1]&&count>=use_limit){
                continue;
            }else if(i>0&&ans[i][1]!=ans[i-1][1]){
                count=0;
            }
            res.add(ans[i][0]);
            count++;
        }
        return res;
    }

    public String shortestCommonSupersequence(String str1, String str2) {
        int n =str1.length(),m = str2.length();
        int [][] ans = LCS(str1,str2);
        int[][]b = LCS(str1,str2);

        return printLCS(b,str1,str2,n,m);
    }

    public int[][] LCS(String s,String t){
        int m = s.length(),n = t.length();
        int[][] c=new int[m+1][n+1];
        int[][] flag = new int[m+1][n+1];

        for(int i=1;i<=m;i++) {
            c[i][0] = i;
            flag[i][0]= -1;
        }
        for(int j=1;j<=n;j++){
            c[0][j] = j;
            flag[0][j] = 1;
        }
        for(int i=1;i<=m;i++){
            for(int j=1;j<=n;j++){
                if(s.charAt(i-1)==t.charAt(j-1)){
                    c[i][j] = c[i-1][j-1]+1;
                    flag[i][j] =0;
                }else{
                    if(c[i-1][j]>c[i][j-1]){
                        c[i][j] = c[i][j-1]+1;
                        flag[i][j] = 1;
                    }else{
                        c[i][j] = c[i-1][j]+1;
                        flag[i][j] = -1;
                    }
                }
            }
        }
        return flag;
    }

    private String printLCS(int[][] b ,String s1,String s2,int i,int j){
        if(i==0||j==0) {
            if (i == 0 && j != 0) {
                return s2.substring(0,j);
            }else if(j==0&&i!=0)
                return s1.substring(0,i);

            return "";
        }
        if(b[i][j]==0) {
            return printLCS(b,s1,s2,i-1,j-1)+s1.charAt(i-1);
        }else if(b[i][j]==1){
            return printLCS(b,s1,s2,i,j-1)+s2.charAt(j-1);
        }else {
            return printLCS(b,s1,s2,i-1,j)+s1.charAt(i-1);
        }

    }

    public int sumOfDigits(int[] A) {
        int n = A.length;
        if(n==0) return 1;
        int min = A[0];
        for(int i=1;i<n;i++){
            if(A[i]<min){min = A[i];}
        }
        int sum =0;
        while(min>0){
            sum += min%10;
            min = min/10;
        }
        if((sum&1)==1) return 0;
        else return 1;
    }

    public int[][] highFive(int[][] items) {
        Arrays.sort(items, (o1, o2) -> {
            if(o1[0]==o2[0]) return o2[1] - o1[1];
            else return o1[0] -o2[0];
        });
        List<int[]> res  = new ArrayList<>();
        int count=0,sum=0;
        for(int i=0;i<items.length;i++){
            if(i>0&&count>=5&&items[i][0]==items[i-1][0]){
                continue;
            }
            if(i>0&&items[i][0]!=items[i-1][0]){
                sum =0;
                count=0;
            }
            sum +=items[i][1];
            count++;
            if(count==5){
                res.add(new int[]{items[i][0],sum/5});
            }
        }
        int[][] ans = new int[res.size()][2];
        for(int i=0;i<res.size();i++) {
            ans[i][0] = res.get(i)[0];
            ans[i][1] = res.get(i)[1];
        }
        return ans;
    }

    public String[] permute(String S) {
        int n = S.length();
        List<List<Character>> list = new ArrayList<>();
        int flag=0;//0 {} 1 {
        List<Character> temp = new ArrayList<>();
        for(int i=0;i<n;i++){
            char c=S.charAt(i);
            if(c==',') continue;
            if(flag==0){
                if(c!='{'&&c!='}'){
                    List<Character>temp1 =new ArrayList<>();
                    temp1.add(c);
                    list.add(temp1);
                }
                if(c=='{'){
                    flag=1;
                }
            }else {
                if(c!='}'){
                    temp.add(c);
                }else {
                    Collections.sort(temp);
                    list.add(new ArrayList<>(temp));
                    temp.clear();
                    flag=0;
                }
            }
        }

        getString(list,"",0,list.size());

        String[] ret = new String[ans.size()];
        for(int i=0;i<ans.size();i++) ret[i] = ans.get(i);

        return ret;
    }
    List<String> ans = new ArrayList<>();
    private void getString(List<List<Character>> list,String s,int i,int n){
        if(i==n){
            ans.add(s);
            return;
        }
        List t = list.get(i);
        for(int j=0;j<t.size();j++){
            getString(list,s+t.get(j),i+1,n);
        }
    }

}






