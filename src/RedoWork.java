import java.util.*;

public class RedoWork {
    public boolean isValidSudoku(char[][] board) {
        int[] row = new int[9], col = new int[9];
        int[][] block = new int[3][3];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] >= '0' && board[i][j] <= '9') {
                    int m = board[i][j] - '0';
                    int n = 1 << m;
                    if ((row[i] & n) != 0 || (col[j] & n) != 0 || (block[i / 3][j / 3] & n) != 0) return false;
                    row[i] |= n;
                    col[j] |= n;
                    block[i / 3][j / 3] |= n;
                }
            }
        }
        return true;
    }

    List<List<Integer>> ans = new ArrayList<>();

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        combinationSum(0, candidates, target, new ArrayList<>());
        return ans;
    }

    public void combinationSum(int i, int[] candidates, int target, List<Integer> list) {
        if (target == 0) {
            ans.add(new ArrayList<>(list));
            return;
        }
        if (i == candidates.length) return;
        if (candidates[i] > target) return;
        int cnt = 0;
        for (; target >= 0; cnt++) {
            combinationSum(i + 1, candidates, target, list);
            target -= candidates[i];
            list.add(candidates[i]);
        }
        while (cnt-- > 0) {
            list.remove(list.size() - 1);
        }

        return;

    }

    Set<List<Integer>> ans2 = new HashSet<>();

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        combinationSum2(0, candidates, target, new ArrayList<>());
        return new ArrayList<>(ans2);
    }

    public void combinationSum2(int i, int[] candidates, int target, List<Integer> list) {
        if (target == 0) {
            ans2.add(new ArrayList<>(list));
            return;
        }
        if (i == candidates.length) return;
        if (candidates[i] > target) return;

        combinationSum2(i + 1, candidates, target, list);
        list.add(candidates[i]);
        combinationSum2(i + 1, candidates, target - candidates[i], list);
        list.remove(list.size() - 1);
        return;
    }

    //桶排序 o(n)空间复杂度为o(n)
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        int f[] = new int[n];
        for (int i = 0; i < n; i++) {
            if (nums[i] > 0 && nums[i] <= n) {
                f[nums[i] - 1] = nums[i];
            }
        }
        for (int i = 0; i < n; i++) {
            if (f[i] != i + 1) return i + 1;
        }
        return n + 1;
    }

    // 空间优化
    public int firstMissingPositive2(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                int temp = nums[i];
                nums[i] = nums[nums[i] - 1];
                nums[temp - 1] = temp;
            }
        }

        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) return i + 1;
        }
        return n + 1;
    }

    public int trap(int[] height) {
        int n = height.length;
        int[] leftMax = new int[n], rightMax = new int[n];
        if (n == 0) return 0;
        leftMax[0] = height[0];
        for (int i = 1; i < n; i++) {
            leftMax[i] = Math.max(leftMax[i - 1], height[i]);
        }

        rightMax[n - 1] = height[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            rightMax[i] = Math.max(rightMax[i + 1], height[i]);
        }

        int ans = 0;
        for (int i = 0; i < n; i++) {
            ans += Math.min(leftMax[i], rightMax[i]) - height[i];
        }

        return ans;
    }

    public String multiply(String num1, String num2) {
        int m = num1.length(), n = num2.length();
        int[] a = new int[m], b = new int[n], c = new int[n + m];
        for (int i = m - 1, k = 0; i >= 0; i--, k++) a[k] = num1.charAt(i) - 48;
        for (int i = n - 1, k = 0; i >= 0; i--, k++) b[k] = num2.charAt(i) - 48;

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                c[i + j] += a[i] * b[j];
                c[i + j + 1] += c[i + j] / 10;
                c[i + j] %= 10;
            }

        int t = n + m - 1;
        while (c[t] == 0 && t > 0) {
            t--;
        }
        String ans ="";
        while(t>=0){ ans +=c[t];t--;}

        return ans;
    }
    public boolean isBoomerang(int[][] points) {
        Set<int[]> set =new  HashSet<>();
        if((points[0][0]==points[1][0]&&points[1][1]==points[0][1])
                ||(points[1][0]==points[2][0]&&points[1][1]==points[2][1])
                ||(points[0][0]==points[2][0]&&points[0][1]==points[2][1]))
            return false;
        if(points[1][0]-points[0][0]==0){
            if(points[2][0] -points[1][0]==0) return false;
            else return true;
        }else {
            if(points[2][0]-points[1][0]==0) return true;
            else{
                return !((points[2][1]-points[1][1])*1.0/(points[2][0]-points[1][0])==(points[1][1]-points[0][1])*1.0/(points[1][0]-points[0][0]));
            }
        }
    }
    public TreeNode bstToGst(TreeNode root) {
        if(root==null)
            return null;
        valOfRight(root, new TreeNode(0));
        return root;

    }
    public TreeNode valOfRight(TreeNode root,TreeNode pre){
        if(root==null )
            return pre;
        root.val = root.val +valOfRight(root.right,pre).val;
        pre = root;
        if(root.left!=null){
            pre =valOfRight(root.left,pre);
        }

        return pre;
    }
//    public int[] numMovesStonesII(int[] stones){
//        if(stones.length<=2) return new int[]{0,0};
//        Arrays.sort(stones);
//        for(int i=1;i<stones.length;i++){
//
//        }
//
//    }
    public int minScoreTriangulation(int[] A) {

        int n = A.length;
        int t[][]= new int[n+1][n+1];
        int s[][] = new int[n+1][n+1];
        for(int i=1; i<=n; i++)
        {
            t[i][i] = 0;
        }
        for(int r=2; r<=n; r++) //r为当前计算的链长（子问题规模）
        {
            for(int i=1; i<=n-r+1; i++)//n-r+1为最后一个r链的前边界
            {
                int j = i+r-1;//计算前边界为r，链长为r的链的后边界

                t[i][j] = t[i+1][j] + A[i-1]*A[i]*A[j];//将链ij划分为A(i) * ( A[i+1:j] )这里实际上就是k=i

                s[i][j] = i;

                for(int k=i+1; k<j; k++)
                {
                    //将链ij划分为( A[i:k] )* (A[k+1:j])
                    int u = t[i][k] + t[k+1][j] + A[i-1]*A[k]*A[j];
                    if(u<t[i][j])
                    {
                        t[i][j] = u;
                        s[i][j] = k;
                    }
                }
            }
        }
        return t[1][n-2];
    }

    public int longestValidParentheses(String s){
        int diff=0,start =0, ans=0;

        for(int i=0;i<s.length();i++){
            if(s.charAt(i)=='(') diff++;
            else {diff--;}

            if(diff<0){ diff =0;start=i+1;}
            else if(diff==0)  ans = Math.max(ans, i -start +1);
        }

        diff = 0; start = s.length()-1;
        for(int i=s.length()-1;i>=0;i--){
            if(s.charAt(i)==')') diff ++;
            else diff--;
            if(diff<0) {diff=0;start = i-1;}
            else if(diff==0) ans = Math.max(ans,start-i+1);
        }
        return ans;
    }

    public int quickRow(int a,int b,int c){
        long nul = a%c;
        long ans = 1;
        while(b>0){
            if((b&1)==1){
                ans = ans*nul%c;
            }
            b >>=1;
            nul = nul*nul%c;
        }
        return (int) ans;
    }

    public String minWindow(String s, String t) {
        Map<Character,Integer> map = new HashMap<>();
        for(int i=0;i<t.length();i++){
            map.put(t.charAt(i),map.getOrDefault(t.charAt(i),0)+1);
        }

        int left =0,right =0,min = Integer.MAX_VALUE,count =0,j=0;
        for(int i=0;i<s.length();i++){
            char c = s.charAt(i);
            if(map.containsKey(c)){
                if(map.get(c)>0) count++;
                map.put(c,map.get(c)-1);
            }
            while(count==t.length()){
                if(i-j<min){
                    left = j;
                    right = i;
                    min = i-j;
                }
                char z =s.charAt(j);
                if(map.containsKey(z)){
                    if(map.get(z)>=0) count--;
                    map.put(z,map.get(z)+1);
                }
                j++;
            }
        }

        return min==Integer.MAX_VALUE?"":s.substring(left,right+1);
    }

}
