import java.util.*;

public class myString {
    /**
     * leetcode 318  最大单词长度乘积
     * 这题的主要思想在于：
     * 1.把字母转换成不同的数字，
     * int c = words[i][j] - 'a';
     * res[i] = res[i] | (1 << c);
     * 2.利用当前的字符串和之前的所有字符串进行比较得到不含公共字母的字符串
     *
     * @param words
     * @return
     */
    public int maxProduct(String[] words) {

        int f[] = new int[words.length];

        for (int i = 0; i < words.length; i++) {
            for (int j = 0; j < words[i].length(); j++) {
                f[i] = f[i] | (1 << words[i].charAt(j) - 'a');
            }
        }
        int res = 0;
        for (int i = 0; i < words.length; i++) {
            for (int j = i + 1; j < words.length; j++) {
                if ((f[i] & f[j]) == 0) res = Math.max(res, words[i].length() * words[j].length());
            }
        }
        return res;
    }

    /**
     * leetcode 290 单词模式
     */
    public boolean wordPattern(String pattern, String str) {
        String[] words = str.split(" ");
        if (words.length != pattern.length()) return false;
        Map<Character, String> map = new HashMap<Character, String>();

        for (int i = 0; i < words.length; i++) {
            char c = pattern.charAt(i);
            if (map.containsKey(c)) {
                if (!map.get(c).equals(words[i])) return false;
            } else {
                if (map.containsValue(words[i])) return false;
                map.put(c, words[i]);
            }
        }

        return true;
    }
    //leetcode 139 单词拆分

    /***
     *  可以当做完全背包问题哦
     *  dp[i]表示到i的子串能被拆分成单词
     * @param s
     * @param wordDict
     * @return
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        int maxLength =0,len = s.length();
        Set<String> m = new HashSet<>();
       for(String w:wordDict){
           maxLength = Math.max(w.length(),maxLength);
           m.add(w);
       }
       boolean dp[] = new boolean [len+1];
       int res[] = new int[len+1];

       dp[0] = true;
       for(int i=1;i<=len;i++){
           for(int j = Math.max(0,i-maxLength);j<i;j++){
               if(dp[j]&&m.contains(s.substring(j,i))){
                   dp[i] = true;
                   break;
               }
           }
       }
       return dp[len];
    }
    /***
     *  2019 京东笔试
     *  给出m个字符串s1,s2,...sm和一个单独的字符串，请选出尽可能多的字符串同时满足：
     *  1） 这些子串在T中不想交
     *  2）这些子串都是S1，s2。中某个串
     */
   public void jidong2(){
       Scanner in = new Scanner(System.in);
       int n = in.nextInt();
       Set<String> m = new HashSet<>();
       int maxLength = 0;
       while(n-->0){
           String s = in.next();
           m.add(s);
           maxLength = Math.max(maxLength,s.length());
       }
       String str = in.next();
       int len = str.length();
       boolean dp[] = new boolean[len+1];
       int  res[] = new int[len+1];
       dp[0] = true;
       for(int i=1;i<=len;i++){
           for(int j=Math.max(0,i-maxLength);j<i;j++){
               if(dp[j]&&m.contains(str.substring(j,i))){
                   res[i] = Math.max(res[i],res[j]+1);
                   dp[i] = true;
               }else{
                   res[i] = Math.max(res[i],res[i-1]);
               }
           }
       }
       System.out.println(res[len]);
   }
    public boolean isPalindrome (String s){
       int i=0,j = s.length()-1;
       char[] chars = s.toCharArray();
       while(i<j){
           if(chars[i]!=chars[j]) return false;
           i++;j--;
       }
       return true;
    }
    public int minCut(String s){
       int len = s.length();
       int dp[] = new int [len+1];
       for(int i=1;i<=len;i++){
           dp[i] = i-1;
       }
       dp[0] = -1;
       for(int i=1;i<=len;i++){
           for(int j=i-1;j>=0;j--){
               if(isPalindrome(s.substring(j,i))){
                   dp[i] =Math.min(dp[i], dp[j]+1);
               }
           }
       }
       return dp[len];
    }

    /***
     * leetcode115 不同的子序列数
     * 01背包求方案数
     * dp[i] 表示体积为到t[i]是否被匹配
     * g[i] 匹配到i的最大方案数
     * @param s
     * @param t
     * @return
     */
    public int numDistinct(String s,String t){
        int n = s.length(),m= t.length();
        boolean [] dp = new boolean[m+1];
        dp[0] = true;
        int[] g = new int[m+1];
        g[0] = 1;
        char[] chars = s.toCharArray();
        char[] ts = t.toCharArray();
        for(int i=0;i<n;i++){
            for(int j = m;j>0;j--){
                if(chars[i]==ts[j-1]){
                    boolean t1 = dp[j]|dp[j-1];//不匹配t[j]或者t[j-1]
                    int s1= 0;
                    if(dp[j]==t1) s1+=g[j];
                    if(dp[j-1]==t1) s1+=g[j-1];
                    dp[j] = t1;
                    g[j] = s1;
                }
            }
        }
        return g[m];
    }

    /**
     *
     * @param strs
     * @return
     */
    public String longestCommonPrefix(String[] strs) {
        int n = strs.length,i=0,m=0;
        boolean flag = false;
        if(strs.length==0) return "";

        String s = "";
        while(true){
            if(i==strs[0].length()) return s;
            char c = strs[0].charAt(i);
            for(int j = 1;j<strs.length;j++){
                if(i>=strs[j].length()||strs[j].charAt(i)!=c){
                    flag= true;
                    break;
                }
            }
            if(flag) return s;
            s +=c;
            i++;
        }
    }
}
