

import java.lang.reflect.Array;
import java.util.*;

public class LeetCode100 {
    //1
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> m = new HashMap<>();
        int res[] = new int[2];
        for (int i = 0; i < nums.length; i++) {
            int n = target - nums[i];
            if (m.containsKey(n)) {
                res[0] = m.get(n);
                res[1] = i;
                return res;
            } else {
                m.put(nums[i], i);
            }
        }

        return res;
    }

    //2
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode n1 = l1, n2 = l2;
        ListNode cur = new ListNode(0);
        ListNode head = cur;
        int c = 0;
        while (l1 != null && l2 != null) {
            int t = l1.val + l2.val + c;
            c = t / 10;
            t %= 10;
            cur.next = new ListNode(t);
            l1 = l1.next;
            l2 = l2.next;
            cur = cur.next;
        }
        while (l1 != null) {
            int t = l1.val + c;
            c = t / 10;
            t %= 10;
            cur.next = new ListNode(t);
            l1 = l1.next;
            cur = cur.next;
        }
        while (l2 != null) {
            int t = l2.val + c;
            c = t / 10;
            t %= 10;
            cur.next = new ListNode(t);
            l2 = l2.next;
            cur = cur.next;
        }
        if (c > 0) cur.next = new ListNode(c);

        return head.next;
    }

    //3
    public int lengthOfLongestSubstring(String s) {
        int n = s.length();
        char[] chars = s.toCharArray();
        int[] pre = new int[26];
        for (int i = 0; i < 26; i++) pre[i] = -1;
        int max = 0, cur = 0;
        for (int i = 0; i < n; i++) {
            if (i - pre[chars[i] - 'a'] > cur) {
                cur++;
            } else {
                if (cur > max) {
                    max = cur;
                }
                cur = i - pre[chars[i] - 'a'];
            }
            pre[chars[i] - 'a'] = i;
        }
        if (cur > max) {
            max = cur;
        }
        return max;
    }

    //4
    public double findMedianSortedArrays(int[] A, int[] B) {
        int m = A.length, n = B.length;
        if (m > n) {//确保m是小的
            int[] temp = A;
            A = B;
            B = temp;
            int t = m;
            m = n;
            n = t;
        }
        int min = 0, max = m, half = (m + n + 1) / 2;
        while (min <= max) {
            int i = (min + max) / 2;
            int j = half - i;
            if (i < max && B[j - 1] > A[i]) {// i 太小
                min = i + 1;
            } else if (i > min && B[j] < A[i - 1]) {// i 太大
                max = i - 1;
            } else {//划分的刚刚好
                int maxLeft = 0;
                if (i == 0) maxLeft = B[j - 1];
                else if (j == 0) maxLeft = A[i - 1];
                else maxLeft = Math.max(A[i - 1], B[j - 1]);

                if ((m + n) % 2 == 1) return (double) maxLeft;

                int minRight = 0;

                if (i == m) minRight = B[j];
                else if (j == n) minRight = A[i];
                else minRight = Math.min(B[j], A[i]);

                return ((double) maxLeft + minRight) / 2.0;
            }
        }
        return 0;
    }

    //5
    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) return "";
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    private int expandAroundCenter(String s, int left, int right) {
        int L = left, R = right;
        while (L >= 0 && R < s.length() && s.charAt(L) == s.charAt(R)) {
            L--;
            R++;
        }
        return R - L - 1;
    }

    //6
    public String convert(String s, int numRows) {
        String[] str = new String[numRows];
        for (int i = 0; i < numRows; i++) str[i] = "";
        if (numRows == 1) return s;
        for (int i = 0; i < s.length(); i += 2 * numRows - 2) {
            for (int j = i, k = 0; j < Math.min(s.length(), numRows + i); k++, j++) {
                str[k] += s.charAt(j);
            }
            for (int j = numRows + i, k = numRows - 2; j < Math.min(s.length(), 2 * numRows - 2 + i) && k >= 1; j++, k--) {
                str[k] += s.charAt(j);
            }
        }
        String res = "";
        for (int i = 0; i < numRows; i++) {
            res += str[i];
        }
        return res;
    }

    //7
    public int reverse(int x) {
        long res = 0;
        while (x != 0) {
            res = res * 10 + x % 10;
            x /= 10;
        }
        if (res > Integer.MAX_VALUE || res < Integer.MIN_VALUE) return 0;
        return (int) res;
    }

    //8
    public int myAtoi(String str) {
        str = str.trim();
        char flag = '+';
        long d = 0;
        if (str.length() <= 0) return 0;

        if (str.charAt(0) == '+' || str.charAt(0) == '-') flag = str.charAt(0);
        else if (str.charAt(0) < '0' || str.charAt(0) > '9') return 0;
        else d = str.charAt(0) - '0';
        long res = 0;
        for (int i = 1; i < str.length() && str.charAt(i) >= '0' && str.charAt(i) <= '9'; i++) {
            res = res * 10 + str.charAt(i) - '0';
            d *= 10;
            if ((d > Integer.MAX_VALUE || res > Integer.MAX_VALUE) && flag == '+') return Integer.MAX_VALUE;
            else if ((d > Integer.MAX_VALUE || res > Integer.MAX_VALUE) && flag == '-') return Integer.MIN_VALUE;
        }
        if (d > 0) res = d + res;

        if (flag == '-') res = -1 * res;

        if (res > Integer.MAX_VALUE) return Integer.MAX_VALUE;
        else if (res < Integer.MIN_VALUE) return Integer.MIN_VALUE;

        return (int) res;
    }

    //9
    public boolean isPalindrome(int x) {
        if (x < 0) return false;
        if (x == 0) return true;
        String str = Integer.toString(x);
        int i = 0, j = str.length() - 1;
        while (i < j) {
            if (str.charAt(i) != str.charAt(j)) return false;
            i++;
            j--;
        }
        return true;
    }

    //10
    public boolean isMatch(String s, String p) {
        if (s == null || p == null) {
            return false;
        }
        int n = s.length(), m = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= m; i++) {
            if (p.charAt(i - 1) == '*' && dp[i - 2][0]) {
                dp[i][0] = true;
            }
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(i - 1) != '.' && p.charAt(i - 1) != '*') {
                    dp[i][j] = s.charAt(j - 1) == p.charAt(i - 1) && dp[i - 1][j - 1];
                } else if (p.charAt(i - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    if (p.charAt(i - 2) != s.charAt(j - 1) && p.charAt(i - 2) != '.') {
                        dp[i][j] = dp[i - 2][j];
                    } else {
                        dp[i][j] = dp[i - 1][j] | dp[i][j - 1] | dp[i - 2][j];
                    }
                }

            }
        }
        return dp[m][n];
    }

    //11
    public int maxArea(int[] height) {
        int max = 0, i = 0, j = height.length - 1;

        while (i < j) {
            int h = Math.min(height[i], height[j]);
            max = Math.max(h * (j - i), max);
            if (height[i] < height[j]) i++;
            else j--;
        }
        return max;
    }

    //12
    public String intToRoman(int num) {
        int[] n = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] s = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        String res = "";
        int i = 0;
        while (num > 0) {
            if (num >= n[i]) {
                res += s[i];
                num -= n[i];
            } else {
                i++;
            }
        }
        return res;
    }

    //13
    public int romanToInt(String s) {
        boolean[] c = new boolean[3];//用于记录I,X,C是否出现过，0-I，1-X,2-C
        int ans = 0;
        for (int i = 0; i < s.length(); i++) {
            char t = s.charAt(i);
            switch (t) {
                case 'M':
                    if (c[2]) {
                        ans += 1000 - 200;
                    } else {
                        ans += 1000;
                    }
                    break;
                case 'D':
                    if (c[2]) {
                        ans += 500 - 200;
                    } else {
                        ans += 500;
                    }
                    break;
                case 'C':
                    if (c[1]) {
                        ans += 100 - 20;
                    } else {
                        ans += 100;
                    }
                    c[2] = true;
                    break;
                case 'L':
                    if (c[1]) {
                        ans += 50 - 20;
                    } else {
                        ans += 50;
                    }
                    break;
                case 'X':
                    if (c[0]) {
                        ans += 10 - 2;
                    } else {
                        ans += 10;
                    }
                    c[1] = true;
                    break;
                case 'V':
                    if (c[0]) {
                        ans += 5 - 2;
                    } else {
                        ans += 5;
                    }
                    break;
                case 'I':
                    ans += 1;
                    c[0] = true;
                    break;
            }

        }

        return ans;
    }

    //14
    public String longestCommonPrefix(String[] strs) {
        if (strs.length <= 0) return "";
        int i = 0;
        String res = "";
        while (true) {
            if (i == strs[0].length()) return res;
            char c = strs[0].charAt(i);
            for (int j = 1; j < strs.length; j++) {
                if (i == strs[j].length() || strs[j].charAt(i) != c) {
                    return res;
                }
            }
            res += c;
            i++;
        }
    }

    //15
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if (nums.length < 3) {
            return res;
        }
        int n = nums.length;
        Arrays.sort(nums);
        int target = 0;
        for (int i = 0; i < n - 2; i++) {
            if (nums[i] > 0) break;
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            target = 0 - nums[i];
            int k = i + 1, j = n - 1;
            while (k < j) {
                //System.out.println(k);

                if (nums[k] + nums[j] == target) {
                    List<Integer> l = new ArrayList<Integer>();
                    l.add(nums[i]);
                    l.add(nums[k]);
                    l.add(nums[j]);
                    res.add(l);
                    while (k < j && nums[k] == nums[k + 1]) {
                        k++;
                    }
                    while (k < j && nums[j] == nums[j - 1]) {
                        j--;
                    }
                    k++;
                    j--;
                } else if (nums[k] + nums[j] < target) {
                    k++;

                } else {
                    j--;
                }

            }
        }
        return res;
    }

    //16
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int n = nums.length;
        int ans = nums[0] + nums[1] + nums[2];
        int dis = target - ans;
        if (dis <= 0) return ans;
        for (int i = 0; i < n - 2; i++) {
            int sum = target - nums[i];
            int j = i + 1, k = n - 1;
            while (j < k) {
                if (sum == nums[j] + nums[k]) return target;
                else {
                    int newDis = nums[j] + nums[k] - sum;
                    if (newDis > 0) {
                        if (dis > newDis) {
                            dis = newDis;
                            ans = nums[j] + nums[k] + nums[i];
                        }
                        k--;
                    } else {
                        if (dis > 0 - newDis) {
                            dis = 0 - newDis;
                            ans = nums[j] + nums[k] + nums[i];
                        }
                        j++;
                    }
                }
            }

        }
        return ans;
    }

    //17
    public List<String> letterCombinations(String digits) {
        Map<Character, String> m = new HashMap<>();
        m.put('2', "abc");
        m.put('3', "def");
        m.put('4', "ghi");
        m.put('5', "jkl");
        m.put('6', "mno");
        m.put('7', "pqrs");
        m.put('8', "tuv");
        m.put('9', "wxyz");
        List<String> ans = new ArrayList<>();
        for (int i = 0; i < digits.length(); i++) {
            int k = ans.size();
            List temp = new ArrayList<>();
            for (int j = 0; j < k; j++) {
                String t = ans.get(j);
                for (char c : m.get(digits.charAt(i)).toCharArray())
                    temp.add(t + c);
            }
            ans = temp;
            if (k == 0) {
                for (char c : m.get(digits.charAt(i)).toCharArray())
                    ans.add("" + c);
            }
        }
        return ans;
    }

    //18
    public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        int n = nums.length;
        for (int i = 0; i < n - 3; i++) {
            for (int j = i + 1; j < n - 2; j++) {
                int sum = target - nums[i] - nums[j];
                int t = j + 1, k = n - 1;
                while (t < k) {
                    if (nums[t] + nums[k] == sum) {
                        List<Integer> l = new ArrayList<>();
                        l.add(nums[i]);
                        l.add(nums[j]);
                        l.add(nums[t]);
                        l.add(nums[k]);
                        ans.add(l);
                        while (t < k && nums[t] == nums[t + 1]) t++;
                        while (k > j && nums[k] == nums[k - 1]) k--;
                        k--;
                        t++;
                    } else if (nums[t] + nums[k] > sum) k--;
                    else t++;
                }
                while (j < n - 2 && j < t && nums[j] == nums[j + 1]) j++;
            }
            while (i < n - 3 && nums[i] == nums[i + 1]) i++;
        }

        return ans;
    }

    //19
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode fast = head;
        ListNode slow = head;
        while (n-- > 0) fast = fast.next;
        if (fast == null) return head.next;
        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        slow.next = slow.next.next;
        return head;
    }

    //20
    public boolean isValid(String s) {
        Deque<Character> deque = new ArrayDeque<>();
        for (int i = 0; i < s.length(); i++) {
            switch (s.charAt(i)) {
                case ')':
                    if (deque.isEmpty() || deque.peekLast() != '(') return false;
                    else deque.pollLast();
                    break;
                case ']':
                    if (deque.isEmpty() || deque.peekLast() != '[') return false;
                    else deque.pollLast();
                    break;
                case '}':
                    if (deque.isEmpty() || deque.peekLast() != '{') return false;
                    else deque.pollLast();
                    break;
                default:
                    deque.addLast(s.charAt(i));
                    break;
            }
        }
        if (deque.isEmpty()) return true;
        return false;
    }

    //21
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        ListNode head = new ListNode(0);
        ListNode curr = head;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                curr.next = l1;
                l1 = l1.next;
            } else {
                curr.next = l2;
                l2 = l2.next;
            }
            curr = curr.next;
        }
        if (l1 != null) curr.next = l1;
        else curr.next = l2;

        return head.next;
    }

    //22 括号生成
    public List<String> generateParenthesis(int n) {
        List<String> ans = new ArrayList();
        backtrack(ans, "", 0, 0, n);
        return ans;
    }

    public void backtrack(List<String> ans, String cur, int open, int close, int max) {
        if (cur.length() == max * 2) {
            ans.add(cur);
            return;
        }

        if (open < max)
            backtrack(ans, cur + "(", open + 1, close, max);
        if (close < open)
            backtrack(ans, cur + ")", open, close + 1, max);
    }

    //23合并
    public ListNode mergeKLists(ListNode[] lists) {
        ListNode head = new ListNode(0);
        ListNode curr = head;
        int min = 0, minIndex = 0;
        boolean flag = false;
        while (true) {
            min = Integer.MAX_VALUE;
            flag = false;
            for (int i = 0; i < lists.length; i++) {
                if (lists[i] != null && lists[i].val < min) {
                    min = lists[i].val;
                    minIndex = i;
                    flag = true;
                }
            }

            if (flag) {
                curr.next = lists[minIndex];
                lists[minIndex] = lists[minIndex].next;
                curr = curr.next;
            } else {
                break;
            }

        }
        return head.next;
    }

    //24两两交换
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode next = head.next;
        head = swapPairs(next.next);
        next = head;
        return next;
    }

    public ListNode swapPairs2(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode next = head.next;
        ListNode cur = head;
        ListNode pre = new ListNode(0);
        head = next;
        while (cur != null && next != null) {
            cur.next = next.next;
            pre.next = next;
            pre = cur;
            cur = cur.next;
            next = cur.next;
        }

        return head;
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        if (k == 1) return head;
        ListNode pre = null;
        ListNode cur = head;
        ListNode next = null;
        ListNode preList = cur;
        ListNode temp = cur;
        int i = 0;
        while (temp != null) {
            i++;
            temp = temp.next;
            if (i > k) break;
        }
        if (i <= k) return head;
        i = 0;
        temp = cur;
        while (i < k) {
            i++;
            temp = temp.next;
        }
        ListNode ans = temp;

        while (cur != null) {
            preList = cur;
            i = 0;
            temp = cur;
            while (temp != null) {
                i++;
                temp = temp.next;
                if (i > k) break;
            }
            if (i < k) break;
            i = 0;
            while (cur != null && i < k) {
                next = cur.next;
                cur.next = pre;
                pre = cur;
                cur = next;
                i++;
            }
            preList.next = cur;
        }
        return ans;
    }

    //26
    public int removeDuplicates(int[] nums) {
        if (nums.length <= 1) return nums.length;
        int index = 1, pre = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == pre) {
                continue;
            } else {
                nums[index++] = nums[i];
                pre = nums[i];
            }
        }
        return index;
    }

    //27
    public int removeElement(int[] nums, int val) {
        int len = nums.length, j = nums.length - 1;
        for (int i = 0; i < len; i++) {
            while (i <= j && nums[i] == val) {
                nums[i] = nums[j--];
                len--;
            }
        }
        return len;
    }

    //28
    public int divide(int dividend, int divisor) {
        if (dividend == 0) return 0;
        if (divisor == 1) return dividend;
        if (divisor == -1) {
            if (dividend == Integer.MAX_VALUE) {
                return Integer.MIN_VALUE;
            } else if (dividend == Integer.MIN_VALUE) {
                return Integer.MAX_VALUE;
            }
            return -dividend;
        }
        long absDividend = Math.abs((long) dividend);
        long absDivisor = Math.abs((long) divisor);
        int ans = 0;
        while (absDivisor <= absDividend) {
            int cnt = 1;
            while ((absDivisor << cnt) <= absDividend) {
                cnt++;
            }
            ans += 1 << (cnt - 1);
            absDividend -= absDivisor << (cnt - 1);
        }
        return (dividend ^ divisor) < 0 ? -ans : ans;
    }

    //29
    public List<Integer> findSubstring(String s, String[] words) {

        List<Integer> ans = new ArrayList<>();
        if (s.length() == 0 || words.length == 0) return ans;

        int len = words[0].length(), m = len * words.length;
        if (s.length() < m) return ans;

        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < words.length; i++) {
            map.put(words[i], map.getOrDefault(words[i], 0) + 1);
        }
        int i = 0;
        while (i <= s.length() - m) {
            if (check(s, map, words, i, i + m, len)) {
                ans.add(i);
            }
            i++;
        }

        return ans;
    }

    public boolean check(String s, Map<String, Integer> map, String[] words, int start, int end, int wordLength) {
        Map<String, Integer> m = new HashMap<>();
        for (int i = start; i < end; i += wordLength) {
            String sub = s.substring(i, i + wordLength);
            if (map.containsKey(sub)) {
                m.put(sub, m.getOrDefault(sub, 0) + 1);
            } else {
                return false;
            }
        }
        for (int i = 0; i < words.length; i++) {
            if (m.getOrDefault(words[i], 0) != map.get(words[i])) {
                return false;
            }
        }
        return true;
    }

    //leetcode 31
    public void nextPermutation(int[] nums) {
        int j = nums.length - 1, i = j - 1;
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }
        if (i != -1) {
            for (int k = i + 1; k <= j; k++) {
                if (k == j || nums[k] > nums[i] && nums[k + 1] <= nums[i]) {
                    int n = nums[i];
                    nums[i] = nums[k];
                    nums[k] = n;
                    break;
                }
            }
        }
        Arrays.sort(nums, i + 1, nums.length);
    }

    //32
    public int longestValidParentheses(String s) {
        int n = s.length();
        int dp[] = new int[n];//dp[i] 表示0-i的字符的最大有效括号数
        int ans = 0, diff = 0;
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) == '(') {
                diff++;
            } else if (diff > 0) {
                dp[i] = dp[i - 1] + 2;
                if (i > dp[i]) dp[i] += dp[i - dp[i]];
                if (ans < dp[i]) ans = dp[i];
                diff--;
            }

        }
        return ans;
    }

    //33
    public int search(int[] nums, int target) {
        int n = nums.length;
        int start = 0, end = n - 1;
        while (start < end) {
            int mid = (end + start) / 2;
            if (target == nums[start]) return start;
            if (target == nums[mid]) return mid;
            if (target == nums[end]) return end;
            if (nums[mid] > nums[start]) {
                if (target > nums[mid] || (nums[end] < nums[mid] && target < nums[end])) {
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            } else {
                if (target < nums[mid] || target > nums[start]) {
                    end = mid - 1;
                } else {
                    start = mid + 1;
                }
            }
        }
        return -1;

    }

    //34
    public int[] searchRange(int[] nums, int target) {
        int n = nums.length, i = 0, j = n - 1;
        int ans[] = new int[]{-1, -1};
        while (i <= j) {
            if (nums[i] == target) {
                ans[0] = i;
                break;
            }
            int mid = (i + j) / 2;
            if (target == nums[mid]) {
                j = mid;
            } else if (target > nums[mid]) {
                i = mid + 1;
            } else {
                j = mid - 1;
            }
        }

        if (ans[0] == -1) return ans;
        i = 0;
        j = n - 1;
        while (i <= j) {
            if (nums[j] == target) {
                ans[1] = j;
                break;
            }
            int mid = (i + j + 1) / 2;
            if (target == nums[mid]) {
                i = mid;
            } else if (target > nums[mid]) {
                i = mid + 1;
            } else {
                j = mid - 1;
            }
        }
        return ans;

    }

    //35
    public int searchInsert(int[] nums, int target) {
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] >= target) return i;
        }
        return nums.length;
    }

    //36 有效的数独
    public boolean isValidSudoku(char[][] board) {
        char[][] borad2 = new char[9][9];
        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 9; j++)
                borad2[j][i] = board[i][j];
        return checkRow(board) && checkRow(borad2) && checkBlcok(board);
    }

    public boolean checkRow(char[][] board) {

        for (int i = 0; i < 9; i++) {
            int[] f = new int[10];
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.')
                    f[board[i][j] - '0']++;
            }
            for (int j = 1; j <= 9; j++)
                if (f[j] > 1) return false;

        }
        return true;
    }

    public boolean checkBlcok(char[][] board) {
        for (int i = 0; i < 9; i += 3) {
            for (int j = 0; j < 9; j += 3) {
                int[] f = new int[10];
                for (int t = i; t < i + 3; t++)
                    for (int k = j; k < j + 3; k++) {
                        if (board[t][k] != '.') {
                            f[board[t][k] - '0']++;
                        }
                    }
                for (int k = 1; k <= 9; k++) {
                    if (f[k] > 1) return false;
                }
            }
        }
        return true;
    }

    //leetcode 36
    public boolean isValidSudoku2(char[][] board) {
        int row[] = new int[9], col[] = new int[9], squ[][] = new int[3][3];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    int num = board[i][j] - '0';
                    int m = 1 << num;
                    if ((row[i] & m) != 0 || (col[j] & m) != 0 || (squ[i / 3][j / 3] & m) != 0) return false;

                    row[i] |= m;
                    col[j] |= m;
                    squ[i / 3][j / 3] |= m;
                }
            }
        }
        return true;
    }

    public String countAndSay(int n) {
        if (n == 1) return "1";

        String s = countAndSay(n - 1);
        String ans = "";
        int t = 1, len = s.length();
        for (int i = 1; i < len; i++) {
            if (s.charAt(i) == s.charAt(i - 1)) t++;
            else {
                ans += t + s.charAt(i - 1);
                t = 1;
            }
        }
        ans += t + "" + s.charAt(len - 1);


        return ans;
    }

    List<List<Integer>> ans = new ArrayList<>();

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        combinationSum(candidates, 0, target, new ArrayList<>());
        return ans;
    }

    public void combinationSum(int[] candiates, int i, int target, List<Integer> list) { //i表示每个数应用了几次
        if (target == 0) {
            List<Integer> temp = new ArrayList<>(list);
            ans.add(temp);
            return;
        }
        if (i == candiates.length) return;

        if (candiates[i] > target) return;//直接返回，排序剪枝

        int cnt = 0;
        for (; target >= 0; cnt++) {//枚举使用当前数字多少次，注意可以使用0次
            combinationSum(candiates, i + 1, target, list);
            target -= candiates[i];
            list.add(candiates[i]);
        }
        while (cnt-- > 0) {
            list.remove(list.size() - 1);
        }

    }

    Set<List<Integer>> myset = new HashSet<>();

    //40
    public List<List<Integer>> combinationSum2(int[] candiates, int target) {
        Arrays.sort(candiates);//排序剪枝优化
        combinationSum2(0, candiates, target, new ArrayList<>());
        return new ArrayList<List<Integer>>(myset);
    }

    public void combinationSum2(int i, int[] candiates, int target, List<Integer> list) {
        if (target == 0) {
            myset.add(new ArrayList<>(list));
            return;
        }

        if (i == candiates.length) return;
        if (target < candiates[i]) return;

        combinationSum2(i + 1, candiates, target, list);//不选择这个数
        //选择这个数
        list.add(candiates[i]);
        combinationSum2(i + 1, candiates, target - candiates[i], list);
        list.remove(list.size() - 1);

        return;
    }

    //41
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                int temp = nums[i];
                nums[i] = nums[nums[i] - 1];
                nums[temp - 1] = temp;
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1)
                return i + 1;
        }
        return n + 1;
    }

    /**
     * (三次线性扫描) O(n)O(n)
     * 观察整个图形，想办法分解计算水的面积。
     * 注意到，每个矩形条上方所能接受的水的高度，是由它左边最高的矩形，和右边最高的矩形决定的。具体地，假设第 i 个矩形条的高度为 height[i]，且矩形条左边最高的矩形条的高度为 left_max[i]，右边最高的矩形条高度为 right_max[i]，则该矩形条上方能接受水的高度为 min(left_max[i], right_max[i]) - height[i]。
     * 需要分别从左向右扫描求 left_max，从右向左求 right_max，最后统计答案即可。
     * 注意特判 n==0。
     * 时间复杂度
     * 都是线性扫描，故只需要O(n)O(n)的时间。
     * <p>
     * 作者：wzc1995
     * 链接：https://www.acwing.com/solution/LeetCode/content/121/
     * 来源：AcWing
     * 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
     *
     * @param height
     * @return
     */
    //双向扫描法，第三次出现了，第一次是分糖果，第二次是有效括号对，第三次是接雨水
    public int trap(int[] height) {
        int n = height.length, ans = 0;
        if (n == 0) return ans;
        int[] leftMax = new int[n], rightMax = new int[n];
        leftMax[0] = height[0];
        for (int i = 1; i < n; i++) {
            leftMax[i] = Math.max(leftMax[i - 1], height[i]);
        }
        rightMax[n - 1] = height[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            rightMax[i] = Math.max(rightMax[i + 1], height[i]);
        }

        for (int i = 0; i < n; i++) {
            ans += Math.min(leftMax[i], rightMax[i]) - height[i];
        }

        return ans;
    }


    //43

    /**
     * 高精度乘法，模拟) O(nm)O(nm)
     * 本题是经典的高精度乘法，可以直接模拟竖式乘法计算。
     * 乘积的最大长度为两个乘数的长度之和。
     *
     * @param num1
     * @param num2
     * @return
     */
    public String multiply(String num1, String num2) {
        int m = num1.length(), n = num2.length();
        int[] a = new int[m], b = new int[n], c = new int[n + m];
        for (int i = m - 1, k = 0; i >= 0; i--, k++) {
            a[k] = num1.charAt(i) - 48;
        }
        for (int i = n - 1, k = 0; i >= 0; i--, k++) {
            b[k] = num2.charAt(i) - 48;
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                c[i + j] += a[i] * b[j];
                c[i + j + 1] += c[i + j] / 10;
                c[i + j] %= 10;
            }
        }
        int l = n + m - 1;
        while (c[l] == 0 && l > 0) {
            l--;
        }
        String ans = "";
        for (int i = l; i >= 0; i--) {
            ans += c[i];
        }
        return ans;
    }

    //leetcode 44通配符匹配
    public boolean isMatch2(String s, String p) {
        int n = s.length(), m = p.length();
        boolean[][] dp = new boolean[n + 1][m + 1];
        dp[0][0] = true;
        for (int j = 0; j < m; j++)
            if (p.charAt(j) == '*') dp[0][j + 1] = dp[0][j];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (p.charAt(j) == '*') {
                    dp[i + 1][j + 1] = dp[i + 1][j] || dp[i][j + 1]; // 匹配0个||匹配多个
                } else if (p.charAt(j) == '?') {
                    dp[i + 1][j + 1] = dp[i][j];
                } else {
                    dp[i + 1][j + 1] = dp[i][j] && (s.charAt(i) == p.charAt(j));
                }
            }
        }

        return dp[n][m];
    }

    //45 跳跃游戏
    public int jump(int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;
        int max = nums[0], next = 0, ans = 0;
        for (int i = 1; i < n; i++) {
            if (max >= n - 1) {
                ans++;
                break;
            }
            if (i <= max) {
                next = Math.max(i + nums[i], next);
            } else {
                max = next;
                i--;
                ans++;
            }
        }
        return ans;
    }

    //46全排列
    List<List<Integer>> ans3 = new ArrayList<>();

    public List<List<Integer>> permute(int[] nums) {
        permute(nums, new boolean[nums.length], new ArrayList<>(), 0);
        return ans3;
    }

    public void permute(int[] nums, boolean visited[], List<Integer> list, int n) {
        if (n == nums.length) {
            ans3.add(new ArrayList<>(list));
            return;
        }
        for (int i = 0; i < nums.length; i++) {

            if (!visited[i]) {
                list.add(nums[i]);
                visited[i] = true;
                permute(nums, visited, list, n + 1);
                list.remove(list.size() - 1);
                visited[i] = false;
            }
        }

    }

    Set<List<Integer>> ans4 = new HashSet<>();

    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        permuteUnique(nums, 0, new boolean[nums.length], new ArrayList<>());
        return new ArrayList<>(ans4);
    }

    public void permuteUnique(int[] nums, int n, boolean[] visited, List<Integer> list) {
        if (n == nums.length) {
            ans4.add(new ArrayList<>(list));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i] || i > 0 && nums[i] == nums[i - 1] && !visited[i - 1]) continue;
            if (!visited[i]) {
                list.add(nums[i]);
                visited[i] = true;
                permuteUnique(nums, n + 1, visited, list);
                visited[i] = false;
                list.remove(list.size() - 1);
            }
        }
        return;
    }

    //44旋转图像

    /**
     * 分解为两步，第一步上下两两交换
     * 第二步转置
     *
     * @param matrix
     */
    public void rotate(int[][] matrix) {
        int i = 0, n = matrix.length, j = n - 1;
        //第一步上下两两交换
        while (i < j) {
            for (int k = 0; k < n; k++) {
                int t = matrix[i][k];
                matrix[i][k] = matrix[j][k];
                matrix[j][k] = t;
            }
            i++;
            j--;
        }
        //第二步转置
        for (i = 0; i < n; i++)
            for (j = i + 1; j < n; j++) {
                int t = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = t;
            }
        return;
    }

    //49 字母异位词
    //计数排序法
    public String getString(int[] nums) {
        String ans = "";
        for (int i = 0; i < nums.length; i++)
            ans += nums[i];
        return ans;
    }

    public List<List<String>> groupAnagrams(String[] strings) {
        Map<String, List<String>> map = new HashMap<>();
        for (int i = 0; i < strings.length; i++) {
            int nums[] = new int[26];
            for (int j = 0; j < strings[i].length(); j++) {
                nums[strings[i].charAt(j) - 'a']++;
            }
            String s = getString(nums);
            if (map.containsKey(s)) {
                map.get(s).add(strings[i]);
            } else {
                map.put(s, new ArrayList<>());
                map.get(s).add(strings[i]);
            }

        }
        List<List<String>> ans = new ArrayList<>();
        for (List<String> list : map.values()) {
            ans.add(new ArrayList<>(list));
        }
        return ans;
    }

    //数组排序法
    public List<List<String>> groupAnagrams2(String[] str) {
        if (str.length == 0) return new ArrayList();
        Map<String, List> m = new HashMap();
        for (String s : str) {
            char[] chars = s.toCharArray();
            Arrays.sort(chars);
            String key = String.valueOf(chars);
            if (!m.containsKey(key)) m.put(key, new ArrayList());
            m.get(key).add(s);
        }
        return new ArrayList(m.values());
    }

    //50 Pow(x,n)快速幂目前还没学
    //N皇后
    //N皇后2

    //53 最大连续数组
    public int maxSubArray(int[] nums) {
        if (nums.length == 0) return 0;
        int max = nums[0], a = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (a + nums[i] > nums[i]) {
                a += nums[i];
            } else {
                a = nums[i];
            }
            if (a > max) max = a;
        }
        return max;
    }

    //54螺旋矩阵
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> rst = new ArrayList<>();
        if (null == matrix || matrix.length == 0)
            return rst;
        int m = matrix.length;
        int n = matrix[m - 1].length;
        int len = Math.min(m, n);
        int times = len % 2 == 0 ? len / 2 : len / 2 + 1;
        int c1 = 0, c2 = n - 1, r1 = 0, r2 = m - 1;
        for (int i = 0; i < times; i++) {
            for (int k = c1; k <= c2; k++)
                rst.add(matrix[r1][k]);
            for (int k = r1 + 1; k <= r2; k++)
                rst.add(matrix[k][c2]);
            if (c1 < c2 && r1 < r2) {
                for (int k = c2 - 1; k >= c1; k--)
                    rst.add(matrix[r2][k]);
                for (int k = r2 - 1; k > r1; k--)
                    rst.add(matrix[k][c1]);
            }
            c1++;
            c2--;
            r1++;
            r2--;
        }
        return rst;
    }

    //57 插入区间
    public int[][] insert57(int[][] intervals, int[] newInterval) {
        int max = 0, min = 0, n = intervals.length, count = 0;
        int[][] temp = new int[n + 1][2];
        boolean flag = false;
        for (int i = 0; i < n; i++) {
            if (intervals[i][0] <= newInterval[1] && intervals[i][1] >= newInterval[0]) {
                newInterval[0] = Math.min(intervals[i][0], newInterval[0]);
                newInterval[1] = Math.max(intervals[i][1], newInterval[1]);
            } else if (intervals[i][0] > newInterval[1] && !flag) {
                temp[count][0] = newInterval[0];
                temp[count++][1] = newInterval[1];
                temp[count][0] = intervals[i][0];
                temp[count++][1] = intervals[i][1];
                flag = true;
            } else {
                temp[count][0] = intervals[i][0];
                temp[count++][1] = intervals[i][1];
            }
        }
        if (!flag) {
            temp[count][0] = newInterval[0];
            temp[count++][1] = newInterval[1];
        }
        int[][] ans = Arrays.copyOf(temp, count);
        return ans;
    }

    //58最后一个单词长度
    public int lengthOfLastWord(String s) {
        s = s.trim();
        if (s.length() == 0) return 0;
        int ans = 0;
        for (int i = s.length() - 1; i >= 0; --i) {
            if (s.charAt(i) != ' ') ans++;
            else return ans;
        }
        return ans;
    }

    //59 螺旋矩阵
    //59 螺旋矩阵
    public int[][] generateMatrix(int n) {
        int ans[][] = new int[n][n];
        int index = 1, i = 0, j = 0, k = 0;
        while (index <= n * n) {
            while (j < n - 1 - k) ans[i][j++] = index++;
            while (i < n - 1 - k) ans[i++][j] = index++;
            while (j > k) ans[i][j--] = index++;
            while (i > k) ans[i--][j] = index++;
            if (ans[i][j] == 0) ans[i][j] = index++;
            i++;
            j++;
            k++;


        }
        return ans;
    }

    //60
    public String getPermutation(int n, int k) {
        String ans = "";
        boolean[] st = new boolean[n];
        int[] f = new int[n];
        f[0] = 1;
        for (int i = 1; i < n; i++) f[i] = f[i - 1] * i;
        for (int i = 0; i < n; i++) {//从高位到低位枚举每一位
            int j = n - 1 - i, next = 0;
            if (k > f[j]) {//确定当前位是第几个未使用过的数
                int t = k / f[j];
                k %= f[j];
                if (k == 0) {
                    k = f[j];
                    t--;
                }
                while (t > 0) {
                    if (!st[next]) t--;
                    next++;
                }
            }
            while (st[next]) next++;
            ans += next + 1;
            st[next] = true;
        }
        return ans;
    }

    //61旋转链表
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null) return null;
        int len = 0;
        ListNode fast = head;
        while (fast != null) {
            len++;
            fast = fast.next;
        }
        k = k % len;
        fast = head;
        ListNode slow = head;
        while (k-- > 0) {
            fast = fast.next;
        }

        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        fast.next = head;
        head = slow.next;
        slow.next = null;

        return head;

    }

    //61 不同路径
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 1; i < m; i++) dp[i][0] = 1;
        for (int j = 1; j < n; j++) dp[0][j] = 1;

        for (int i = 1; i < m; i++)
            for (int j = 1; j < n; j++)
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        return dp[m - 1][n - 1];
    }

    //62不同路径II
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length, n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++)
            if (obstacleGrid[i][0] == 1) break;
            else dp[i][0] = 1;
        for (int j = 0; j < n; j++)
            if (obstacleGrid[0][j] == 1) break;
            else dp[0][j] = 1;

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) dp[i][j] = 0;
                else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    //64最小路径和
    public int minPathSum(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++)
            for (int j = 1; j <= n; j++) {
                if (i == 1) dp[i][j] = dp[i][j - 1] + grid[i - 1][j - 1];
                else if (j == 1) dp[i][j] = dp[i - 1][j] + grid[i - 1][j - 1];
                else dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i - 1][j - 1];
            }
        return dp[m][n];
    }

    //65 验证数
    public boolean isNumber(String s) {
        s = s.trim();
        boolean num = false, numAfterE = false, seeE = false, seePoint = false;
        for (int i = 0; i < s.length(); i++) {
            char t = s.charAt(i);
            if (t == '+' || t == '-') {
                if (i != 0 && s.charAt(i - 1) != 'e')
                    return false;
            } else if (t >= '0' && t <= '9') {
                num = true;
                numAfterE = true;
            } else if (t == 'e') {
                if (!num || seeE) return false;
                seeE = true;
                numAfterE = false;
            } else if (t == '.') {
                if (seeE || seePoint) return false;
                seePoint = true;
            } else {
                return false;
            }


        }

        return num && numAfterE;
    }

    //66加一
    public int[] plusOne(int[] digits) {
        int n = digits.length, c = 1;
        for (int i = n - 1; i >= 0; i--) {
            if (c > 0) {
                digits[i] += c;
                c = digits[i] / 10;
                digits[i] %= 10;
            } else return digits;
        }
        if (c == 0) return digits;
        int[] ans = new int[n + 1];
        ans[0] = 1;
        for (int i = 1; i <= n; i++) ans[i] = digits[i - 1];

        return ans;
    }

    //68 文本左右对齐
    public List<String> fullJustify(String[] words, int maxWidth) {
        int num = 0, len = 0, i = 0;
        List<String> ans = new ArrayList<>();
        while (i < words.length) {
            if (num != 0) len += 1;
            len += words[i].length();
            num++;
            if (len > maxWidth) {
                len -= words[i].length();
                num--;
                if (num > 1) len -= 1;

                int k = maxWidth - len;
                int t = 0;
                if (num > 1) {
                    t = k % (num - 1);
                    k /= num - 1;
                }

                String s = "";
                for (int j = i - num; j < i - 1; j++) {
                    s += words[j] + " ";
                    for (int m = 0; m < k; m++) s += " ";
                    if (t > 0) {
                        s += " ";
                        t--;
                    }
                }
                s += words[i - 1];
                for (int j = s.length() + 1; j <= maxWidth; j++) s += " ";


                ans.add(s);

                num = 1;
                len = words[i].length();

            }
            i++;
        }
        String s = "";
        for (int j = i - num; j < i - 1; j++) {
            s += words[j] + " ";
        }
        s += words[i - 1];
        for (int j = len + 1; j <= maxWidth; j++) s += " ";
        ans.add(s);

        return ans;
    }

    //69平方根（牛顿迭代法）
    public static double sqrt(double c) {
        if (c < 0) return Double.NaN;
        double err = 1e-15;
        double t = c;
        while (Math.abs(t - c / t) > err * t)
            t = (c / t + t) / 2.0;
        return t;
    }

    //70爬楼梯
    public int climbStairs(int n) {
        if (n == 1) return 1;
        if (n == 2) return 2;
        int a = 1, b = 2, i = 3, c = a + b;
        while (i <= n) {
            c = a + b;
            a = b;
            b = c;
            i++;
        }
        return c;
    }

    //71简化路径
    public String simplifyPath(String path) {

        int n = path.length();
        if (path.charAt(n - 1) != '/') {
            n += 1;
            path += "/";
        }
        char[] ans = new char[n];
        int count = 0;
        char c;
        String s = "";
        for (int i = 0; i < n; i++) {
            c = path.charAt(i);
            if (count == 0) ans[count++] = c;
            else if (c == '/') {
                if (s.equals("..")) {
                    if (count > 1) {
                        int j = count - 1;
                        while (ans[--j] != '/') ;
                        count = j + 1;
                    }
                } else if (!s.equals("") && !s.equals(".")) {
                    for (int j = 0; j < s.length(); j++) ans[count++] = s.charAt(j);
                    ans[count++] = '/';
                }
                s = "";
            } else {
                s += c;
            }
        }
        if (count > 1) count--;
        return String.valueOf(ans, 0, count);
    }

    //72 编辑距离
    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int dp[][] = new int[m + 1][n + 1];
        for (int i = 1; i <= n; i++) dp[0][i] = i;
        for (int i = 1; i <= m; i++) dp[i][0] = i;

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1]) + 1);
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
                }
            }
        }
        return dp[m][n];
    }

    //73 置零
    public void setZeroes(int[][] matrix) {
        for (int i = 0; i < matrix.length; i++)
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == 0) dfs(matrix, i, j);
            }
        for (int i = 0; i < matrix.length; i++)
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == -99) matrix[i][j] = 0;
            }
    }

    public void dfs(int[][] matrix, int i, int j) {
        for (int k = 0; k < matrix[0].length; k++) {
            if (matrix[i][k] != 0) matrix[i][k] = -99;
        }
        for (int k = 0; k < matrix.length; k++)
            if (matrix[k][j] != 0) matrix[k][j] = -99;
    }

    //74 搜索二维矩阵
    public boolean binSearch(int[] arr, int target) {
        int n = arr.length;
        if (target < arr[0]) return false;
        if (target > arr[n - 1]) return false;
        int l = 0, r = n - 1;
        while (l <= r) {
            int mid = l + r >> 1;
            if (arr[mid] == target) return true;
            else if (arr[mid] < target) l = mid + 1;
            else r = mid - 1;
        }
        return false;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0) return false;
        int m = matrix.length, n = matrix[0].length;
        if (n == 0) return false;
        if (target < matrix[0][0] || target > matrix[m - 1][n - 1]) return false;
        if (target == matrix[0][0] || target == matrix[m - 1][n - 1]) return true;

        int l = 0, r = m - 1;
        while (l <= r) {
            if (l == r) {
                return binSearch(matrix[l], target);
            }
            int mid = l + r >> 1;
            if (matrix[mid][0] == target || matrix[mid][n - 1] == target) return true;
            else if (matrix[mid][0] > target) r = mid - 1;
            else if (matrix[mid][0] < target && matrix[mid][n - 1] > target) {
                l = mid;
                r = mid;
            } else l = mid + 1;
        }
        return false;

    }

    //75 颜色分类 双指针
    public void sortColors(int[] nums) {
        int l = 0, i = 0, r = nums.length - 1, temp = 0;
        while (i <= r) {
            if (nums[i] == 2) {
                temp = nums[i];
                nums[i] = nums[r];
                nums[r] = temp;
                r--;
            }
            if (nums[i] == 0) {
                temp = nums[l];
                nums[l] = nums[i];
                nums[i] = temp;
                l++;
                i++;
            }
            if (i <= r && nums[i] == 1) {
                i++;
            }
        }
    }
    //76最小覆盖子串

    public String minWindow(String s, String t) {
        String string = "";

        //hashmap来统计t字符串中各个字母需要出现的次数
        HashMap<Character, Integer> map = new HashMap<>();
        for (char c : t.toCharArray())
            map.put(c, map.containsKey(c) ? map.get(c) + 1 : 1);

        //用来计数 判断当前子字符串中是否包含了t中全部字符
        int count = 0;
        //记录当前子字符串的左右下标值
        int left = 0;
        int right = 0;
        //记录当前最小子字符串的大小以及第一最后字符的下标值
        int min = Integer.MAX_VALUE;
        int minLeft = 0;
        int minRight = 0;

        for (; right < s.length(); right++) {
            char temp = s.charAt(right);
            if (map.containsKey(temp)) {//向后遍历出所包含t的字符串
                count = map.get(temp) > 0 ? count + 1 : count;
                map.put(temp, map.get(temp) - 1);
            }
            while (count == t.length()) {//得出一个符合条件的子字符串
                if (right - left < min) {//更新min minLeft minRight 信息
                    min = right - left;
                    minLeft = left;
                    minRight = right;
                }
                char c = s.charAt(left);
                if (map.containsKey(c)) {//向左收缩 判断所删减的字符是否在map中
                    if (map.get(c) >= 0) count--;//count--时需要判断一下是否需要--
                    map.put(c, map.get(c) + 1);
                }
                left++;
            }
        }
        return min == Integer.MAX_VALUE ? "" : s.substring(minLeft, minRight + 1);
    }

    // 77组合
    List<List<Integer>> lists = new ArrayList<>();

    public List<List<Integer>> combine(int n, int k) {
        int[] a = new int[n];
        for (int i = 1; i <= n; i++) {
            a[i] = i;
        }
        combine(0, k, n, a, new ArrayList<>());
        return lists;
    }

    private void combine(int i, int k, int n, int[] a, List<Integer> list) {
        if (list.size() == k) {
            lists.add(new ArrayList<>(list));
            return;
        }
        int cnt = list.size();
        for (int j = i; j < n - k + cnt + 1; j++) {
            list.add(a[j]);
            combine(j + 1, k, n, a, list);
            list.remove(list.size() - 1);
        }
        return;
    }
    //子集

    public List<List<Integer>> subsets(int[] nums) {
        int n = nums.length;
        for (int i = 0; i <= n; i++) {
            combine(0, i, n, nums, new ArrayList<>());
        }
        return lists;
    }

    //79 单词搜索、
    public boolean exist(char[][] board, String word) {

        int n = board.length, m = board[0].length;
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == word.charAt(0) && dfs(i, j, board, word, 0, new boolean[n][m])) {
                    return true;
                }
            }
        }
        return false;
    }

    public boolean dfs(int i, int j, char[][] board, String word, int count, boolean[][] visited) {
        if (count == word.length()) return true;
        if (i < 0 || j < 0 || i >= board.length || j >= board[0].length) return false;
        if (word.charAt(count) == board[i][j] && !visited[i][j]) {
            count++;
            visited[i][j] = true;
            boolean ans = dfs(i - 1, j, board, word, count, visited) || dfs(i + 1, j, board, word, count, visited)
                    || dfs(i, j + 1, board, word, count, visited) || dfs(i, j - 1, board, word, count, visited);
            visited[i][j] = false;
            return ans;
        }
        return false;
    }

    //80删除排序数组中重复项
    public int removeDuplicates2(int[] nums) {
        int count = 0;
        for (int n : nums) {
            if (count < 2 || n > nums[count - 2]) {
                nums[count++] = n;
            }
        }
        return count;
    }

    //88 搜索旋转排序数组
    public boolean search2(int[] nums, int target) {
        if (nums == null || nums.length == 0) return false;
        int low = 0, high = nums.length - 1;
        while (low <= high) {
            //处理重复数字
            while (low < high && nums[low] == nums[low + 1]) low++;
            while (low < high && nums[high] == nums[high - 1]) high--;
            int mid = low + (high - low) / 2;
            if (nums[mid] == target) return true;
            if (nums[mid] >= nums[low]) { //左边部分有序
                if (target < nums[mid] && target >= nums[low]) high = mid - 1; //落在左边部分
                else low = mid + 1;
            } else {
                //右边升序的部分
                //落在右边部分
                if (target > nums[mid] && target <= nums[high]) low = mid + 1;
                else high = mid - 1;
            }
        }
        return false;
    }

    //82 删除排序链表中的重复元素II
    public ListNode deleteDuplicates2(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode node = head;
        boolean flag = false;
        while (head.next != null && head.val == head.next.val) {
            head = head.next;
            flag = true;
        }
        if (!flag) {
            node.next = deleteDuplicates(head.next);
            return node;
        }
        return deleteDuplicates(head.next);
    }

    //83
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode pre = head;
        ListNode curr = head.next;
        while (curr != null) {
            if (curr.val != pre.val) {
                pre.next = curr;
                pre = curr;
            }
            curr = curr.next;
        }
        pre.next = null;
        return head;
    }

    //84
    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int[] h = new int[n + 1];
        for (int i = 0; i < n; i++) h[i] = heights[i];
        h[n] = -1;
        Deque<Integer> st = new ArrayDeque<>();
        int ans = 0;
        for (int i = 0; i <= n; i++) {
            while (!st.isEmpty() && h[i] < h[st.peekLast()]) {
                int curr = st.pollLast();
                if (st.isEmpty()) {
                    ans = Math.max(i * h[curr], ans);
                } else {
                    ans = Math.max((i - st.peekLast() - 1) * h[curr], ans);
                }
            }
            st.addLast(i);
        }

        return ans;
    }

    //leetcode 85最大的矩形
    public int maximalRectangle(char[][] matrix) {
        if (matrix.length == 0) return 0;
        int m = matrix.length, n = matrix[0].length;
        int[] h = new int[n + 1], a = new int[n + 1];
        int ans = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') h[j] = h[j] + 1;
                else h[j] = 0;
                a[j] = h[j];
            }
            a[n] = -1;

            ans = Math.max(ans, sortStack(a, n));

        }
        return ans;
    }

    private int sortStack(int[] a, int n) {
        int[] s = new int[n + 1];
        int ans = 0, top = 0, t = 0;
        for (int i = 0; i <= n; i++) {
            if (top == 0 || a[s[top]] <= a[i]) {
                s[(++top)] = i;
            } else {
                while (top > 0 && a[s[top]] > a[i]) {
                    t = s[top];
                    ans = Math.max(ans, (i - t) * a[t]);
                    --top;
                }
                s[++top] = t;
                a[t] = a[i];

            }
        }
        return ans;

    }

    //leetcode86 分割链表
    public ListNode partition(ListNode head, int x) {
        ListNode l1 = new ListNode(-1);
        ListNode l2 = new ListNode(-1);
        ListNode cur1 = l1, cur2 = l2;
        while (head != null) {
            if (head.val < x) {
                cur1.next = new ListNode(head.val);
                cur1 = cur1.next;
            } else {
                cur2.next = new ListNode(head.val);
                cur2 = cur2.next;
            }
            head = head.next;

        }
        cur1.next = l2.next;
        return l1.next;
    }

    //88 合并两个有序数组
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int[] nums3 = new int[m + n];
        int i = 0, j = 0, k = 0;
        while (i < m && j < n) {
            if (nums1[i] < nums2[j]) nums3[k++] = nums1[i++];
            else nums3[k++] = nums2[j++];
        }
        while (i < m) nums3[k++] = nums1[i++];
        while (j < n) nums3[k++] = nums2[j++];

        for (i = 0; i < m + n; i++) {
            nums1[i] = nums3[i];
        }

        return;
    }

    //89格雷码
    // G(i) = i ^ (i/2);
    public List<Integer> grayCode(int n) {
        List<Integer> ret = new ArrayList<>();
        for (int i = 0; i < 1 << n; ++i)
            ret.add(i ^ i >> 1);
        return ret;
    }


    // 90 子集II
    List<List<Integer>> ans90 = new ArrayList<>();

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        for (int i = 0; i <= n; i++)
            subsetsWithDup(0, nums, i, n, new ArrayList<>());
        return ans90;
    }

    private void subsetsWithDup(int i, int[] nums, int k, int n, List<Integer> list) {
        if (k == list.size()) {
            ans90.add(new ArrayList<>(list));
            return;
        }
        int cnt = list.size();
        for (int j = i; j <= n + cnt - k; j++) {
            list.add(nums[j]);
            subsetsWithDup(j + 1, nums, k, n, list);
            list.remove(list.size() - 1);
            while (j < n + cnt - k && nums[j] == nums[j + 1]) {
                j++;
            }
        }
        return;
    }

    //91 解码方法
    public int numDecodings(String s) {
        //动态规划标记
        int[] f = new int[s.length()];
        char[] c = s.toCharArray();
        //边界情况
        if (c.length == 0) {
            return 0;
        }
        //第一个元素
        f[0] = c[0] > '0' ? 1 : 0;

        if (c.length == 1) {
            return f[0];
        }
        //f[1]的值是关键，写不好，将会出现各种错误
        int k = c[0] > '0' && c[1] > '0' ? 1 : 0;
        f[1] = k + (c[0] == '1' || c[0] == '2' && c[1] <= '6' ? 1 : 0);

        //从前往后遍历
        for (int i = 2; i < c.length; i++) {
            if (c[i] > '0') {//第一个元素大于0，添加情况
                f[i] += f[i - 1];
            }
            //在10-26之间则添加两个字母组成一个的情况
            if (c[i - 1] == '1' || (c[i - 1] == '2' && c[i] <= '6')) {
                f[i] += f[i - 2];
            }
        }

        return f[c.length - 1];
    }

    //92. 反转链表II
    public ListNode reverseBetween(ListNode head, int m, int n) {
        ListNode newHead = new ListNode(0);
        newHead.next = head;
        ListNode curr = newHead, next = null, pre = null;
        int i = 1;
        while (i < m) {
            curr = curr.next;
            i++;
        }

        ListNode p = curr.next;
        while (i < n) {
            ListNode tmp = p.next;
            p.next = tmp.next;
            tmp.next = curr.next;
            curr.next = tmp;
            i++;
        }

        return newHead.next;
    }

    //93复原IP地址
    public List<String> restoreIpAddresses(String s) {
        List<String> result = new ArrayList<>();
        doRestore(result, "", s, 0);
        return result;
    }

    private void doRestore(List<String> result, String path, String s, int k) {

        if (s.isEmpty() || k == 4) {
            if (s.isEmpty() && k == 4)
                result.add(path.substring(1));
            return;
        }
        // 以0开头的时候单作为一部分
        for (int i = 1; i <= (s.charAt(0) == '0' ? 1 : 3) && i <= s.length(); i++) {
            String part = s.substring(0, i);
            if (Integer.parseInt(part) <= 255)
                doRestore(result, path + "." + part, s.substring(i), k + 1);
            else return;
        }
    }

    //94 二叉树的中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        inorderTraversal(root, ans);
        return ans;
    }

    public void inorderTraversal(TreeNode root, List<Integer> list) {
        if (root == null) return;

        inorderTraversal(root.left, list);
        list.add(root.val);
        inorderTraversal(root.right, list);
    }

    //95 不同的二叉搜索树II
    public List<TreeNode> generateTrees(int n) {
        if (n < 1) return new ArrayList<>();
        return generateTrees(1, n, new List[n + 2][n + 2]);
    }

    private List<TreeNode> generateTrees(int start, int end, List[][] note) {
        //使用备忘录记录
        // note[i][j] 表示i-j的所有生成树
        List<TreeNode> res = new ArrayList<>();
        if (end < start) {
            res.add(null);
            return res;
        }

        for (int i = start; i <= end; i++) {
            List<TreeNode> list1 = note[start][i - 1];//左孩子的所有情况 start  --- i-1
            if (list1 == null) {
                note[start][i - 1] = generateTrees(start, i - 1, note);// 递归生成
                list1 = note[start][i - 1];
            }
            for (TreeNode left : list1) {
                List<TreeNode> list2 = note[i + 1][end]; // 右孩子的所有情况 i+1 -- end
                if (list2 == null) {
                    note[i + 1][end] = generateTrees(i + 1, end, note);
                    list2 = note[i + 1][end];
                }
                for (TreeNode right : list2) {//组成树结构
                    TreeNode root = new TreeNode(i);
                    root.left = left;
                    root.right = right;
                    res.add(root);
                }
            }
        }

        return res;

    }

    //96  不同的二叉搜索树
    public int numTrees(int n) {
        int[] dp = new int[n+1];
        dp[0] = 1;
        for(int i=1;i<=n;i++){
            for(int j=i-1;j>=i>>1;j--){
                if((i&1)==1&&j==i>>1)
                    dp[i] += dp[j]*dp[i-1-j];
                else
                    dp[i] += dp[j]*dp[i-1-j]*2;
            }
        }

        return dp[n];
    }

    //97 交错字符串
    public boolean isInterleave(String s1, String s2, String s3) {
        int n =s1.length(),m = s2.length(),l=s3.length();
        if(l!=n+m) return false;
        if(l==0) return true;
        boolean [][]dp = new boolean[n+1][m+1];
        dp[0][0] = true;
        for(int i=1;i<=n;i++){
            if(s1.charAt(i-1)==s3.charAt(i-1)) dp[i][0] = dp[i-1][0];
        }
        for(int i=1;i<=m;i++){
            if(s2.charAt(i-1)==s3.charAt(i-1)) dp[0][i] = dp[0][i-1];
        }
        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
                char c = s3.charAt(i+j-1);
                if(c== s1.charAt(i-1)&&c==s2.charAt(j-1)){
                    dp[i][j] = dp[i-1][j]||dp[i][j-1];
                }else if(c==s1.charAt(i-1)){
                    dp[i][j] = dp[i-1][j];
                }else if(c==s2.charAt(j-1)){
                    dp[i][j] = dp[i][j-1];
                }
            }
        }

        return dp[n][m];
    }

    //98 验证二叉搜索树
    public boolean isValidBST(TreeNode root) {
        if(root==null){
            return true;
        }
        List<Integer> l = new ArrayList<Integer>();
        travels(root,l);
        for(int i=0;i<l.size()-1;i++){
            if(l.get(i)>=l.get(i+1)){
                return false;
            }
        }
        return true;
    }
    public void travels(TreeNode r,List<Integer> l){
        if(r==null){
            return ;
        }
        travels(r.left,l);
        l.add(r.val);
        travels(r.right,l);
        return ;
    }

    //99 恢复二叉搜索树
    List<Integer> num = new ArrayList<>();
    int index = 0;
    public void recoverTree(TreeNode root){
        travel(root);
        Collections.sort(num);
        setNode(root);
    }
    private  void travel(TreeNode root){
        if(root==null) return ;
        travel(root.left);
        num.add(root.val);
        travel(root.right);
    }
    private void setNode(TreeNode root){
        if(root==null) return;
        setNode(root.left);
        root.val = num.get(index ++);
        setNode(root.right);
    }

    //100 相同的树
    public boolean isSameTree(TreeNode p,TreeNode q){
        if(p==null&&q==null) return true;
        if((p!=null&&q==null)|| (p==null&&q!=null)|| p.val!=q.val) return false;

        return isSameTree(p.left,q.left)&&isSameTree(p.right,q.right);
    }

}


