package com.leetcode;

import java.util.HashSet;

/**
 * @author: pwz
 * @create: 2022/10/10 9:46
 * @Description:
 * @FileName: Hot100
 */
public class Hot100 {

    /**
     * @param s
     * @return int
     * @Description: 3. 无重复字符的最长子串
     * @author pwz
     * @date 2022/10/10 10:36
     */
    public int lengthOfLongestSubstring(String s) { // 滑动窗口59/42
        HashSet<Object> set = new HashSet<>();
        int length = s.length(), rk = -1, res = 0;
        for (int i = 0; i < length; i++) {
            if (i != 0) {
                set.remove(s.charAt(i - 1));
            }
            while (rk + 1 < length && !set.contains(s.charAt(rk + 1))) {
                set.add(s.charAt(rk + 1));
                rk++;
            }
            res = Math.max(res, rk - i + 1);
        }
        return res;
    }

    public int lengthOfLongestSubstring_1(String s) { // 滑动窗口100/82
        // 记录字符上一次出现的位置
        int[] last = new int[128];
        int n = s.length();
        int res = 0;
        int start = 0; // 窗口开始位置
        for (int i = 0; i < n; i++) {
            int index = s.charAt(i);
            start = Math.max(start, last[index] + 1);
            res = Math.max(res, i - start + 2);
            last[index] = i + 1;
        }
        return res;
    }

    /**
     * @param s
     * @return java.lang.String
     * @Description: 5. 最长回文子串
     * @author pwz
     * @date 2022/10/11 10:25
     */
    public String longestPalindrome(String s) { // 动态规划28/18
        boolean[][] dp = new boolean[s.length()][s.length()];
        int left = 0, maxLen = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            for (int j = i; j < s.length(); j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    if (j - i <= 1) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }
                if (dp[i][j] && j - i + 1 > maxLen) {
                    left = i;
                    maxLen = j - i + 1;
                }
            }
        }
        return s.substring(left, left + maxLen);
    }

    public String longestPalindrome_1(String s) { // 双指针、中心扩散法71/60
        String s1 = "", s2 = "", res = "";
        for (int i = 0; i < s.length(); i++) {
            s1 = extend(s, i, i);
            res = Math.max(s1.length(), res.length()) == res.length() ? res : s1;
            s2 = extend(s, i, i + 1);
            res = Math.max(s2.length(), res.length()) == res.length() ? res : s2;
        }
        return res;
    }

    private String extend(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return s.substring(left + 1, right);
    }

    public String longestPalindrome_2(String s) { // 双指针、中心扩散法 优化 100/98
        int[] range = new int[2];
        char[] str = s.toCharArray();
        for (int i = 0; i < str.length; i++) {
            // 跳过重复字符的遍历，减少时间
            i = findLongest(str, i, range);
        }
        return s.substring(range[0], range[1] + 1);
    }

    private int findLongest(char[] str, int low, int[] range) {
        int high = low;
        while (high < str.length - 1 && str[high + 1] == str[low]) {
            high++;
        }
        int ans = high;
        while (low > 0 && high < str.length - 1 && str[low - 1] == str[high + 1]) {
            low--;
            high++;
        }
        if (high - low > range[1] - range[0]) {
            range[0] = low;
            range[1] = high;
        }
        return ans;
    }
}