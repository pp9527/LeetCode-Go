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
     * @Description: 3. 无重复字符的最长子串
     *                  滑动窗口
     * @author pwz
     * @date 2022/10/10 10:36
     * @param s
     * @return int
     */
    public int lengthOfLongestSubstring(String s) { // 59%/42%
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

    public int lengthOfLongestSubstring_1(String s) { // 100%/82%
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
}