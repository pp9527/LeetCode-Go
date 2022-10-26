package com.leetcode.pre;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @ClassName Solution
 * @Author PanWZ
 * @Data 2022/1/24 - 20:58
 * @Version: 1.8
 */
public class Solution {
    public boolean exist(char[][] board, String word) {
        char words[] = word.toCharArray();
        for (int i = 0;i < board.length;i++) {
            for (int j = 0;j < board[0].length;j++) {
                if (dfs(board, words, i, j, 0)) return true;
            }
        }
        return false;
    }

    boolean dfs(char[][] board, char[] words, int i, int j, int k) {
        if (i >= board.length || i < 0 || j >= board[0].length || j < 0 || board[i][j]
                != words[k]) return false;
        if (k == words.length - 1) return true;
        board[i][j] = '\0';
        boolean res = dfs(board, words, i + 1, j, k + 1) || dfs(board, words, i - 1, j, k + 1) ||
                      dfs(board, words, i, j + 1, k + 1) || dfs(board, words, i, j - 1, k + 1);
        board[i][j] = words[k];
        return res;
    }

    public void printOddTimesNum1(int arr[]) {
        int res = 0;
        for (int num : arr) {
            res = res ^ num;
        }
        System.out.println(res);
    }

    public void printOddTimesNum2(int arr[]) {
        int res = 0;
        for (int num : arr) {
            res = res ^ num;
        }
        int rightOne = res & (~res + 1); //取最右侧位的1
        int res2 = 0;
        for (int num : arr) {
            if ((num & rightOne) == 0)
                res2 = res2 ^ num;
        }
        System.out.println(res2 + "," + (res2 ^ res));
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();

        List<Integer> output = new ArrayList<>();
        for (int num : nums) {
            output.add(num);
        }

        int n = nums.length;
        backtrack(n, output, res, 0);
        return res;
    }
    public void backtrack(int n, List<Integer> output, List<List<Integer>> res, int first) {
        // 所有数都填完了
        if (first == n) {
            res.add(new ArrayList<Integer>(output));
        }
        for (int i = first; i < n; i++) {
            // 动态维护数组
            Collections.swap(output, first, i);
            // 继续递归填下一个数
            backtrack(n, output, res, first + 1);
            // 撤销操作
            Collections.swap(output, first, i);
        }
    }
}
