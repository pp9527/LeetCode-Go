package com.utils;

import com.leetcode.LCodeSolution;

/**
 * @author: pwz
 * @create: 2022/7/19 16:20
 * @Description:
 * @FileName: TreeNode
 */
public class TreeNode {
    public int val;
    public TreeNode left;
    public TreeNode right;

    public TreeNode() {
    }

    public TreeNode(int val) {
        this.val = val;
    }

    public TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}