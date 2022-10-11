package com.utils;

import com.leetcode.LCodeSolution;

/**
 * @author: pwz
 * @create: 2022/10/11 9:52
 * @Description:
 * @FileName: ListNode
 */
public class ListNode {
    public int val;
    public ListNode next;

    public ListNode() {
    }

    public ListNode(int val) {
        this.val = val;
    }

    public ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}