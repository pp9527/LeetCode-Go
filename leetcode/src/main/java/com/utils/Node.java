package com.utils;

import java.util.List;

/**
 * @author: pwz
 * @create: 2022/10/11 9:53
 * @Description:
 * @FileName: Node
 */
public class Node {
    public int val;
    public List<Node> children;

    public Node() {
    }

    public Node(int val) {
        this.val = val;
    }

    public Node(int val, List<Node> children) {
        this.val = val;
        this.children = children;
    }
}