package com.utils;

import java.util.Stack;

/**
 * @author: pwz
 * @create: 2022/10/11 9:55
 * @Description:
 * @FileName: MinStack
 */
public class MinStack {

    Stack<Integer> stack;
    Stack<Integer> min_stack;

    public MinStack() {
        stack = new Stack<Integer>();
        min_stack = new Stack<Integer>();
        min_stack.push(Integer.MAX_VALUE);
    }

    public void push(int val) {
        stack.push(val);
        min_stack.push(Math.min(min_stack.peek(), val));
    }

    public void pop() {
        stack.pop();
        min_stack.pop();
    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return min_stack.peek();
    }
}