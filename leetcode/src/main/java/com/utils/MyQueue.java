package com.utils;

import java.util.Stack;

/**
 * @author: pwz
 * @create: 2022/10/11 9:55
 * @Description:
 * @FileName: MyQueue
 */
public class MyQueue {

    Stack<Integer> stack1;
    Stack<Integer> stack2;

    public MyQueue() {
        stack1 = new Stack<>();
        stack2 = new Stack<>();
    }

    public void push(int x) {
        stack1.push(x);
    }

    public int pop() {
        stack1ToStack2();
        return stack2.pop();
    }

    public int peek() {
        stack1ToStack2();
        return stack2.peek();
    }

    public boolean empty() {
        stack1ToStack2();
        if (stack2.isEmpty()) return true;
        return false;
    }

    void stack1ToStack2() {
        if (stack2.isEmpty()) {
            while (!stack1.isEmpty()) {
                stack2.push(stack1.pop());
            }
        }
    }
}