package com.utils;

import java.util.LinkedList;
import java.util.Queue;

/**
 * @author: pwz
 * @create: 2022/10/11 9:50
 * @Description:
 * @FileName: MyStack
 */
public class MyStack {

    Queue<Integer> queue1;
    Queue<Integer> queue2;
    public MyStack() {
        queue1 = new LinkedList<>();
        queue2 = new LinkedList<>();
    }

    public void push(int x) {
        queue2.offer(x);
        while (!queue1.isEmpty()) {
            queue2.offer(queue1.poll());
        }
        Queue<Integer> temQueue;
        temQueue = queue1;
        queue1 = queue2;
        queue2 = temQueue;
    }

    public int pop() {
        return queue1.poll();
    }

    public int top() {
        return queue1.peek();
    }

    public boolean empty() {
        return queue1.isEmpty();
    }
}