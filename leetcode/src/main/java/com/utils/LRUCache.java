package com.utils;

import com.sun.org.apache.bcel.internal.generic.NEW;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * @author: pwz
 * @create: 2022/10/28 12:01
 * @Description: 146. LRU 缓存
 * @FileName: LRUCache
 */
class LRUCache {

    class DLinkedNode {
       int key;
       int value;
       DLinkedNode pre;
       DLinkedNode next;
       DLinkedNode() {};
       DLinkedNode(int key1, int value1) {key = key1; value = value1;};
    }

    public int size;
    public int capacity;
    public Map<Integer, DLinkedNode> cache = new HashMap<>();
    DLinkedNode head, tail;

    public LRUCache(int capacity) {
       size = 0;
       this.capacity = capacity;
       head = new DLinkedNode();
       tail = new DLinkedNode();
       head.next = tail;
       tail.next = head;
    }

    public int get(int key) {
        DLinkedNode node = cache.get(key);
        if (node == null) return -1;
        moveToHead(node);
        return node.value;
    }

    private void moveToHead(DLinkedNode node) {
        removeNode(node);
        addToHead(node);
    }

    private void addToHead(DLinkedNode node) {
        node.next = head.next;
        node.pre = head;
        head.next.pre = node;
        head.next = node;
    }

    private void removeNode(DLinkedNode node) {
        node.pre.next = node.next;
        node.next.pre = node.pre;
    }

    public void put(int key, int value) {
        DLinkedNode node = cache.get(key);
        if (node == null) {
            DLinkedNode newNode = new DLinkedNode(key, value);
            cache.put(key, newNode);
            addToHead(newNode);
            size++;
            if (size > capacity) {
                DLinkedNode tail = removeTail(newNode);
                cache.remove(tail.key);
                size--;
            }
        } else {
            node.value = value;
            moveToHead(node);
        }
    }

    private DLinkedNode removeTail(DLinkedNode newNode) {
        DLinkedNode node = tail.pre;
        removeNode(node);
        return node;
    }
}