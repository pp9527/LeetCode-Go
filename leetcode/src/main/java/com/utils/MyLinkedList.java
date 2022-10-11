package com.utils;

/**
 * @author: pwz
 * @create: 2022/10/11 9:51
 * @Description:
 * @FileName: MyLinkedList
 */
public // 707
class MyLinkedList {
    /**
     * 设计链表  实现接口
     */
    int size;
    ListNode head;

    public MyLinkedList() {
        size = 0;
        head = new ListNode(0);
    }

    public int get(int index) {
        if (index < 0 || index >= size)
            return -1;
        ListNode listNode = head;
        for (int i = 0;i <= index;i++) {
            listNode = listNode.next;
        }
        return listNode.val;
    }

    public void addAtHead(int val) {
        addAtIndex(0, val);
    }

    public void addAtTail(int val) {
        addAtIndex(size, val);
    }

    public void addAtIndex(int index, int val) {
        if (index < 0) index = 0;
        if (index > size) return;
        size++;
        ListNode pre = head;
        for (int i = 0;i < index;i++) {
            pre = pre.next;
        }
        ListNode add = new ListNode(val);
        add.next = pre.next;
        pre.next = add;
    }

    public void deleteAtIndex(int index) {
        if (index < 0 || index >= size) return;
        ListNode pre = head;
        size--;
        for (int i = 0;i < index;i++) {
            pre = pre.next;
        }
        pre.next = pre.next.next;
    }
}