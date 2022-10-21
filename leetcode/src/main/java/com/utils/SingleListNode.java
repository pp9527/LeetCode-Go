package com.utils;

/**
 * @author: pwz
 * @create: 2022/10/21 12:21
 * @Description:
 * @FileName: SingleListNode
 */
class SingleListNode {
    ListNode head = new ListNode(0);

    public void add(ListNode listNode) {
        if (head == null) {
            head = listNode;
        } else {
            ListNode temp = head;
            while (temp.next != null) {
                temp = temp.next;
            }
            temp.next = listNode;
        }
    }

    public void list() {
        if (head.next == null) {
            System.out.println("空表...");
            return;
        }
        ListNode temp = head.next;
        while (temp != null) {
            System.out.println(temp.val);
            temp = temp.next;
        }
    }

    public static void main(String[] args) {
        SingleListNode head = new SingleListNode();
        ListNode node1 = new ListNode(3);
        ListNode node2 = new ListNode(4);
        ListNode node3 = new ListNode(1);
        head.list();
        head.add(node1);
        head.add(node2);
        head.add(node3);
        head.list();
    }

}