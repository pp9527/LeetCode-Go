package com.leetcode.cur;

import com.utils.ListNode;
import com.utils.TreeNode;

import java.util.*;
import java.util.stream.IntStream;

/**
 * @author: pwz
 * @create: 2022/6/1 17:06
 * @Description: 代码随想录
 * @FileName: Solution
 */
public class CodeCapricorns {

    /**
     * @param nums
     * @param target
     * @return int
     * @Description: 704. 二分查找
     * @author pwz
     * @date 2022/10/12 10:03
     */
    public int search(int[] nums, int target) { // 二分查找 O(n)
        int low = 0, high = nums.length - 1;
        while (low <= high) {
            int mid = (high - low) / 2 + low;
            int num = nums[mid];
            if (num == target) return mid;
            else if (num > target) high = mid - 1;
            else if (num < target) low = mid + 1;
        }
        return -1;
    }

    /**
     * @param nums
     * @param val
     * @return int
     * @Description: 27. 移除元素
     * @author pwz
     * @date 2022/10/12 10:06
     */
    public int removeElement(int[] nums, int val) { // 双指针 O(n)
        int pre = 0, n = nums.length;
        for (int i = 0; i < n; i++) {
            if (nums[i] != val) {
                nums[pre] = nums[i];
                pre++;
            }
        }
        return pre;
    }

    /**
     * @param nums
     * @return int[]
     * @Description: 977. 有序数组的平方
     * @author pwz
     * @date 2022/10/18 9:43
     */
    public int[] sortedSquares(int[] nums) { // 双指针 O(n)
        int left = 0, right = nums.length - 1;
        int[] ans = new int[right + 1];
        for (int i = right; i >= 0; i--) {
            int res1 = nums[left] * nums[left], res2 = nums[right] * nums[right];
            if (res1 > res2) {
                ans[i] = res1;
                left++;
            } else {
                ans[i] = res2;
                right--;
            }
        }
        return ans;
    }

    /**
     * @param target
     * @param nums
     * @return int
     * @Description: 209. 长度最小的子数组
     * @author pwz
     * @date 2022/10/18 9:52
     */
    public int minSubArrayLen(int target, int[] nums) { // 滑动窗口 O(n)
        int ans = Integer.MAX_VALUE;
        int sum = 0;
        int left = 0;
        for (int right = 0; right < nums.length; right++) {
            sum += nums[right];
            while (sum >= target) {
                ans = Math.min(ans, right - left + 1);
                sum -= nums[left++];
            }
        }
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }

    /**
     * @param n
     * @return int[][]
     * @Description: 59. 螺旋矩阵 II
     * @author pwz
     * @date 2022/10/18 9:53
     */
    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        int start = 0;
        int loop = 0;
        int count = 1;
        int i, j;
        while (loop++ < n / 2) {
            for (j = start; j < n - loop; j++) {
                res[start][j] = count++;
            }
            for (i = start; i < n - loop; i++) {
                res[i][j] = count++;
            }
            for (; j >= loop; j--) {
                res[i][j] = count++;
            }
            for (; i >= loop; i--) {
                res[i][j] = count++;
            }
            start++;
        }
        if (n % 2 == 1) {
            res[start][start] = count;
        }
        return res;
    }

    /**
     * @param head
     * @param val
     * @return com.utils.ListNode
     * @Description: 203. 移除链表元素
     * @author pwz
     * @date 2022/10/18 10:39
     */
    public ListNode removeElements(ListNode head, int val) {
        /**
         * 设置虚拟头节点处理头结点的删除情况
         */
        if (head == null) return null;
        ListNode dummy = new ListNode(-1, head);
        ListNode pre = dummy, cur = head;
        while (cur != null) {
            if (cur.val == val) {
                pre.next = cur.next;
            } else {
                pre = cur;
            }
            cur = cur.next;
        }
        return dummy.next;
    }

    /**
     * @author pwz
     * @Description: 707. 设计链表
     * @date 2022/10/20 10:31
     * @return null
     */
    class MyLinkedList {
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
            for (int i = 0; i <= index; i++) {
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
            for (int i = 0; i < index; i++) {
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
            for (int i = 0; i < index; i++) {
                pre = pre.next;
            }
            pre.next = pre.next.next;
        }
    }

    /**
     * @Description: 206. 翻转链表
     * @author pwz
     * @date 2022/10/20 10:37
     */
    public ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode temp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = temp;
        }
        return pre;
    }

    /**
     * @Description: 24. 两两交换链表中的节点
     * @author pwz
     * @date 2022/10/21 10:10
     */
    public ListNode swapPairs(ListNode head) {
        ListNode dummyNode = new ListNode(0);
        dummyNode.next = head;
        ListNode temp = dummyNode;
        while (temp.next != null && temp.next.next != null) {
            ListNode node1 = temp.next;
            ListNode node2 = temp.next.next;
            temp.next = node2;
            node1.next = node2.next;
            node2.next = node1;
            temp = node1;
        }
        return dummyNode.next;
    }

    /**
     * @Description: 19. 删除链表的倒数第 N 个结点
     * @author pwz
     * @date 2022/10/26 11:45
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode cur = dummy, pre = dummy;
        while (n >= 0) {
            pre = pre.next;
            n--;
        }
        while (pre != null) {
            pre = pre.next;
            cur = cur.next;
        }
        cur.next = cur.next.next;
        return dummy.next;
    }

    /**
     * @Description: 160. 链表相交
     * @author pwz
     * @date 2022/10/26 12:19
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        ListNode A = headA;
        ListNode B = headB;
        while (A != B) {
            A = A == null ? headB : A.next;
            B = B == null ? headA : B.next;
        }
        return A;
    }

    /**
     * @Description: 142. 环形链表 II
     * @author pwz
     * @date 2022/10/26 13:06
     */
    public ListNode detectCycle(ListNode head) {
        /**
         * 环形链表2
         * 1、哈希表
         */
//        HashSet<ListNode> set = new HashSet<>();
//        if (head == null) return null;
//        while (head != null) {
//            if (!set.add(head)) return head;
//            head = head.next;
//        }
//        return null;
        /**
         * 2、快慢指针
         */
        ListNode fast = head, low = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            low = low.next;
            if (fast == low) {
                fast = head;
                while (fast != low) {
                    fast = fast.next;
                    low = low.next;
                }
                return fast;
            }
        }
        return null;
    }

    /**
     * @Description: 242. 有效的字母异位词
     * @author pwz
     * @date 2022/10/18 10:42
     */
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) return false;
        int[] arr = new int[26];
        for (char ch : s.toCharArray()) {
            arr[ch - 'a']++;
        }
        for (char ch : t.toCharArray()) {
            arr[ch - 'a']--;
        }
        for (int i : arr) {
            if (i != 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * @param nums1
     * @param nums2
     * @return int[]
     * @Description: 349. 两个数组的交集
     * @author pwz
     * @date 2022/10/19 10:29
     */
    public int[] intersection(int[] nums1, int[] nums2) {
        HashSet<Integer> set = new HashSet<>();
        for (int i : nums1) {
            set.add(i);
        }
        HashSet<Integer> ans = new HashSet<>();
        for (int i : nums2) {
            if (set.contains(i)) {
                ans.add(i);
            }
        }
        return ans.stream().mapToInt(x -> x).toArray();
    }

    /**
     * @Description: 202. 快乐数
     * @author pwz
     * @date 2022/10/26 13:16
     */
    public boolean isHappy(int n) {
        HashSet<Integer> hashSet = new HashSet<>();
        while (n != 1 && !hashSet.contains(n)) {
            hashSet.add(n);
            n = getNextNum(n);
        }
        return n == 1;
    }

    int getNextNum(int n) {
        int res = 0;
        while (n > 0) {
            int tem = n % 10;
            res += tem * tem;
            n = n / 10;
        }
        return res;
    }

    /**
     * @Description: 1. 两数之和
     * @author pwz
     * @date 2022/10/26 13:25
     */
    public int[] twoSum(int[] nums, int target) {
        /**
         * 两数之和 返回下标
         * 1、暴力
         */
//        int[] ans = new int[2];
//        for (int i = 0;i < nums.length;i++) {
//            for (int j = i + 1;j < nums.length;j++) {
//                if (nums[i] + nums[j] == target){
//                    ans[0] = i;
//                    ans[1] = j;
//                    return ans;
//                }
//            }
//        }
//        return ans;
        /**
         * 2、哈希表
         */
        Map<Integer, Integer> hashMap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (hashMap.containsKey(target - nums[i]))
                return new int[]{hashMap.get(target - nums[i]), i};
            hashMap.put(nums[i], i);
        }
        return new int[0];
    }

    /**
     * @Description: 454. 四数相加 II
     * @author pwz
     * @date 2022/10/27 10:49
     */
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        Map<Integer, Integer> map = new HashMap<>();
        int res = 0;
        for (int i : nums1) {
            for (int j : nums2) {
                map.put(i + j, map.getOrDefault(i + j, 0) + 1);
            }
        }
        for (int i : nums3) {
            for (int j : nums4) {
                if (map.containsKey(-i - j)) {
                    res++;
                }
            }
        }
        return res;
    }

    /**
     * @Description: 383. 赎金信
     * @author pwz
     * @date 2022/10/27 13:11
     */
    public boolean canConstruct(String ransomNote, String magazine) {
        int[] arr = new int [26];
        for (char i : magazine.toCharArray()) {
            arr[i - 'a']++;
        }
        for (char i : ransomNote.toCharArray()) {
            if (--arr[i - 'a'] < 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * @Description: 18. 四数之和
     * @author pwz
     * @date 2022/10/28 10:22
     */
    public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < nums.length - 3; i++) {
            // 当最小值大于target结束
            if (nums[i] > target && target >= 0) break;
            // 去重
            if (i > 0 && nums[i - 1] == nums[i]) continue;
            // 当组合最小值大于target结束
            if ((long) nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) break;
            // 当该层循环组合最大值小于target  i++进入下层循环
            if ((long) nums[i] + nums[nums.length - 1] + nums[nums.length - 2] + nums[nums.length - 3] < target)
                continue;
            for (int j = i + 1; j < nums.length - 2; j++) {
                // 去重
                if (j > i + 1 && nums[j - 1] == nums[j]) continue;
                // 当该层组合最小值大于target结束
                if ((long) nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target) break;
                // 当该层循环组合最大值小于target  j++进入下层循环
                if ((long) nums[i] + nums[nums.length - 1] + nums[nums.length - 2] + nums[j] < target) continue;
                int left = j + 1, right = nums.length - 1;
                while (left < right) {
                    long sum =  (long) nums[i] + nums[j] + nums[left] + nums[right];
                    if (sum > target) {
                        right--;
                    } else if (sum < target) {
                        left++;
                    } else {
                        res.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        while (left < right && nums[left] == nums[left + 1]) left++;
                        while (left < right && nums[right] == nums[right - 1]) right--;
                        left++;
                        right--;
                    }
                }
            }
        }
        return res;
    }

    /**
     * @Description: 344. 反转字符串
     * @author pwz
     * @date 2022/10/28 11:01
     */
    public void reverseString(char[] s) {
        int l = 0;
        int r = s.length - 1;
        while (l < r) {
            s[l] ^= s[r];  //构造 a ^ b 的结果，并放在 a 中
            s[r] ^= s[l];  //将 a ^ b 这一结果再 ^ b ，存入b中，此时 b = a, a = a ^ b
            s[l] ^= s[r];  //a ^ b 的结果再 ^ a ，存入 a 中，此时 b = a, a = b 完成交换
            l++;
            r--;
        }
    }

    /**
     * @Description: 541. 反转字符串 II
     * @author pwz
     * @date 2022/10/29 9:36
     */
    public String reverseStr(String s, int k) {
        char[] ch = s.toCharArray();
        for (int i = 0; i < ch.length; i += 2 * k) {
            int start = i;
            int end = Math.min(ch.length - 1, start + k - 1);
            while (start < end) {
                ch[start] ^= ch[end];
                ch[end] ^= ch[start];
                ch[start] ^= ch[end];
                start++;
                end--;
            }
        }
        return String.valueOf(ch);
    }

    /**
     * @Description: 剑指 Offer 05. 替换空格
     * @author pwz
     * @date 2022/10/29 11:00
     */
    public String replaceSpace(String s) {
        if (s == null || s.length() == 0) return s;
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == ' ') {
                str.append("  ");
            }
        }
        int left = s.length() - 1;
        s += str.toString();
        int right = s.length() - 1;
        char[] chars = s.toCharArray();
        while (left < right) {
            if (chars[left] == ' ') {
                chars[right--] = '0';
                chars[right--] = '2';
                chars[right--] = '%';
            } else {
                chars[right--] = chars[left];
            }
            left--;
        }
        return new String(chars);
    }

    /**
     * @Description: 151. 反转字符串中的单词
     * @author pwz
     * @date 2022/10/31 11:07
     */
    public String reverseWords(String s) {
        s = s.trim();
        StringBuilder sb = new StringBuilder(s);
        reverse(sb, 0, sb.length() - 1);
        int start = 0;
        for (int i = 0; i < sb.length(); i++) {
            if (sb.charAt(i) == ' ' && sb.charAt(i - 1) != ' ') {
                reverse(sb, start, i - 1);
            }
            if (sb.charAt(i) == ' ' && sb.charAt(i + 1) != ' ') {
                start = i + 1;
            }
        }
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < sb.length(); i++) {
            if (sb.charAt(i) != ' ' || res.charAt(res.length() - 1) != ' ') {
                res.append(sb.charAt(i));
            }
        }
        return res.toString();
    }

    private void reverse(StringBuilder sb, int start, int end) {
        while (start < end) {
            char ch = sb.charAt(start);
            sb.setCharAt(start, sb.charAt(end));
            sb.setCharAt(end, ch);
            start++;
            end--;
        }
    }







































    // 473
    public boolean makeSquare(int[] matchsticks) {
        int total = Arrays.stream(matchsticks).sum();
        if (total % 4 != 0) return false;
        int[] edges = new int[4];
        Arrays.sort(matchsticks);
        for (int i = 0, j = matchsticks.length - 1; i < j; i++, j--) {
            int temp = matchsticks[i];
            matchsticks[i] = matchsticks[j];
            matchsticks[j] = temp;
        }
        return dfs(0, matchsticks, edges, total / 4);
    }

    public boolean dfs(int index, int[] matchsticks, int[] edges, int len) {
        if (index == matchsticks.length) return true;
        for (int i = 0; i < edges.length; i++) {
            edges[i] += matchsticks[index];
            if (edges[i] <= len && dfs(index + 1, matchsticks, edges, len))
                return true;
            edges[i] -= matchsticks[index];
        }
        return false;
    }

    // 459
    public boolean repeatedSubstringPattern(String s) {
        if (s.equals("")) {
            return false;
        }
        int len = s.length();
        s = " " + s;
        char[] chars = s.toCharArray();
        int[] next = new int[len + 1];
        for (int i = 2, j = 0; i <= len; i++) {
            while (j > 0 && chars[i] != chars[j + 1]) {
                j = next[j];
            }
            if (chars[i] == chars[j + 1]) {
                j++;
            }
            next[i] = j;
        }
        if (next[len] > 0 && len % (len - next[len]) == 0) {
            return true;
        }
        return false;
    }

    // 150
    public int evalRPN(String[] tokens) {
        Deque<Integer> stack = new LinkedList<>();
        for (String s : tokens) {
            if ("+".equals(s)) {
                stack.push(stack.pop() + stack.pop());
            } else if ("-".equals(s)) {
                stack.push(-stack.pop() + stack.pop());
            } else if ("*".equals(s)) {
                stack.push(stack.pop() * stack.pop());
            } else if ("/".equals(s)) {
                int tem1 = stack.pop();
                int tem2 = stack.pop();
                stack.push(tem2 / tem1);
            } else {
                stack.push(Integer.valueOf(s));
            }
        }
        return stack.pop();
    }

    // 239
    public int[] maxSlidingWindow(int[] nums, int k) {
        Deque<Integer> deque = new LinkedList<>();
        int n = nums.length;
        int[] res = new int[n - k + 1];
        for (int i = 0; i < k; i++) {
            while (!deque.isEmpty() && nums[deque.peekLast()] <= nums[i]) {
                deque.pollLast();
            }
            deque.offerLast(i);
        }
        res[0] = nums[deque.peekFirst()];
        for (int i = k; i < n; i++) {
            while (!deque.isEmpty() && nums[deque.peekLast()] <= nums[i]) {
                deque.pollLast();
            }
            deque.offerLast(i);
            while (deque.peekFirst() <= i - k) {
                deque.pollFirst();
            }
            res[i - k + 1] = nums[deque.peekFirst()];
        }
        return res;
    }

    // 347
    public int[] topKFrequent(int[] nums, int k) {
        int[] res = new int[k];
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        Set<Map.Entry<Integer, Integer>> entries = map.entrySet();
        PriorityQueue<Map.Entry<Integer, Integer>> queue = new PriorityQueue<>(
                ((o1, o2) -> o1.getValue() - o2.getValue())
        );
        for (Map.Entry<Integer, Integer> entry : entries) {
            queue.offer(entry);
            if (queue.size() > k) {
                queue.poll();
            }
        }
        for (int i = k - 1; i >= 0; i--) {
            res[i] = queue.poll().getKey();
        }
        return res;
    }

    // 222
    public int countNodes(TreeNode root) {
        /**
         * @Description: 求完全二叉树节点个数
         * @author: pwz
         * @date: 2022/7/19
         * @param root
         * @return: int
         */
        if (root == null) {
            return 0;
        }
        int leftDepth = lever(root.left);
        int rightDepth = lever(root.right);
        if (leftDepth == rightDepth) {
            return countNodes(root.right) + (1 << leftDepth);
        } else {
            return (1 << rightDepth) + countNodes(root.left);
        }
    }

    public int lever(TreeNode root) {
        int depth = 0;
        while (root != null) {
            depth++;
            root = root.left;
        }
        return depth;
    }

    // 110
    public boolean isBalanced(TreeNode root) {
        /**
         * @Description: 判断是否平衡二叉树
         * @author: pwz
         * @date: 2022/7/19
         * @param root
         * @return: boolean
         */
        if (root == null) {
            return true;
        } else {
            return Math.abs(getHeight(root.left) - getHeight(root.right)) <= 1
                    && isBalanced(root.left) && isBalanced(root.right);
        }
    }

    public int getHeight(TreeNode root) {
        if (root == null) {
            return 0;
        } else {
            return Math.max(getHeight(root.left), getHeight(root.right)) + 1;
        }
    }

    // 257
    public List<String> binaryTreePaths(TreeNode root) {
        /**
         * @Description: 求二叉树的所有路径
         * @author: pwz
         * @date: 2022/7/19
         * @param root
         * @return: java.util.List<java.lang.String>
         */
        List<String> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        List<Integer> paths = new ArrayList<>();
        traver(root, paths, res);
        return res;
    }

    public void traver(TreeNode root, List<Integer> paths, List<String> res) {
        paths.add(root.val);
        if (root.left == null && root.right == null) {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < paths.size() - 1; i++) {
                builder.append(paths.get(i)).append("->");
            }
            builder.append(paths.get(paths.size() - 1));
            res.add(builder.toString());
            return;
        }
        if (root.left != null) {
            traver(root.left, paths, res);
            paths.remove(paths.size() - 1);
        }
        if (root.right != null) {
            traver(root.right, paths, res);
            paths.remove(paths.size() - 1);
        }
    }

    // 700
    public TreeNode searchBST(TreeNode root, int val) {
        /**
         * @Description: 找到以等于给定值的节点为根节点的子树，返回子树的根节点
         *               常规递归方法、结合二叉搜索树特点优化递归
         * @author: pwz
         * @date: 2022/7/29 10:11
         * @param root
         * @param val
         * @return: new_solution.TreeNode
         */
        if (root == null || root.val == val) {
            return root;
        }
        if (val < root.val) {
            return searchBST(root.left, val);
        } else {
            return searchBST(root.right, val);
        }
    }

    // 98
    long pre = Long.MIN_VALUE;

    public boolean isValidBST(TreeNode root) {
        /**
         * @Description:判断二叉搜索树
         * @author: pwz
         * @date: 2022/7/21
         * @param root
         * @return: boolean
         */
        if (root == null) {
            return true;
        }
        if (!isValidBST(root.left)) {
            return false;
        }
        if (root.val <= pre) {
            return false;
        }
        pre = root.val;
        return isValidBST(root.right);
    }

    // 530
    TreeNode prev;
    int res = Integer.MAX_VALUE;

    public int getMinimumDifference(TreeNode root) {
        /**
         * @Description: 给你一棵所有节点为非负值的二叉搜索树，请你计算树中任意两节点的差的绝对值的最小值。
         *              二叉搜索树相当于有序数组。
         * @author: pwz
         * @date: 2022/7/21
         * @param root
         * @return: int
         */
        if (root == null) {
            return 0;
        }
        traver1(root);
        return res;
    }

    public void traver1(TreeNode root) {
        if (root == null) {
            return;
        }
        traver1(root.left);
        if (prev != null) {
            res = Math.min(res, root.val - prev.val);
        }
        prev = root;
        traver1(root.right);
    }

    // 701
    public TreeNode insertIntoBST(TreeNode root, int val) {
        /**
         * @Description: 二叉搜索树的插入
         * @author: pwz
         * @date: 2022/7/25
         * @param root
         * @param val
         * @return: new_solution.TreeNode
         */
        if (root == null) {
            return new TreeNode(val);
        }
        if (val > root.val) {
            root.right = insertIntoBST(root.right, val);
        } else if (val < root.val) {
            root.left = insertIntoBST(root.left, val);
        }
        return root;
    }

    // 669
    public TreeNode trimBST(TreeNode root, int low, int high) {
        /**
         * @Description: 修剪二叉搜索树，使其在给定范围内（2）
         * @author: pwz
         * @date: 2022/7/25
         * @param root
         * @param low
         * @param high
         * @return: new_solution.TreeNode
         */
        if (root == null) {
            return null;
        }
        if (root.val < low) {
            return trimBST(root.right, low, high);
        }
        if (root.val > high) {
            return trimBST(root.left, low, high);
        }
        root.left = trimBST(root.left, low, high);
        root.right = trimBST(root.right, low, high);
        return root;
    }

    // 108
    public TreeNode sortedArrayToBST(int[] nums) {
        /**
         * @Description: 将有序数组转化为高度平衡二叉搜索树
         * @author: pwz
         * @date: 2022/7/25
         * @param nums
         * @return: new_solution.TreeNode
         */
        TreeNode root = traversal(nums, 0, nums.length - 1);
        return root;
    }

    public TreeNode traversal(int[] nums, int left, int right) {
        if (left > right) {
            return null;
        }
        int mid = (right - left) / 2 + left;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = traversal(nums, left, mid - 1);
        root.right = traversal(nums, mid + 1, right);
        return root;
    }

    // 77
    List<List<Integer>> result77 = new ArrayList<>();
    LinkedList<Integer> path77 = new LinkedList<>();

    public List<List<Integer>> combine(int n, int k) {
        /**
         * @Description: 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
         * @author: pwz
         * @date: 2022/7/29 11:01
         * @param n
         * @param k
         * @return: java.util.List<java.util.List < java.lang.Integer>>
         */
        backTracking(n, k, 1);
        return result77;
    }

    public void backTracking(int n, int k, int startIndex) {
        if (path77.size() == k) {
            result77.add(new ArrayList<>(path77));
            return;
        }
        for (int i = startIndex; i <= n - (k - path77.size()) + 1; i++) {
            path77.add(i);
            backTracking(n, k, i + 1);
            path77.removeLast();
        }
    }

    // 216
    List<List<Integer>> result216 = new ArrayList<>();
    LinkedList<Integer> path216 = new LinkedList<>();

    public List<List<Integer>> combinationSum3(int k, int n) {
        /**
         * @Description: 找出所有相加之和为n的k个数的组合，只使用1-9，每个数最多使用一次
         * @author: pwz
         * @date: 2022/7/29 11:01
         * @param k
         * @param n
         * @return: java.util.List<java.util.List < java.lang.Integer>>
         */
        backTracking3(n, k, 0, 1);
        return result216;
    }

    public void backTracking3(int targetSum, int k, int sum, int startIndex) {
        if (sum > targetSum) {
            return;
        }
        if (path216.size() == k) {
            if (sum == targetSum) result216.add(new ArrayList<>(path216));
            return;
        }
        for (int i = startIndex; i <= 9 - (k - path216.size()) + 1; i++) {
            sum += i;
            path216.add(i);
            backTracking3(targetSum, k, sum, i + 1);
            sum -= i;
            path216.removeLast();
        }
    }

    // 40
    List<List<Integer>> result40 = new ArrayList<>();
    LinkedList<Integer> path40 = new LinkedList<>();
    int sum40 = 0;

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        /**
         * @Description: 给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
         *               candidates 中的每个数字在每个组合中只能使用一次。
         *               candidates的元素是有重复的。
         * @author: pwz
         * @date: 2022/7/29 11:01
         * @param candidates
         * @param target
         * @return: java.util.List<java.util.List < java.lang.Integer>>
         */
        Arrays.sort(candidates);
        backTracking2(candidates, target, 0);
        return result40;
    }

    public void backTracking2(int[] candidates, int target, int start) {
        if (sum40 == target) {
            result40.add(new ArrayList<>(path40));
            return;
        }
        for (int i = start; i < candidates.length && sum40 + candidates[i] <= target; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            sum40 += candidates[i];
            path40.add(candidates[i]);
            backTracking2(candidates, target, i + 1);
            sum40 -= path40.getLast();
            path40.removeLast();
        }
    }

    // 131
    List<List<String>> result131 = new ArrayList<>();
    LinkedList<String> path131 = new LinkedList<>();

    public List<List<String>> partition(String s) {
        /**
         * @Description: 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。
         * @author: pwz
         * @date: 2022/7/29 11:00
         * @param s
         * @return: java.util.List<java.util.List < java.lang.String>>
         */
        backTracking131(s, 0);
        return result131;
    }

    public void backTracking131(String s, int startIndex) {
        /**
         * @Description: 回溯主体
         * @author: pwz
         * @date: 2022/7/29 11:00
         * @param s
         * @param startIndex
         * @return: void
         */
        if (startIndex >= s.length()) {
            result131.add(new ArrayList<>(path131));
            return;
        }
        for (int i = startIndex; i < s.length(); i++) {
            if (isPalindrome(s, startIndex, i)) {
                path131.addLast(s.substring(startIndex, i + 1));
            } else {
                continue;
            }
            backTracking131(s, i + 1);
            path131.removeLast();
        }
    }

    public boolean isPalindrome(String s, int startIndex, int end) {
        /**
         * @Description: 判断回文子串
         * @author: pwz
         * @date: 2022/7/29 11:00
         * @param s
         * @param startIndex
         * @param end
         * @return: boolean
         */
        for (int i = startIndex, j = end; i < j; i++, j--) {
            if (s.charAt(i) != s.charAt(j)) {
                return false;
            }
        }
        return true;
    }

    // 93
    List<String> result93 = new ArrayList<>();

    public List<String> restoreIpAddresses(String s) {
        /**
         * @Description: 给定一个只包含数字的字符串 s ，用以表示一个 IP 地址，返回所有可能的有效 IP 地址
         * @author: pwz
         * @date: 2022/7/29 11:00
         * @param s
         * @return: java.util.List<java.lang.String>
         */
        backTracking93(s, 0, 0);
        return result93;
    }

    public void backTracking93(String s, int startIndex, int pointIndex) {
        /**
         * @Description: 回溯主体
         * @author: pwz
         * @date: 2022/7/29 10:59
         * @param s
         * @param startIndex
         * @param pointIndex
         * @return: void
         */
        if (pointIndex == 3) {
            if (isValid(s, startIndex, s.length() - 1)) {
                result93.add(s);
            }
            return;
        }
        for (int i = startIndex; i <= s.length() - 1; i++) {
            if (isValid(s, startIndex, i)) {
                s = s.substring(0, i + 1) + '.' + s.substring(i + 1);
                pointIndex++;
                backTracking93(s, i + 2, pointIndex);
                s = s.substring(0, i + 1) + s.substring(i + 2);
                pointIndex--;
            } else {
                break;
            }
        }
    }

    public boolean isValid(String s, int start, int end) {
        /**
         * @Description: 判断每段地址是否合法
         * @author: pwz
         * @date: 2022/7/29 10:59
         * @param s
         * @param start
         * @param end
         * @return: boolean
         */
        if (start > end) {
            return false;
        }
        if (s.charAt(start) == '0' && start != end) {
            return false;
        }
        int num = 0;
        for (int i = start; i <= end; i++) {
            if (s.charAt(i) > '9' || s.charAt(i) < '0') {
                return false;
            }
            num = num * 10 + (s.charAt(i) - '0');
            if (num > 255) {
                return false;
            }
        }
        return true;
    }

    // 90
    List<List<Integer>> result90 = new ArrayList<>();
    LinkedList<Integer> path90 = new LinkedList<>();

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        /**
         * @Description: 求子集，数组可能包含重复元素
         * @author: pwz
         * @date: 2022/7/29 10:59
         * @param nums
         * @return: java.util.List<java.util.List < java.lang.Integer>>
         */
        Arrays.sort(nums);
        backTracking90(nums, 0);
        return result90;
    }

    public void backTracking90(int[] nums, int startIndex) {
        result90.add(new ArrayList<>(path90));
        for (int i = startIndex; i < nums.length; i++) {
            if (i > startIndex && nums[i - 1] == nums[i]) {
                continue;
            }
            path90.add(nums[i]);
            backTracking90(nums, i + 1);
            path90.removeLast();
        }
    }

    // 491
    List<List<Integer>> result491 = new ArrayList<>();
    LinkedList<Integer> path491 = new LinkedList<>();

    public List<List<Integer>> findSubsequences(int[] nums) {
        /**
         * @Description: 返回所有递增子序列
         * @author: pwz
         * @date: 2022/7/29 10:59
         * @param nums
         * @return: java.util.List<java.util.List < java.lang.Integer>>
         */
        getSubsequences(nums, 0);
        return result491;
    }

    public void getSubsequences(int[] nums, int start) {
        if (path491.size() > 1) {
            result491.add(new ArrayList<>(path491));
        }
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = start; i < nums.length; i++) {
            if (!path491.isEmpty() && path491.getLast() > nums[i]) {
                continue;
            }
            if (map.getOrDefault(nums[i], 0) >= 1) {
                continue;
            }
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
            path491.add(nums[i]);
            getSubsequences(nums, i + 1);
            path491.removeLast();
        }
    }

    // 47
    List<List<Integer>> result47 = new ArrayList<>();
    LinkedList<Integer> path47 = new LinkedList<>();

    public List<List<Integer>> permuteUnique(int[] nums) {
        /**
         * @Description: 给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
         * @author: pwz
         * @date: 2022/7/29 10:59
         * @param nums
         * @return: java.util.List<java.util.List < java.lang.Integer>>
         */
        boolean[] used = new boolean[nums.length];
        Arrays.fill(used, false);
        Arrays.sort(nums);
        backTracking47(nums, used);
        return result47;
    }

    public void backTracking47(int[] nums, boolean[] used) {
        if (path47.size() == nums.length) {
            result47.add(new ArrayList<>(path47));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1] && used[i - 1] == false) {
                continue;
            }
            if (used[i] == false) {
                used[i] = true;
                path47.add(nums[i]);
                backTracking47(nums, used);
                path47.removeLast();
                used[i] = false;
            }
        }
    }

    // 455
    public int findContentChildren(int[] g, int[] s) {
        /**
         * @Description: 给孩子分饼干，每个孩子一个
         * @author: pwz
         * @date: 2022/7/29 10:58
         * @param g 孩子
         * @param s 饼干
         * @return: int
         */
        Arrays.sort(g);
        Arrays.sort(s);
        int count = 0;
        int start = 0;
        for (int i = 0; i < s.length && start < g.length; i++) {
            if (s[i] >= g[start]) {
                start++;
                count++;
            }
        }
        return count;
    }

    // 376
    public int wiggleMaxLength(int[] nums) {
        /**
         * @Description: 如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 摆动序列 。
         *               第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。
         *               子序列 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。
         *               给你一个整数数组 nums ，返回 nums 中作为 摆动序列 的 最长子序列的长度 。
         * @author: pwz
         * @date: 2022/7/29 10:58
         * @param nums
         * @return: int
         */
        if (nums.length <= 1) {
            return nums.length;
        }
        int curDiff = 0;
        int preDiff = 0;
        int result = 1;
        for (int i = 0; i < nums.length - 1; i++) {
            curDiff = nums[i + 1] - nums[i];
            if ((curDiff > 0 && preDiff <= 0) || (curDiff < 0 && preDiff >= 0)) {
                result++;
                preDiff = curDiff;
            }
        }
        return result;
    }

    // 122
    public int maxProfit(int[] prices) {
        /**
         * @Description: 股票的最大利润 给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。
         *               在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。
         *               你也可以先购买，然后在 同一天 出售。返回 你能获得的 最大 利润 。
         *               贪心算法只收集正利润。
         * @author: pwz
         * @date: 2022/7/30 9:55
         * @param prices
         * @return: int
         */
        int result122 = 0;
        for (int i = 1; i < prices.length; i++) {
            result122 += Math.max(prices[i] - prices[i - 1], 0);
        }
        return result122;
    }

    // 55
    public boolean canJump(int[] nums) {
        /**
         * @Description: 跳跃游戏
         *               给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。
         *               数组中的每个元素代表你在该位置可以跳跃的最大长度。
         *               判断你是否能够到达最后一个下标。
         *               贪心算法：不需要知道怎么跳的，只看覆盖范围，能到即可
         * @author: pwz
         * @date: 2022/7/30 9:59
         * @param nums
         * @return: boolean
         */
        if (nums.length == 1) {
            return true;
        }
        int coverRange = 0;
        for (int i = 0; i <= coverRange; i++) {
            coverRange = Math.max(coverRange, i + nums[i]);
            if (coverRange >= nums.length - 1) {
                return true;
            }
        }
        return false;
    }

    // 45
    public int jump(int[] nums) {
        /**
         * @Description: 跳跃游戏2
         *               给你一个非负整数数组 nums ，你最初位于数组的第一个位置。
         *               数组中的每个元素代表你在该位置可以跳跃的最大长度。
         *               你的目标是使用最少的跳跃次数到达数组的最后一个位置。
         *               假设你总是可以到达数组的最后一个位置。
         *               贪心算法：不关心什么时候跳。只关心最大覆盖范围。自动在最大范围跳。
         * @author: pwz
         * @date: 2022/7/30 10:16
         * @param nums
         * @return: int
         */
        int curRange = 0;
        int result45 = 0;
        int nextRange = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            nextRange = Math.max(nextRange, nums[i] + i);
            if (i == curRange) {
                result45++;
                curRange = nextRange;
            }
        }
        return result45;
    }

    // 1005
    public int largestSumAfterKNegations(int[] nums, int k) {
        /**
         * @Description: K 次取反后最大化的数组和
         *               给你一个整数数组 nums 和一个整数 k ，按以下方法修改该数组：
         *               选择某个下标 i 并将 nums[i] 替换为 -nums[i] 。
         *               重复这个过程恰好 k 次。可以多次选择同一个下标 i 。
         *               以这种方式修改数组后，返回数组 可能的最大和 。
         *               贪心策略：修改绝对值小的值
         * @author: pwz
         * @date: 2022/8/1 14:12
         * @param nums
         * @param k
         * @return: int
         */
        // 将数组按照绝对值大小从大到小排序，注意要按照绝对值的大小
        nums = IntStream.of(nums)
                .boxed()
                .sorted((o1, o2) -> Math.abs(o2) - Math.abs(o1))
                .mapToInt(Integer::intValue).toArray();
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            if (nums[i] < 0 && k > 0) {
                nums[i] = -nums[i];
                k--;
            }
        }
        if (k % 2 == 1) {
            nums[len - 1] = -nums[len - 1];
        }
        return Arrays.stream(nums).sum();
    }

    // 134
    public int canCompleteCircuit(int[] gas, int[] cost) {
        /**
         * @Description: 加油站
         *               在一条环路上有 n 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
         *               你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。
         *               你从其中的一个加油站出发，开始时油箱为空。给定两个整数数组 gas 和 cost ，如果你可以绕环路
         *               行驶一周，则返回出发时加油站的编号，否则返回 -1 。如果存在解，则 保证 它是 唯一 的。
         *               贪心策略：找剩余油量大于0的点作为起点
         * @author: pwz
         * @date: 2022/8/1 14:30
         * @param gas
         * @param cost
         * @return: int
         */
        int curSum = 0;
        int start = 0;
        int totalSum = 0;
        for (int i = 0; i < gas.length; i++) {
            curSum += gas[i] - cost[i];
            totalSum += gas[i] - cost[i];
            if (curSum < 0) {
                start = i + 1;
                curSum = 0;
            }
        }
        if (totalSum < 0) {
            return -1;
        }
        return start;
    }

    // 135
    public int candy(int[] ratings) {
        /**
         * @Description: 分发糖果
         *               n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。
         *               你需要按照以下要求，给这些孩子分发糖果：
         *               1、每个孩子至少分配到 1 个糖果。
         *               2、相邻两个孩子评分更高的孩子会获得更多的糖果。
         *               计算并返回需要准备的 最少糖果数目 。
         *               贪心策略：使用两次贪心，分别考虑左右两边的情况
         * @author: pwz
         * @date: 2022/8/1 14:47
         * @param ratings
         * @return: int
         */
        int[] candyVec = new int[ratings.length];
        candyVec[0] = 1;
        for (int i = 1; i < ratings.length; i++) {
            if (ratings[i] > ratings[i - 1]) {
                candyVec[i] = candyVec[i - 1] + 1;
            } else {
                candyVec[i] = 1;
            }
        }
        for (int i = ratings.length - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1]) {
                candyVec[i] = Math.max(candyVec[i + 1] + 1, candyVec[i]);
            }
        }
        int result135 = 0;
        for (int i : candyVec) {
            result135 += i;
        }
        return result135;
    }

    // 860
    public boolean lemonadeChange(int[] bills) {
        /**
         * @Description: 找零钱
         * @author: pwz
         * @date: 2022/8/2 15:18
         * @param bills
         * @return: boolean
         */
        int five = 0, ten = 0;
        for (int i : bills) {
            if (i == 5) {
                five++;
            } else if (i == 10) {
                if (five == 0) {
                    return false;
                }
                five--;
                ten++;
            } else if (i == 20) {
                if (ten > 0 && five > 0) {
                    ten--;
                    five--;
                } else if (five >= 3) {
                    five -= 3;
                } else {
                    return false;
                }
            }
        }
        return true;
    }

    // 452
    public int findMinArrowShots(int[][] points) {
        /**
         * @Description: 用最少数量的箭引爆气球
         * @author: pwz
         * @date: 2022/8/2 15:20
         * @param points
         * @return: int
         */
        if (points.length == 0) return 0;
        Arrays.sort(points, Comparator.comparingInt(p -> p[1]));
        int minLeftBound = points[0][1];
        int count = 1;
        for (int i = 1; i <= points.length - 1; i++) {
            if (points[i][0] > minLeftBound) {
                count++;
                minLeftBound = points[i][1];
            } else {
                minLeftBound = Math.max(minLeftBound, points[i][1]);
            }
        }
        return count;
    }

    // 435
    public int eraseOverlapIntervals(int[][] intervals) {
        /**
         * @Description: 无重叠区间
         *               给定一个区间的集合 intervals ，其中 intervals[i] = [starti, endi] 。
         *               返回 需要移除区间的最小数量，使剩余区间互不重叠 。
         * @author: pwz
         * @date: 2022/8/2 16:35
         * @param intervals
         * @return: int
         */
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[1]));
        int count = 0;
        int minRightBound = intervals[0][1];
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] >= minRightBound) {
                minRightBound = intervals[i][1];
            } else {
                count++;
            }
        }
        return count;
    }

    // 763
    public List<Integer> partitionLabels(String s) {
        /**
         * @Description: 划分字母区间
         *               字符串 S 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，
         *               同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。
         * @author: pwz
         * @date: 2022/8/4 15:05
         * @param s
         * @return: java.util.List<java.lang.Integer>
         */
        int edge[] = new int[26];
        List<Integer> list = new LinkedList<>();
        char hash[] = s.toCharArray();
        for (int i = 0; i < s.length(); i++) {
            edge[hash[i] - 'a'] = i;
        }
        int index = 0;
        int start = -1;
        for (int i = 0; i < s.length(); i++) {
            index = Math.max(index, edge[hash[i] - 'a']);
            if (index == i) {
                list.add(index - start);
                start = index;
            }
        }
        return list;
    }

    // 56
    public int[][] merge(int[][] intervals) {
        /**
         * @Description: 合并区间
         *               以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。
         *               请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
         * @author: pwz
         * @date: 2022/8/4 15:06
         * @param intervals
         * @return: int[][]
         */
        List<int[]> list = new LinkedList<>();
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));
        int maxRightBound = intervals[0][1];
        int start = intervals[0][0];
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] > maxRightBound) {
                list.add(new int[]{start, maxRightBound});
                start = intervals[i][0];
                maxRightBound = intervals[i][1];
            } else {
                maxRightBound = Math.max(maxRightBound, intervals[i][1]);
            }
        }
        list.add(new int[]{start, maxRightBound});
        return list.toArray(new int[list.size()][]);
    }

    // 738
    public int monotoneIncreasingDigits(int n) {
        /**
         * @Description: 单调递增的数字
         *               当且仅当每个相邻位数上的数字 x 和 y 满足 x <= y 时，我们称这个整数是单调递增的。
         *               给定一个整数 n ，返回 小于或等于 n 的最大数字，且数字呈 单调递增 。
         * @author: pwz
         * @date: 2022/8/4 15:26
         * @param n
         * @return: int
         */
        char[] chars = String.valueOf(n).toCharArray();
        int start = chars.length;
        for (int i = chars.length - 2; i >= 0; i--) {
            if (chars[i] > chars[i + 1]) {
                chars[i]--;
                start = i + 1;
            }
        }
        for (int i = start; i < chars.length; i++) {
            chars[i] = '9';
        }
        return Integer.parseInt(String.valueOf(chars));
    }

    // 714 ||
    public int maxProfit(int[] prices, int fee) {
        /**
         * @Description: 买股票的最佳时机2（含手续费）
         * @author: pwz
         * @date: 2022/8/5 10:01
         * @param prices
         * @param fee
         * @return: int
         */
        int minPrice = prices[0] + fee;
        int profit = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > minPrice) {
                profit += prices[i] - minPrice;
                minPrice = prices[i];
            } else if (prices[i] < minPrice - fee) {
                minPrice = prices[i] + fee;
            }
        }
        return profit;
    }

    // 968 |||
    int result968 = 0;

    public int minCameraCover(TreeNode root) {
        /**
         * @Description: 监控二叉树
         *               0：无覆盖
         *               1：有摄像头
         *               2：有覆盖
         * @author: pwz
         * @date: 2022/8/5 10:20
         * @param root
         * @return: int
         */
        if (traversal968(root) == 0) {
            result968++;
        }
        return result968;
    }

    public int traversal968(TreeNode root) {
        if (root == null) return 2;
        int left = traversal968(root.left);
        int right = traversal968(root.right);
        if (left == 2 && right == 2) return 0;
        else if (left == 0 || right == 0) {
            result968++;
            return 1;
        } else return 2;
    }

    // 509 |
    public int fib(int n) {
        /**
         * @Description: 动态规划第一题，斐波那契
         *               状态转移方程：dp[i] = dp[i - 1] + dp[i - 2]
         * @author: pwz
         * @date: 2022/8/5 10:53
         * @param n
         * @return: int
         */
        if (n < 2) return n;
//        int[] dp = new int[n + 1];
//        dp[0] = 0;
        int a = 0;
//        dp[1] = 1;
        int b = 1;
        int res = 0;
        for (int i = 2; i <= n; i++) {
            res = a + b;
            a = b;
            b = res;
        }
        return res;
    }

    // 746 |
    public int minCostClimbingStairs(int[] cost) {
        /**
         * @Description: 最小花费爬楼梯
         *               状态转移方程：dp[i] = Math.min(dp[i - 1], dp[i - 2]) + cost[i];
         * @author: pwz
         * @date: 2022/8/8 9:59
         * @param cost
         * @return: int
         */
        int len = cost.length;
        int[] dp = new int[len];
        dp[0] = cost[0];
        dp[1] = cost[1];
        for (int i = 2; i < len; i++) {
            dp[i] = Math.min(dp[i - 1], dp[i - 2]) + cost[i];
        }
        return Math.min(dp[len - 1], dp[len - 2]);
    }

    // 62 ||
    public int uniquePaths(int m, int n) {
        /**
         * @Description: 不同路径
         *               一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
         *               机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
         *               问总共有多少条不同的路径？
         *               状态转移方程：dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
         * @author: pwz
         * @date: 2022/8/8 10:11
         * @param m
         * @param n
         * @return: int
         */
//        int[][] dp = new int[m][n];
//        for (int i = 0; i < m; i++) {
//            dp[i][0] = 1;
//        }
//        for (int i = 0; i < n; i++) {
//            dp[0][i] = 1;
//        }
//        for (int i = 1; i < m; i++) {
//            for (int j = 1; j < n; j++) {
//                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
//            }
//        }
//        return dp[m - 1][n - 1];
        int[] f = new int[n];
        f[0] = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (j - 1 >= 0) {
                    f[j] += f[j - 1];
                }
            }
        }
        return f[n - 1];
    }

    // 63 ||
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        /**
         * @Description: 不同路径2
         *               一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
         *               机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish”）。
         *               现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
         *               网格中的障碍物和空位置分别用 1 和 0 来表示。
         * @author: pwz
         * @date: 2022/8/8 10:19
         * @param obstacleGrid
         * @return: int
         */
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[] f = new int[n];
        f[0] = obstacleGrid[0][0] == 0 ? 1 : 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    f[j] = 0;
                    continue;
                }
                if (j - 1 >= 0 && obstacleGrid[i][j - 1] == 0) {
                    f[j] += f[j - 1];
                }
            }
        }
        return f[n - 1];
    }

    // 416 ||
    public boolean canPartition(int[] nums) {
        /**
         * @Description: 分割等和子集
         *               给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，
         *               使得两个子集的元素和相等。
         *               dp[j]：用数组中的数能凑成的最接近j的值
         *               dp[j] = Math.max(dp[j], dp[j - nums[i]] + nums[i])
         * @author: pwz
         * @date: 2022/8/9 14:36
         * @param nums
         * @return: boolean
         */
        if (nums == null || nums.length == 0) return false;
        int sum = 0;
        for (int i : nums) {
            sum += i;
        }
        if (sum % 2 == 1) return false;
        int target = sum / 2;
        int[] dp = new int[target + 1];
        for (int i = 0; i < nums.length; i++) {
            for (int j = target; j >= nums[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - nums[i]] + nums[i]);
            }
        }
        return dp[target] == target;
    }

    // 1049 ||
    public int lastStoneWeightII(int[] stones) {
        /**
         * @Description: 最后一块石头的重量2
         *               有一堆石头，用整数数组 stones 表示。其中 stones[i] 表示第 i 块石头的重量。
         *               每一回合，从中选出任意两块石头，然后将它们一起粉碎。
         *               假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：
         *               1如果 x == y，那么两块石头都会被完全粉碎；
         *               2如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
         *               最后，最多只会剩下一块 石头。返回此石头 最小的可能重量 。如果没有石头剩下，就返回 0。
         *               转化思路：分成重量最接近的两堆，即sum / 2的背包最多装多少，最后返回两堆差值（大减小）
         *               dp[j]：用数组中的数能凑成的最接近j的值
         *               dp[j] = Math.max(dp[j], dp[j - stones[i]] + stones[i])
         * @author: pwz
         * @date: 2022/8/9 15:48
         * @param stones
         * @return: int
         */
        int sum = 0;
        for (int i : stones) {
            sum += i;
        }
        int target = sum / 2;
        int[] dp = new int[target + 1];
        for (int i = 0; i < stones.length; i++) {
            for (int j = target; j >= stones[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - stones[i]] + stones[i]);
            }
        }
        return sum - 2 * dp[target];
    }

    // 474 ||
    public int findMaxForm(String[] strs, int m, int n) {
        /**
         * @Description: 一和零
         *               给你一个二进制字符串数组 strs 和两个整数 m 和 n 。
         *               请你找出并返回 strs 的最大子集的长度，该子集中 最多 有 m 个 0 和 n 个 1 。
         *               dp[i][j]：子集中最多有i个0, j个1的最大子集长度
         *               dp[i][j] = Math.max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1)
         * @author: pwz
         * @date: 2022/8/9 16:08
         * @param strs
         * @param m
         * @param n
         * @return: int
         */
        int[][] dp = new int[m + 1][n + 1];
        for (String string : strs) {
            int zeroNum = 0;
            int oneNum = 0;
            for (char ch : string.toCharArray()) {
                if (ch == '0') zeroNum++;
                else oneNum++;
            }
            for (int i = m; i >= zeroNum; i--) {
                for (int j = n; j >= oneNum; j--) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1);
                }
            }
        }
        return dp[m][n];
    }

    // 322 ||
    public int coinChange(int[] coins, int amount) {
        /**
         * @Description: 零钱兑换
         *               给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
         *               计算并返回可以凑成总金额所需的最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
         *               你可以认为每种硬币的数量是无限的。
         *               dp[j]：凑足金额j所需的最少硬币数
         *               dp[j] = Math.min(dp[j], dp[j - coins[i]] + 1)
         * @author: pwz
         * @date: 2022/8/10 9:35
         * @param coins
         * @param amount
         * @return: int
         */
        int[] dp = new int[amount + 1];
        dp[0] = 0;
        for (int i = 1; i < dp.length; i++) {
            dp[i] = Integer.MAX_VALUE;
        }
        for (int i = 0; i < coins.length; i++) {
            for (int j = coins[i]; j <= amount; j++) {
                if (dp[j - coins[i]] != Integer.MAX_VALUE) {
                    dp[j] = Math.min(dp[j], dp[j - coins[i]] + 1);
                }
            }
        }
        return dp[amount] == Integer.MAX_VALUE ? -1 : dp[amount];
    }

    // 518 ||
    public int change(int amount, int[] coins) {
        /**
         * @Description: 零钱兑换2
         *               给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。
         *               请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。
         *               假设每一种面额的硬币有无限个。
         *               dp[j]：凑成金额j的硬币组合数
         *               dp[j] += dp[j - coins[i]]
         * @author: pwz
         * @date: 2022/8/10 10:07
         * @param amount
         * @param coins
         * @return: int
         */
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int i = 0; i < coins.length; i++) {
            for (int j = coins[i]; j <= amount; j++) {
                dp[j] += dp[j - coins[i]];
            }
        }
        return dp[amount];
    }

    // 377
    public int combinationSum4(int[] nums, int target) {
        /**
         * @Description: 组合总和4
         *               给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。
         *               请你从 nums 中找出并返回总和为 target 的元素组合的个数。
         *               dp[i]：总和为j的元素组合个数
         * @author: pwz
         * @date: 2022/8/10 11:03
         */
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 0; i <= target; i++) {
            for (int j = 0; j < nums.length; j++) {
                if (i >= nums[j]) {
                    dp[i] += dp[i - nums[j]];
                }
            }
        }
        return dp[target];
    }

    /**
     * 动归五部曲：
     * 1、确定dp数组以及下标的含义
     * 2、确定递推公式
     * 求装满背包有几种方法，一般公式都是：dp[j] += dp[j - nums[i]]
     * 3、dp数组如何初始化
     * 4、确定遍历顺序
     * 01背包问题和纯完全背包问题在二维dp中for循环的先后循环是可以颠倒的。
     * 01背包问题一维dp的遍历顺序不可颠倒，物品外循环，背包内循环，且内循环从大到小（为了保证每个物品仅被添加一次。）。
     * 纯完全背包问题一维dp的中for循环的先后循环是可以颠倒的，内循环从小到大（因为物品可以重复使用）
     * 完全背包求排列组合问题中：求组合先遍历物品、后遍历背包；求排列先遍历背包，后遍历物品。
     * 5、举例推导dp数组
     */

    // 279 ||
    public int numSquares(int n) {
        /**
         * @Description: 完全平方数
         *               给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。
         *               例输入：n = 12 输出：3 解释：12 = 4 + 4 + 4
         * @author: pwz
         * @date: 2022/8/12 9:49
         */
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            dp[i] = Integer.MAX_VALUE;
        }
        for (int i = 0; i <= n; i++) {
            for (int j = 1; j * j <= i; j++) {
                dp[i] = Math.min(dp[i], dp[i - j * j]);
            }
        }
        return dp[n];
    }

    // 139 ||
    public boolean wordBreak(String s, List<String> wordDict) {
        /**
         * @Description: 单词拆分
         *               给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
         *               dp[i]：true表示长度为i的字符串可由字典中单词组成
         * @author: pwz
         * @date: 2022/8/12 10:03
         * @param s
         * @param wordDict
         * @return: boolean
         */
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (wordDict.contains(s.substring(j, i)) && dp[j]) {
                    dp[i] = true;
                }
            }
        }
        return dp[s.length()];
    }

    /**
     * @param nums
     * @Description: 198||打家劫舍
     * 你是小偷计划偷沿街的房屋，相邻的房屋装有相互连通的防盗系统，
     * 如果相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
     * 给定一个代表每个房屋存放金额的非负整数数组，计算在不触动警报情况下 ，今晚能够偷窃到的最高金额。
     * dp[i]：包括下标i在内的房屋，所能盗窃的最大金额
     * dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i])
     * @author: pwz
     * @date: 2022/8/12 10:35
     * @return: int
     */
    public int rob1(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[nums.length - 1];
    }

    /**
     * @param nums
     * @Description: 213||打家劫舍2
     * 所有的房屋都围成一圈，同时，相邻的房屋装有相互连通的防盗系统，
     * 如果相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
     * 给定一个代表每个房屋存放金额的非负整数数组，计算在不触动警报情况下 ，今晚能够偷窃到的最高金额。
     * @author: pwz
     * @date: 2022/8/12 10:26
     * @return: int
     */
    public int rob2(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int len = nums.length;
        if (len == 1) return nums[0];
        return Math.max(rob(nums, 0, len - 1), rob(nums, 1, len));
    }

    public int rob(int[] nums, int start, int end) {
        int prePre = 0, pre = 0, max = 0;
        for (int i = start; i < end; i++) {
            pre = max;
            max = Math.max(pre, prePre + nums[i]);
            prePre = pre;
        }
        return max;
    }

    // 337 ||
    public int rob3(TreeNode root) {
        /**
         * @Description: 打家劫舍2 (树形dp)
         *               如果 两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。
         *               给定二叉树的 root 。返回 在不触动警报的情况下 ，小偷能够盗取的最高金额 。
         *               dp[2]：长度为2的dp数组
         *               dp[0]表示不抢当前节点得到的最大值
         *               dp[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
         *               dp[1]表示抢当前节点得到的最大值
         *               dp[1] = root.val + left[0] + right[0];
         * @author: pwz
         * @date: 2022/8/13 12:46
         * @param root
         * @return: int
         */
        int[] res = rob(root);
        return Math.max(res[0], res[1]);
    }

    public int[] rob(TreeNode root) {
        int[] dp = new int[2];
        if (root == null) return dp;
        int[] left = rob(root.left);
        int[] right = rob(root.right);
        dp[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        dp[1] = root.val + left[0] + right[0];
        return dp;
    }

    // 123 |||
    public int maxProfit3(int[] prices) {
        /**
         * @Description: 买股票的最大时机3
         *                  定义 5 种状态:
         *                  0: 没有操作, 1: 第一次买入, 2: 第一次卖出, 3: 第二次买入, 4: 第二次卖出
         *                  没有操作不记录，定义dp[][4]
         *                  dp[i][j]：第i天j状态下的最大现金
         * @author: pwz
         * @date: 2022/8/13 13:24
         * @param prices
         * @return: int
         */
        int[][] dp = new int[prices.length][4];
        dp[0][0] = -prices[0];
        dp[0][2] = -prices[0];
        for (int i = 1; i < prices.length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], -prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
            dp[i][2] = Math.max(dp[i - 1][2], dp[i - 1][1] - prices[i]);
            dp[i][3] = Math.max(dp[i - 1][3], dp[i - 1][2] + prices[i]);
        }
        return dp[prices.length - 1][3];
    }

    // 188 |||
    public int maxProfit4(int k, int[] prices) {
        /**
         * @description: 卖股票的最佳时期4
         *               给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。
         *               设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
         * @author: pwz
         * @date: 2022/8/15 10:17
         * @param: [k, prices]
         * @return: int
         **/
        if (prices.length == 0) return 0;
        int[][] dp = new int[prices.length][2 * k + 1];
        for (int i = 1; i < 2 * k; i += 2) {
            dp[0][i] = -prices[0];
        }
        for (int i = 1; i < prices.length; i++) {
            for (int j = 0; j < 2 * k - 1; j += 2) {
                dp[i][j + 1] = Math.max(dp[i - 1][j + 1], dp[i - 1][j] - prices[i]);
                dp[i][j + 2] = Math.max(dp[i - 1][j + 2], dp[i - 1][j + 1] + prices[i]);
            }
        }
        return dp[prices.length - 1][2 * k];
    }

    // 309 ||
    public int maxProfit5(int[] prices) {
        /**
         * @description: 卖股票的最佳时期5
         *               卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
         *               定义 4 种状态:
         *               0: 保持买入状态  1: 保持卖出状态（过了冷冻期）  2: 卖出状态（今天卖出）  3: 处在冷冻期
         * @author: pwz
         * @date: 2022/8/15 11:00
         * @param: [prices]
         * @return: int
         **/
        int[][] dp = new int[prices.length][4];
        dp[0][0] = -prices[0];
        for (int i = 1; i < prices.length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], Math.max(dp[i - 1][3], dp[i - 1][1]) - prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][3]);
            dp[i][2] = dp[i - 1][0] + prices[i];
            dp[i][3] = dp[i - 1][2];
        }
        return Math.max(dp[prices.length - 1][3], Math.max(dp[prices.length - 1][1], dp[prices.length - 1][2]));
    }

    /**
     * @description: 300 || 最长递增子序列
     * 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
     * 子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。
     * 例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
     * @author: pwz
     * @date: 2022/8/15 13:58
     * @param: [nums]
     * @return: int
     */
    public int lengthOfLIS(int[] nums) {

        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }
        int res = 0;
        for (int i : dp) {
            res = Math.max(res, i);
        }
        return res;
    }

    /**
     * @param nums
     * @return int
     * @Description: 674 | 最长连续递增序列
     * 给定一个未经排序的整数数组，找到最长且连续递增的子序列，并返回该序列的长度。
     * dp[i]：表示以下标i结尾的最长子序列
     * @author pwz
     * @date 2022/8/16 15:43
     */
    public int findLengthOfLCIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > nums[i - 1]) {
                dp[i] = dp[i - 1] + 1;
            }
        }
        int res = 1;
        for (int i : dp) {
            res = i > res ? i : res;
        }
        return res;
    }

    /**
     * @param nums1
     * @param nums2
     * @return int
     * @Description: 718 || 最长重复子数组
     * 给两个整数数组 nums1 和 nums2 ，返回 两个数组中 公共的 、长度最长的子数组的长度 。
     * dp[i][j]：nums1以i结尾的子数组和nums2以j结尾的子数组的最长公共子数组
     * @author pwz
     * @date 2022/8/16 15:57
     */
    public int findLength(int[] nums1, int[] nums2) {
        int res = 0;
        int[][] dp = new int[nums1.length + 1][nums2.length + 1];
        for (int i = 1; i <= nums1.length; i++) {
            for (int j = 1; j <= nums2.length; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }
                if (dp[i][j] > res) res = dp[i][j];
            }
        }
        return res;
    }

    /**
     * @param text1
     * @param text2
     * @return int
     * @Description: 1143 || 最长公共子序列
     * 给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在公共子序列 ，返回 0 。
     * 字符串的子序列：由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
     * 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
     * @author pwz
     * @date 2022/8/16 17:06
     */
    public int longestCommonSubsequence(String text1, String text2) {
        int[][] dp = new int[text1.length() + 1][text2.length() + 1];
        int res = 0;
        for (int i = 1; i <= text1.length(); i++) {
            for (int j = 1; j <= text2.length(); j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
                if (dp[i][j] > res) res = dp[i][j];
            }
        }
        return res;
    }

    /**
     * @param nums1
     * @param nums2
     * @return int
     * @Description: 1035 || 不相交的线=求最长公共子序列
     * @author pwz
     * @date 2022/8/17 14:47
     */
    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        int[][] dp = new int[nums1.length + 1][nums2.length + 1];
        int res = 0;
        for (int i = 1; i <= nums1.length; i++) {
            for (int j = 1; j <= nums2.length; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
                if (dp[i][j] > res) res = dp[i][j];
            }
        }
        return res;
    }

    /**
     * @param s
     * @param t
     * @return boolean
     * @Description: 392 | 判断子序列
     * 给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
     * 求最长公共子序列的长度与s的长度比较
     * @author pwz
     * @date 2022/8/17 18:53
     */
    public boolean isSubsequence(String s, String t) {
//        int[][] dp = new int[s.length() + 1][t.length() + 1];
//        int res = 0;
//        for (int i = 1; i <= s.length(); i++) {
//            for (int j = 1; j <= t.length(); j++) {
//                if (s.charAt(i - 1) == t.charAt(j - 1)) {
//                    dp[i][j] = dp[i - 1][j - 1] + 1;
//                } else {
//                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
//                }
//                if (dp[i][j] > res) res = dp[i][j];
//            }
//        }
//        return res == s.length();
        int res = longestCommonSubsequence(s, t);   // 求最长公共子序列
        return res == s.length();
    }

    /**
     * @param s
     * @param t
     * @return int
     * @Description: 115 ||| 不同的子序列
     * 给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。
     * @author pwz
     * @date 2022/8/25 12:39
     */
    public int numDistinct(String s, String t) {
        int[][] dp = new int[s.length() + 1][t.length() + 1];
        for (int i = 0; i < s.length() + 1; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 1; j <= t.length(); j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[s.length()][t.length()];
    }

    /**
     * @param word1
     * @param word2
     * @return int
     * @Description: 583 || 两个字符串的删除操作
     * 给定两个单词 word1 和 word2 ，返回使得 word1 和  word2 相同所需的最小步数。
     * 每步 可以删除任意一个字符串中的一个字符。
     * @author pwz
     * @date 2022/8/26 9:58
     */
    public int minDistance(String word1, String word2) {
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        for (int i = 1; i <= word1.length(); i++) {
            dp[i][0] = i;
        }
        for (int i = 1; i <= word2.length(); i++) {
            dp[0][i] = i;
        }
        for (int i = 1; i < word1.length() + 1; i++) {
            for (int j = 1; j < word2.length() + 1; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1] + 2, Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1));
                }
            }
        }
        return dp[word1.length()][word2.length()];
    }

    /**
     * @param word1
     * @param word2
     * @return int
     * @Description: 72 ||| 编辑距离
     * 给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。增删改操作
     * @author pwz
     * @date 2022/8/26 10:31
     */
    public int minDistance1(String word1, String word2) {
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        for (int i = 1; i <= word1.length(); i++) {
            dp[i][0] = i;
        }
        for (int i = 1; i <= word2.length(); i++) {
            dp[0][i] = i;
        }
        for (int i = 1; i < word1.length() + 1; i++) {
            for (int j = 1; j < word2.length() + 1; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1] + 1, Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1));
                }
            }
        }
        return dp[word1.length()][word2.length()];
    }

    /**
     * @param s
     * @return int
     * @Description: 647 || 回文子串
     * 给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。
     * dp[i][j]:区间i,j（左闭右闭）是否为回文串
     * @author pwz
     * @date 2022/8/26 10:34
     */
    public int countSubstrings(String s) {
        boolean[][] dp = new boolean[s.length()][s.length()];
        int result = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            for (int j = i; j < s.length(); j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    if (j - i <= 1) {
                        dp[i][j] = true;
                        result++;
                    } else if (dp[i + 1][j - 1]) {
                        dp[i][j] = true;
                        result++;
                    }
                }
            }
        }
        return result;
    }

    /**
     * @param s
     * @return int
     * @Description: 516 || 最长回文子串
     * 给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。
     * @author pwz
     * @date 2022/8/29 16:52
     */
    public int longestPalindromeSubSeq(String s) {
        int[][] dp = new int[s.length()][s.length()];
        for (int i = s.length() - 1; i >= 0; i--) {
            dp[i][i] = 1;
            for (int j = i + 1; j < s.length(); j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][s.length() - 1];
    }

    /**
     * @param nums1
     * @param nums2
     * @return int[]
     * @Description: 496 | 下一个更大元素
     * nums1 中数字 x 的 下一个更大元素 是指 x 在 nums2 中对应位置 右侧 的 第一个 比 x 大的元素。
     * @author pwz
     * @date 2022/8/30 10:41
     */
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        int[] res = new int[nums1.length];
        Arrays.fill(res, -1);
        Stack<Integer> stack = new Stack<>();
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums1.length; i++) {
            map.put(nums1[i], i);
        }
        stack.add(0);
        for (int i = 1; i < nums2.length; i++) {
            if (nums2[i] > nums2[stack.peek()]) {
                while (!stack.isEmpty() && nums2[stack.peek()] < nums2[i]) {
                    if (map.containsKey(nums2[stack.peek()])) {
                        res[map.get(nums2[stack.peek()])] = nums2[i];
                    }
                    stack.pop();
                }
            }
            stack.add(i);
        }
        return res;
    }

    /**
     * @param nums
     * @return int[]
     * @Description: 503 || 下一个更大元素2
     * @author pwz
     * @date 2022/8/30 11:15
     */
    public int[] nextGreaterElements(int[] nums) {
        int[] res = new int[nums.length];
        Arrays.fill(res, -1);
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < nums.length * 2; i++) {
            while (!stack.isEmpty() && nums[stack.peek()] < nums[i % nums.length]) {
                res[stack.peek()] = nums[i % nums.length];
                stack.pop();
            }
            stack.add(i % nums.length);
        }
        return res;
    }

    /**
     * @param height
     * @return int
     * @Description: 42 ||| 接雨水
     * 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
     * @author pwz
     * @date 2022/8/30 14:58
     */
    public int trap(int[] height) {
        Stack<Integer> stack = new Stack<>();
        stack.add(0);
        int sum = 0;
        for (int i = 1; i < height.length; i++) {
            if (height[i] < height[stack.peek()]) {
                stack.add(i);
            } else if (height[i] == height[stack.peek()]) {
                stack.pop();
                stack.add(i);
            } else {
                while (!stack.isEmpty() && height[i] > height[stack.peek()]) {
                    int mid = stack.pop();
                    if (!stack.isEmpty()) {
                        int hold = (i - stack.peek() - 1) *
                                (Math.min(height[stack.peek()], height[i]) - height[mid]);
                        sum += hold > 0 ? hold : 0;
                    }
                }
                stack.add(i);
            }
        }
        return sum;
    }

    /**
     * @param heights
     * @return int
     * @Description: 84 ||| 柱状图中最大的矩形
     * 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1
     * 求在该柱状图中，能够勾勒出来的矩形的最大面积。
     * @author pwz
     * @date 2022/8/30 15:02
     */
    public int largestRectangleArea(int[] heights) {
        int[] newHeights = new int[heights.length + 2];
        newHeights[0] = newHeights[newHeights.length - 1] = 0;
        for (int i = 1; i <= heights.length; i++) {
            newHeights[i] = heights[i - 1];
        }
        heights = newHeights;
        Stack<Integer> stack = new Stack<>();
        stack.push(0);
        int res = 0;
        for (int i = 1; i < heights.length; i++) {
            if (heights[i] > heights[stack.peek()]) {
                stack.push(i);
            } else if (heights[i] == heights[stack.peek()]) {
                stack.pop();
                stack.push(i);
            } else {
                while (heights[i] < heights[stack.peek()]) {
                    res = Math.max(heights[stack.pop()] * (i - stack.peek() - 1), res);
                }
                stack.push(i);
            }
        }
        return res;
    }


    public static void main(String[] args) {
        int[] next = new int[10];
        CodeCapricorns solution = new CodeCapricorns();
        int[][] ints = new int[][]{
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0}
        };
        int[] test416 = new int[]{1, 5, 11, 5};
        solution.coinChange(new int[]{1, 2, 5}, 11);
        solution.change(5, new int[]{1, 2, 5});
        System.out.println(solution.longestPalindromeSubSeq("bbbab"));
//        solution.canPartition(test416);
//        solution.uniquePaths(3, 7);
//        solution.uniquePathsWithObstacles(ints);
//        Lock lock = new ReentrantLock();
//        Arrays.asList(1, 2, 3, 4).stream().sorted().forEach(System.out::println);
//        System.out.println(solution.repeatedSubstringPattern("abcabc"));
//        System.out.println(Arrays.toString(solution.maxSlidingWindow(new int[]{1, 3, 3, 3}, 2)));
//        System.out.println(solution.subsetsWithDup(new int[]{1, 2, 3}));
//        System.out.println(solution.findSubsequences(new int[]{4, 7, 6, 7}));
    }


}