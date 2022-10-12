package com.leetcode;

import com.utils.ListNode;
import com.utils.Node;
import com.utils.TreeNode;

import java.util.*;

/**
 * @ClassName Hot100
 * @Author PanWZ
 * @Data 2022/3/12 - 11:12
 * @Version: 1.8
 */
public class LCodeSolution {
    TreeNode ans236 = new TreeNode();

    // 21
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode list3 = new ListNode();
        ListNode index = list3;
        while (list1 != null && list2 != null) {
            if (list1.val < list2.val) {
                index.next = list1;
                list1 = list1.next;
            } else {
                index.next = list2;
                list2 = list2.next;
            }
            index = index.next;
        }
        index.next = list1 == null ? list2 : list1;
        return list3.next;
    }

    // 70
    public int climbStairs(int n) {
        if (n <= 2) return n;
        int a = 1, b = 2, tem = 0;
        for (int i = 3; i <= n; i++) {
            tem = a + b;
            a = b;
            b = tem;
        }
        return tem;
    }

    // 461
    public int hammingDistance(int x, int y) {
        int z = x ^ y, res = 0;
        while (z != 0) {
            res += z & 1;
            z >>= 1;
        }
        return res;
    }

    // 226
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        return root;
    }

    // 617
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null) return root2;
        if (root2 == null) return root1;
        TreeNode merge = new TreeNode(root1.val + root2.val);
        merge.left = mergeTrees(root1.left, root2.left);
        merge.right = mergeTrees(root1.right, root2.right);
        return merge;
    }

    // 338
    public int[] countBits(int n) {
        int res[] = new int[n + 1];
        for (int i = 0;i <= n;i++) {
            res[i] = Integer.bitCount(i);
        }
        return res;
    }

    // 104 & offer 55-1
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    // 448
    public List<Integer> findDisappearedNumbers(int[] nums) {
        for (int i = 0;i < nums.length;i++) {
            nums[Math.abs(nums[i]) - 1] = -Math.abs(nums[Math.abs(nums[i]) - 1]);
        }
        ArrayList<Integer> res = new ArrayList<Integer>();
        for (int i = 0;i < nums.length;i++) {
            if (nums[i] > 0) res.add(i + 1);
        }
        return res;
    }

    // 169
    public int majorityElement(int[] nums) {
//        Arrays.sort(nums);
//        return nums[nums.length / 2];
        /**
         *摩尔投票法
         */
        int res = nums[0], count = 1;
        for (int i = 1;i < nums.length;i++) {
            if (nums[i] != res) {
                if (--count == 0) {
                    res = nums[i];
                    count = 1;
                }
            } else {
                count++;
            }
        }
        return res;
    }

    // 136
    public int singleNumber(int[] nums) {
        int res = 0;
        for (int n : nums)
            res ^= n;
        return res;
    }

    // 283
    public void moveZeroes(int[] nums) {
        int pre = 0;
        for (int i = 0;i < nums.length;i++)
            if (nums[i] != 0) {
                nums[pre] = nums[i];
                pre++;
            }
        for (int i = pre;i < nums.length;i++) {
            nums[i] = 0;
        }
    }

    // 94
    public List<Integer> inorderTraversal(TreeNode root) {
        ArrayList<Integer> res = new ArrayList<Integer>();
        inorder(res, root);
        return res;
    }
    void inorder(List<Integer> res, TreeNode root) {
        if (root == null) return;
        inorder(res, root.left);
        res.add(root.val);
        inorder(res, root.right);
    }

    // 160
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

    // 121
    public int maxProfit(int[] prices) {
        int max_profit = 0, min_price = Integer.MAX_VALUE;
        for (int i = 0;i < prices.length;i++) {
            if (prices[i] < min_price) {
                min_price = prices[i];
            } else if (prices[i] - min_price > max_profit) {
                max_profit = prices[i] - min_price;
            }
        }
        return max_profit;
    }

    // 101 & offer 28
    public boolean isSymmetric(TreeNode root) {
        return check(root, root);
    }
    boolean check(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;
        return p.val == q.val && check(p.left, q.right) && check(p.right, q.left);
    }

    // 543
    int res543 = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        deep(root);
        return res543 - 1;
    }
    int deep(TreeNode root) {
        if (root == null) return 0;
        int L = deep(root.left);
        int R = deep(root.right);
        res543 = Math.max(res543, L + R + 1);
        return Math.max(L, R) + 1;
    }

    // 53
    public int maxSubArray(int[] nums) {
        int index = 0, max = nums[0];
        for (int i = 0;i < nums.length;i++) {
            index = Math.max(index + nums[i], nums[i]);
            max = Math.max(index, max);
        }
        return max;
    }

    // 141
    public boolean hasCycle(ListNode head) {
        /**
         * 环形链表1
         */
        HashSet<ListNode> set = new HashSet<>();
        if (head == null) return false;
        while (head != null) {
            if (!set.add(head)) return true;
            head = head.next;
        }
        return false;
    }

    // 142
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
        if (head == null) return null;
        ListNode fast = head, slow = head;
        while (fast != null) {
            slow = slow.next;
            if (fast.next != null) {
                fast = fast.next.next;
            } else {
                return null;
            }
            if (fast == slow) {
                ListNode ans = head;
                while (ans != slow) {
                    ans = ans.next;
                    slow = slow.next;
                }
                return ans;
            }
        }
        return null;
    }

    // 234
    public boolean isPalindrome(ListNode head) {
        ArrayList<Integer> list = new ArrayList<Integer>();
        while (head != null) {
            list.add(head.val);
            head = head.next;
        }
        int low = 0, height = list.size() - 1;
        while (low < height) {
            if (list.get(low) != list.get(height)) return false;
            height--;
            low++;
        }
        return true;
    }

    // 78
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        res.add(new ArrayList<Integer>());
        for (int i = 0; i < nums.length; i++) {
            int all = res.size();
            for (int j = 0; j < all; j++) {
                List<Integer> tmp = new ArrayList<Integer>(res.get(j));
                tmp.add(nums[i]);
                res.add(tmp);
            }
        }
        return res;
    }

    // 46
    public List<List<Integer>> permute(int[] nums) {
        /**
         * 多回来看看  回溯实现全排列
         */
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> output = new ArrayList<Integer>();
        for (int num : nums) {
            output.add(num);
        }
        int n = nums.length;
        backtrack(n, output, res, 0);
        return res;
    }
    public void backtrack(int n, List<Integer> output, List<List<Integer>> res, int first) {
        if (first == n) {
            res.add(new ArrayList<Integer>(output));
        }
        for (int i = first; i < n; i++) {
            Collections.swap(output, first, i);
            backtrack(n, output, res, first + 1);
            Collections.swap(output, first, i);
        }
    }

    // 22
    ArrayList<String> res = new ArrayList<>();
    public List<String> generateParenthesis(int n) {
        generate("", n, n);
        return res;
    }
    void generate(String string, int left, int right) {
        if (left == 0 && right == 0) {
            res.add(string);
            return;
        }
        if (left == right) {
            generate(string + '(', left - 1, right);
        } else {
            if (left > 0) generate(string + '(', left - 1, right);
            generate(string + ')', left, right - 1);
        }
    }

    // 406
    public int[][] reconstructQueue(int[][] people) {
        /**
         * 常回来看看  匿名函数  Arrays.sort
         */
        Arrays.sort(people, (int[] a, int[] b) -> (a[0] == b[0] ? a[1] - b[1] : b[0] - a[0]));
        List<int[]> list = new ArrayList<>();
        for (int[] i : people) {
            list.add(i[1], i);
        }
        return list.toArray(new int[0][2]);
    }

    // 48
    public void rotate(int[][] matrix) {
        /**
         * 矩阵原地旋转90度
         * 1、两次翻转
         * 2、循环
         */
        int tem;
        for (int i = 0;i < matrix.length / 2;i++) {
            for (int j = 0;j < matrix[0].length;j++) {
                tem = matrix[i][j];
                matrix[i][j] = matrix[matrix.length - i - 1][j];
                matrix[matrix.length - i - 1][j] = tem;
            }
        }
        for (int i = 1;i < matrix.length;i++) {
            for (int j = 0;j < i;j++) {
                tem = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tem;
            }
        }
    }

    // 2
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null || l2 == null) return l1 == null ? l2 : l1;
        ListNode listNode = new ListNode();
        ListNode p = listNode;
        int carry = 0;
        while (l1 != null || l2 != null || carry > 0) {
            int n1 = l1 != null ? l1.val : 0;
            int n2 = l2 != null ? l2.val : 0;
            int sum = n1 + n2 + carry;
            p.next = new ListNode(sum % 10);
            carry = sum / 10;
            l1 = l1 != null ? l1.next : null;
            l2 = l2 != null ? l2.next : null;
            p = p.next;
        }
        return listNode.next;
    }

    // 238
    public int[] productExceptSelf(int[] nums) {
        int length = nums.length;
        int[] ans = new int[length];
        ans[0] = 1;
        for (int i = 1;i < length;i++) {
            ans[i] = nums[i - 1] * ans[i - 1];
        }
        int R = 1;
        for (int i = length - 1;i >= 0;i--) {
            ans[i] = ans[i] * R;
            R = R * nums[i];
        }
        return ans;
    }

    // 39
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        /**
         * 回溯
         */
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> combine = new ArrayList<>();
        dfs(candidates, target, ans, combine, 0);
        return ans;
    }
    void dfs(int[] candidates, int target, List<List<Integer>> ans, List<Integer> combine, int index) {
        if (index == candidates.length) return;
        if (target == 0) {
            ans.add(new ArrayList<>(combine));
            return;
        }
        dfs(candidates, target, ans, combine, index + 1);
        if (target - candidates[index] >= 0) {
            combine.add(candidates[index]);
            dfs(candidates, target - candidates[index], ans, combine, index);
            combine.remove(combine.size() - 1);
        }
    }

    // 17
    public List<String> letterCombinations(String digits) {
        /**
         * 回溯
         */
        List<String> ans = new ArrayList<>();
        if (digits.length() == 0) return ans;
        Map<Character, String> map = new HashMap<Character, String>() {{
            put('2', "abc");
            put('3', "def");
            put('4', "ghi");
            put('5', "jkl");
            put('6', "mno");
            put('7', "pqrs");
            put('8', "tuv");
            put('9', "wxyz");
        }};
        letterCombination(ans, map, digits, 0, new StringBuilder());
        return ans;
    }
    void letterCombination(List<String> ans, Map<Character, String> map,
                           String digits, int index, StringBuilder combine) {
        if (index == digits.length()) ans.add(combine.toString());
        else {
            char ch = digits.charAt(index);
            String string = map.get(ch);
            int length = string.length();
            for (int i = 0;i < length;i++) {
                combine.append(string.charAt(i));
                letterCombination(ans, map, digits, index + 1, combine);
                combine.deleteCharAt(index);
            }
        }
    }

    // 494
    int ans = 0;
    public int findTargetSumWays(int[] nums, int target) {
        /**
         * 回溯时间待优化， 下次试试动态规划
         */
        findTarget(nums, target,0, 0);
        return ans;
    }
    void findTarget(int[] nums, int target, int index, int sum) {
        if (index == nums.length){
            if (sum == target)
                ans++;
        } else {
            findTarget(nums, target, index + 1, sum + nums[index]);
            findTarget(nums, target, index + 1, sum - nums[index]);
        }
    }

    // 79
    public boolean exist(char[][] board, String word) {
        /**
         * 棋盘回溯 常回来看看
         */
        int h = board.length, l = board[0].length;
        boolean[][] visited = new boolean[h][l];
        for (int i = 0;i < h;i++) {
            for (int j = 0;j < l;j++) {
                boolean flag = existWord(board, visited, i, j, word, 0);
                if (flag) return true;
            }
        }
        return false;
    }
    boolean existWord(char[][] board, boolean[][] visited, int i, int j, String word, int k) {
        if (board[i][j] != word.charAt(k)) return false;
        else if (k == word.length() - 1) return true;
        visited[i][j] = true;
        boolean ans = false;
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int[] dir : directions) {
            int newI = i + dir[0], newJ = j + dir[1];
            if (newI >= 0 && newI < board.length && newJ >= 0 && newJ < board[0].length) {
                if (!visited[newI][newJ]) {
                    boolean flag = existWord(board, visited, newI, newJ, word, k + 1);
                    if (flag) {
                        ans = true;
                        break;
                    }
                }
            }
        }
        visited[i][j] = false;
        return ans;
    }

    // 114
    public void flatten(TreeNode root) {
        List<TreeNode> list = new ArrayList<>();
        dfs(list, root);
        int length = list.size();
        for (int i = 1;i < length;i++) {
            TreeNode pre = list.get(i - 1), cur = list.get(i);
            pre.left = null;
            pre.right = cur;
        }
    }
    void dfs(List<TreeNode> arrayList, TreeNode treeNode) {
        if (treeNode == null) return;
        arrayList.add(treeNode);
        dfs(arrayList, treeNode.left);
        dfs(arrayList, treeNode.right);
    }

    // 739
    public int[] dailyTemperatures(int[] temperatures) {
        /**
         * 单调栈  常回看   输出数组  Arrays.toString
         */
        int length = temperatures.length;
        int[] ans = new int[length];
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0;i < length;i++) {
            int temperature = temperatures[i];
            while (!stack.isEmpty() && temperature > temperatures[stack.peek()]) {
                int pre = stack.pop();
                ans[pre] = i - pre;
            }
            stack.push(i);
        }
        return ans;
    }

    // 105
    Map<Integer, Integer> indexMap;
    public TreeNode buildTree(int[] pre_order, int[] in_order) {
        /**
         * 递归  继续练
         */
        indexMap = new HashMap<Integer, Integer>();
        int n = pre_order.length;
        for (int i = 0;i < n;i++) {
            indexMap.put(in_order[i], i);
        }
        return myBuildTree(pre_order, in_order, 0, n - 1, 0, n - 1);
    }
    TreeNode myBuildTree(int[] pre, int[] in, int pre_left, int pre_right, int in_left, int in_right) {
        if (pre_left > pre_right) return null;
        int pre_order_root = pre_left;
        int inorder_root = indexMap.get(pre[pre_order_root]);
        TreeNode root = new TreeNode(pre[pre_order_root]);
        int size_left_tree = inorder_root - in_left;
        root.left = myBuildTree(pre, in, pre_left + 1, pre_left + size_left_tree,
                in_left, inorder_root - 1);
        root.right = myBuildTree(pre, in, pre_left + 1 + size_left_tree, pre_right,
                inorder_root + 1, in_right);
        return root;
    }

    // 538
    int sum = 0;
    public TreeNode convertBST(TreeNode root) {
        /**
         * 反序中序遍历  右根左
         */
        if (root != null) {
            convertBST(root.right);
            sum += root.val;
            root.val = sum;
            convertBST(root.left);
        }
        return root;
    }

    // 49
    public List<List<String>> groupAnagrams(String[] str) {
        Map<String, List<String>> map = new HashMap<>();
        for (String string : str) {
            char[] chars = string.toCharArray();
            Arrays.sort(chars);
            String key = new String(chars);
            List<String> list = map.getOrDefault(key, new ArrayList<>());
            list.add(string);
            map.put(key, list);
        }
        return new ArrayList<>(map.values());
    }

    // 148
    public ListNode sortList(ListNode head) {
        /**
         * 请用归并排序实现常数级空间复杂度
         */
        ListNode pre = head;
        int n = 0;
        while (pre != null) {
            n++;
            pre = pre.next;
        }
        int[] arr = new int[n];
        pre = head;
        for (int i = 0;i < n;i++) {
            arr[i] = pre.val;
            pre = pre.next;
        }
        Arrays.sort(arr);
        pre = head;
        for (int i = 0;i < n;i++) {
            pre.val = arr[i];
            pre = pre.next;
        }
        return head;
    }

    // 1
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
        for (int i = 0;i < nums.length;i++) {
            if (hashMap.containsKey(target - nums[i]))
                return new int[] {hashMap.get(target - nums[i]), i};
            hashMap.put(nums[i], i);
        }
        return new int[0];
    }

    // 15
    public List<List<Integer>> threeSum(int[] nums) {
        /**
         * 三数之和  梦开始的地方！
         */
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        int n = nums.length;
        for (int first = 0;first < n - 2;first++) {
            if (first > 0 && nums[first] == nums[first - 1]) continue;
            int target = -nums[first];
            int third = n - 1;
            for (int second = first + 1;second < n - 1;second++) {
                if (second > first + 1 && nums[second] == nums[second - 1]) continue;
                while (second < third && nums[second] + nums[third] > target) --third;
                if (second == third) break;
                if (nums[second] + nums[third] == target){
                    ArrayList<Integer> list = new ArrayList<>();
                    list.add(nums[first]);
                    list.add(nums[second]);
                    list.add(nums[third]);
                    ans.add(list);
                }
            }
        }
        return ans;
    }

    // 96
    public int numTrees(int n) {
        /**
         * 动态规划  1-n依次做根节点，左右子树集合的乘积之和 == G[n]
         */
        int[] G = new int[n + 1];
        G[1] = 1;
        G[0] = 1;
        for (int i = 2;i <= n;i++) {
            for (int j = 1;j <= i;j++) {
                G[i] += G[j - 1] * G[i - j];
            }
        }
        return G[n];
    }

    // 237
    public void deleteNode(ListNode node) {
        int tmp;
        tmp = node.val;
        node.val = node.next.val;
        node.next.val = tmp;
        node.next = node.next.next;
    }

    // 344
    public void reverseString(char[] s) {
        int n = s.length;
        char tmp;
        for (int i = 0;i < n / 2;i++) {
            tmp = s[i];
            s[i] = s[n - 1 - i];
            s[n - 1 - i] = tmp;
        }
    }

    // 557
    public String reverseWords(String s) {
        /**
         * 1、截取单词、调用reverseString()反转单词再拼接
         */
//        int length = s.length(), index = 0;
//        String ans = "";
//        for (int i = 0;i < length;i++) {
//            if (s.charAt(i) == ' ' || i == length - 1) {
//                if (i == length - 1) i++;
//                char[] s1 = s.substring(index, i).toCharArray();
//                reverseString(s1);
//                ans += String.valueOf(s1);
//                if (i != length) ans += ' ';
//                index = i + 1;
//            }
//        }
//        return ans;
        /**
         * 2、直接截取单词、将字符逆序加入答案
         */
        StringBuilder ans = new StringBuilder();
        int n = s.length(), i = 0;
        while (i < n){
            int start = i;
            while (i < n && s.charAt(i) != ' ') i++;
            for (int index = i - 1;index >= start;index--) {
                ans.append(s.charAt(index));
            }
            while (i < n && s.charAt(i) == ' ') {
                i++;
                ans.append(' ');
            }
        }
        return ans.toString();
    }

    // 217
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> numSet = new HashSet<>();
        for (int i : nums) {
            numSet.add(i);
        }
        return nums.length != numSet.size();
    }

    // 88
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        /**
         * 逆向双指针  合并两个非递减数组
         */
        int i = m - 1, j = n - 1, length = m;
        while (i >= 0 && j >= 0) {
            if (nums1[i] <= nums2[j]) {
                nums1[j + i + 1] = nums2[j];
                j--;
            } else {
                nums1[j + i + 1] = nums1[i];
                i--;
            }
        }
        while (j >= 0) nums1[j + i + 1] = nums2[j--];
    }

    // 292
    public boolean canWinNim(int n) {
        /**
         * 巴什博奕 笑死  官方送温暖
         */
        return n % 4 != 0;
    }

    // 231
    public boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    // 235
    public TreeNode lowestCommonAncestor235(TreeNode root, TreeNode p, TreeNode q) {
        /**
         * 二叉搜索树两节点的公共祖先
         */
        while (root != null) {
            if (root.val > p.val && root.val > q.val) {
                root = root.left;
            } else if (root.val < p.val && root.val < q.val) {
                root = root.right;
            } else return root;
        }
        return null;
    }

    // 14
    public String longestCommonPrefix(String[] str) {
        int length = str[0].length(), l = str.length;
        for (int i = 0;i < length;i++) {
            char ch = str[0].charAt(i);
            for (int j = 1;j < l;j++) {
                if (i == str[j].length() || str[j].charAt(i) != ch)
                    return str[0].substring(0, i);
            }
        }
        return str[0];
    }

    // 762
    public int countPrimeSetBits(int left, int right) {
        int ans = 0;
        for (int i = left;i <= right;i++) {
            if (isPrim(Integer.bitCount(i)))
                ans++;
        }
        return ans;
    }
    public boolean isPrim(int x) {
        if (x < 2) return false;
        for (int i = 2;i <= x / 2;i++) {
            if (x % i == 0)
                return false;
        }
        return true;
    }

    // 504
    public String convertToBase7(int num) {
        StringBuilder res = new StringBuilder();
        if (num == 0) return "0";
        boolean flag = num < 0;
        num = Math.abs(num);
        while (num != 0) {
            res.append(num % 7);
            num = num / 7;
        }
        if (flag == true) res.append('-');
        return res.reverse().toString();
    }

    // 2055
    public int[] platesBetweenCandles(String s, int[][] queries) {
        int[] answer = new int[queries.length];
        return answer;
    }

    // 9
    public boolean isPalindrome(int x) {
//        char[] chars = Integer.toString(x).toCharArray();
//        for (int i = 0;i < chars.length / 2;i++) {
//            if (chars[i] != chars[chars.length - 1 - i])
//                return false;
//        }
//        return true;
        if (x < 0 || (x % 10 == 0 && x != 0)) return false;
        int reversalNumber = 0;
        while (x > reversalNumber) {
            reversalNumber = reversalNumber * 10 + x % 10;
            x /= 10;
        }
        return x == reversalNumber || x == reversalNumber / 10;
    }

    // 20
    public boolean isValid(String s) {
        Map<Character, Character> characterMap = new HashMap<Character, Character>() {{
            put('(', ')');
            put('[', ']');
            put('{', '}');
            put('?', '?');
        }};
        LinkedList<Character> stack = new LinkedList<Character>() {{
            add('?');
        }};
        for (char c : s.toCharArray()) {
            if (characterMap.containsKey(c)) stack.addLast(c);
            else if (characterMap.get(stack.removeLast()) != c) return false;
        }
        return stack.size() == 1;
    }

    // 206
    public ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }

    // 208
    class Trie {
        /**
         * 前缀树
         */
        private Trie[] children;
        private boolean isEnd;

        public Trie() {
            children = new Trie[26];
            isEnd = false;
        }

        public void insert(String word) {
            Trie node = this;
            int n = word.length();
            for (int i = 0;i < n;i++) {
                char ch = word.charAt(i);
                int index = ch - 'a';
                if (node.children[index] == null)
                    node.children[index] = new Trie();
                node = node.children[index];
            }
            node.isEnd = true;
        }

        public boolean search(String word) {
            Trie node = this.searchPrefix(word);
            return node != null && node.isEnd == true;
        }

        public boolean startsWith(String prefix) {
            return this.searchPrefix(prefix) != null;
        }

        public Trie searchPrefix(String prefix) {
            Trie node = this;
            int n = prefix.length();
            for (int i = 0;i < n;i++) {
                char ch = prefix.charAt(i);
                int index = ch - 'a';
                if (node.children[index] == null) return null;
                node = node.children[index];
            }
            return node;
        }
    }

    // offer 27
    public TreeNode mirrorTree(TreeNode root) {
        /**
         * 翻转二叉树
         */
        if (root == null) return null;
        TreeNode left = mirrorTree(root.left);
        TreeNode right = mirrorTree(root.right);
        root.left = right;
        root.right = left;
        return root;
    }

    // offer 54
    int res54, k;
    public int kthLargest(TreeNode root, int k) {
        /**
         * 逆中序遍历 到第 k 个结束
         */
        this.k = k;
        dfs(root);
        return res54;
    }
    void dfs(TreeNode root) {
        if (root == null) return;
        dfs(root.right);
        if (k == 0) return;;
        if (--k == 0) res54 = root.val;
        dfs(root.left);
    }

    // offer 57-2
    public int[][] findContinuousSequence(int target) {
        /**
         * 1、暴力枚举
         */
//        List<int[]> ans = new ArrayList<>();
//        int sum = 0;
//        for (int i = 1;i <= target / 2;i++) {
//            for (int j = i;;j++) {
//                sum += j;
//                if (sum > target) {
//                    sum = 0;
//                    break;
//                } else if (sum == target) {
//                    int[] res = new int[j - i + 1];
//                    for (int k = i;k <= j;k++) {
//                        res[k - i] = k;
//                    }
//                    ans.add(res);
//                    sum = 0;
//                    break;
//                }
//            }
//        }
//        return ans.toArray(new int[ans.size()][]);      //初始化数组长度
        /**
         * 2、双指针优化
         */
        List<int[]> ans = new ArrayList<>();
        int sum = 0;
        for (int l = 1, r = 2;l < r;) {
            sum = (l + r) * (r - l + 1) / 2;
            if (sum > target) {
                l++;
            } else if (sum == target) {
                int[] res = new int[r - l + 1];
                for (int k = l;k <= r;k++) {
                    res[k - l] = k;
                }
                ans.add(res);
                l++;
            } else {
                r++;
            }
        }
        return ans.toArray(new int[ans.size()][]);
    }

    // 236 & offer 68-2
    public TreeNode lowestCommonAncestor236(TreeNode root, TreeNode p, TreeNode q) {
        /**
         * 二叉树两节点的公共祖先
         */
        dfs236(root, p, q);
        return ans236;
    }
    boolean dfs236(TreeNode root, TreeNode p, TreeNode q) {
        /**
         * 深度搜索树中是否包含 p 或 q
         */
        if (root == null) return false;
        boolean left = dfs236(root.left, p, q);
        boolean right = dfs236(root.right, p, q);
        if ((left && right) || (root.val == p.val || root.val == q.val) && (left || right))
            ans236 = root;
        return left || right || root.val == p.val || root.val == q.val;
    }

    // 154 & offer 11
    public int minArray(int[] numbers) {
        int low = 0, high = numbers.length - 1;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (numbers[mid] > numbers[high]) {
                low = mid + 1;
            } else if (numbers[mid] < numbers[high]) {
                high = mid;
            } else {
                high--;
            }
        }
        return numbers[low];
    }

    // 102 & offer 32-2
    List<List<Integer>> ans102 = new ArrayList<>();
    public List<List<Integer>> levelOrder(TreeNode root) {
        /**
         * 1、队列  广度优先搜索
         */
//        List<List<Integer>> ans = new ArrayList<>();
//        if (root == null) return ans;
//        Queue<TreeNode> queue = new LinkedList<>();
//        queue.offer(root);
//        while (!queue.isEmpty()) {
//            List<Integer> list = new ArrayList<>();
//            int length = queue.size();
//            for (int i = 0;i < length;i++) {
//                TreeNode treeNode = queue.poll();
//                list.add(treeNode.val);
//                if (treeNode.left != null)
//                    queue.offer(treeNode.left);
//                if (treeNode.right != null)
//                    queue.offer(treeNode.right);
//            }
//            ans.add(list);
//        }
//        return ans;
        /**
         * 2、递归层次遍历
         */
        lever(root, 0);
        return ans102;
    }
    void lever(TreeNode root, int k) {
        if (root != null) {
            if (ans102.size() <= k)
                ans102.add(new ArrayList<>());
            ans102.get(k).add(root.val);
            lever(root.left, k + 1);
            lever(root.right, k + 1);
        }
    }

    // offer 57
    public int[] twoSum57(int[] nums, int target) {
        /**
         * 数组递增排序   两数之和  返回数字
         * 1、老方法 借助哈希表
         */
//        int length = nums.length;
//        Map<Integer, Integer> hashMap = new HashMap<>();
//        for (int i = 0;i < length;i++) {
//            if (hashMap.containsKey(target - nums[i])) {
//                return new int[] {hashMap.get(target - nums[i]), nums[i]};
//            }
//            if (nums[i] < target) {
//                hashMap.put(nums[i], nums[i]);
//            } else {
//              break;
//            }
//        }
//        return new int[0];
        /**
         * 2、数组为排序数组  使用双指针（对撞指针）
         */
        int low = 0, high = nums.length - 1, sum;
        while (low < high) {
            sum = nums[low] + nums[high];
            if (sum > target) {
                high--;
            } else if (sum < target) {
                low++;
            } else {
                return new int[] {nums[low], nums[high]};
            }
        }
        return new int[0];
    }

    // 796
    public boolean rotateString(String s, String goal) {
        /**
         * 旋转字符串
         * 1、模拟
         */
//        int length = s.length();
//        if (length != goal.length()) {
//            return false;
//        }
//        for (int i = 0;i < length;i++) {
//            int index = i, j = 0;
//            while (s.charAt(index) == goal.charAt(j)) {
//                System.out.println(s.charAt(index));
//                System.out.println(goal.charAt(index));
//                System.out.println(s.charAt(index) == goal.charAt(j));
//                index = (index + 1) % length;
//                if (length - 1 == j) {
//                    return true;
//                }
//                j++;
//            }
//        }
//        return false;
        /**
         * 2、搜索子字符串   只要 goal 和 s 长度一致 且 goal 是 s + s 的子字符串即可
         */
        return goal.length() == s.length() && (s + s).contains(goal);
    }

    // offer 21
    public int[] exchange(int[] nums) {
        /**
         * 1、双指针同步
         */
//        int low = 0, high = nums.length - 1;
//        while (low < high) {
//            if (nums[low] % 2 == 0 && nums[high] % 2 != 0) {
//                int tem = nums[low];
//                nums[low] = nums[high];
//                nums[high] = tem;
//                low++;
//                high--;
//            } else if (nums[low] % 2 == 0 && nums[high] % 2 == 0) {
//                high--;
//            } else if (nums[low] % 2 != 0 && nums[high] % 2 != 0) {
//                low++;
//            } else {
//                low++;
//                high--;
//            }
//        }
//        return nums;
        /**
         * 2、双指针异步
         */
        int low = 0, high = nums.length - 1;
        while (low < high) {
            while (low < high && nums[low] % 2 != 0) {
                low++;
            }
            while (low < high && nums[high] % 2 == 0) {
                high--;
            }
            int tem = nums[low];
            nums[low] = nums[high];
            nums[high] = tem;
        }
        return nums;
    }

    // offer 40
    public int[] getLeastNumbers(int[] arr, int k) {
        Arrays.sort(arr);
        int[] ans = new int[k];
        for (int i = 0;i < k;i++) {
            ans[i] = arr[i];
        }
        return ans;
    }

    // offer 53-1
    public int search(int[] nums, int target) {
        /**
         * 排序数组必二分查找
         */
        int count = 0, length = nums.length;
        for (int i = 0;i < length;i++) {
            if (nums[i] == target) {
                count++;
            }
        }
        return count;
    }

    // offer 61
    public boolean isStraight(int[] nums) {
        /**
         * 判断顺子  0可以看做任意数
         */
        Set<Integer> set = new HashSet<>();
        int max = 0, min = 14;
        for (int num : nums) {
            if (num == 0) continue;
            max = Math.max(max, num);
            min = Math.min(min, num);
            if (set.contains(num)) return false;
            set.add(num);
        }
        return max - min < 5;
    }

    // offer 53-2
    public int missingNumber(int[] nums) {
        for (int i = 0;i < nums.length;i++) {
            if (nums[i] != i) return i;
        }
        return nums.length;
    }

    // offer 18
    public ListNode deleteNode(ListNode head, int val) {
        if (head.val == val) return head.next;
        ListNode pre = head, cur = head.next;
        while (cur != null && cur.val != val) {
            pre = cur;
            cur = cur.next;
        }
        if (cur != null) pre.next = cur.next;
        return head;
    }

    public List<List<Integer>> levelOrder(Node root) {
        /**
         * n叉树的层次遍历
         */
        if (root == null) return new ArrayList<>();
        List<List<Integer>> ans = new ArrayList<>();
        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            List<Integer> res = new ArrayList<>();
            int length = queue.size();
            for (int i = 0;i < length;i++){
                Node node = queue.poll();
                res.add(node.val);
                for (Node node1 : node.children) {
                    queue.offer(node1);
                }
            }
            ans.add(res);
        }
        return ans;
    }

    // 541
    public String reverseStr(String s, int k) {
        int length = s.length();
        char[] chars = s.toCharArray();
        for (int i = 0;i < length;i += 2 * k) {
            reverse(chars, i, Math.min(i + k - 1, length - 1));
        }
        return new String(chars);
    }
    void reverse(char[] chars, int left, int right) {
        while (left < right) {
            char tem = chars[left];
            chars[left] = chars[right];
            chars[right] = tem;
            left++;
            right--;
        }
    }

    // 780
    public boolean reachingPoints(int sx, int sy, int tx, int ty) {
        while (tx > sx && ty > sy && tx != ty) {
            if (tx > ty) {
                tx %= ty;
            } else {
                ty %= tx;
            }
        }
        if (tx == ty) {
            return true;
        } else if (tx == sx) {
            return ty > sy && (ty - sy) % tx == 0;
        } else if (ty == sy) {
            return tx > sx && (tx - sx) % ty == 0;
        } else {
            return false;
        }
    }

    // 804
    public int uniqueMorseRepresentations(String[] words) {
        String[] MORSE = {".-", "-...", "-.-.", "-..", ".", "..-.",
                "--.", "....", "..", ".---", "-.-", ".-..", "--",
                "-.", "---", ".--.", "--.-", ".-.", "...", "-",
                "..-", "...-", ".--", "-..-", "-.--", "--.."};
        Set<String> ans = new HashSet<>();
        for (String string : words) {
            StringBuilder stringBuilder = new StringBuilder();
            int length = string.length();
            for (int i = 0;i < length;i++) {
                char ch = string.charAt(i);
                stringBuilder.append(MORSE[ch - 'a']);
            }
            ans.add(new String(stringBuilder));
        }
        return ans.size();
    }

    // 203
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

    // 19
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(-1,head);
        ListNode low = dummy, high = dummy;
        while (n >= 0) {
            high = high.next;
            n--;
        }
        while (high != null) {
            high = high.next;
            low = low.next;
        }
        low.next = low.next.next;
        return dummy.next;
    }

    // 242
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) return false;
        int[] ints = new int[26];
        for (char ch : s.toCharArray()) {
            ints[ch - 'a']++;
        }
        for (char ch : t.toCharArray()) {
            ints[ch - 'a']--;
        }
        for (int i : ints) {
            if (i != 0) {
                return false;
            }
        }
        return true;
    }

    // 349
    public int[] intersection(int[] nums1, int[] nums2) {
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i : nums1) {
            hashMap.put(i, i);
        }
        HashSet<Integer> ans = new HashSet<>();
        for (int i : nums2) {
            if (hashMap.containsKey(i)) {
                ans.add(i);
            }
        }
        int index = 0;
        int[] res = new int[ans.size()];
        for (int i : ans) {
            res[index++] = i;
        }
        return res;
    }

    // 202
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

    // 806
    public int[] numberOfLines(int[] widths, String s) {
        int sum = 0, count = 1;
        for (int i = 0;i < s.length();i++) {
            sum += widths[s.charAt(i) - 'a'];
            if (sum > 100) {
                sum = widths[s.charAt(i) - 'a'];
                count++;
            }
        }
        return new int[] {count, sum};
    }

    // 383
    public boolean canConstruct(String ransomNote, String magazine) {
        /**
         * 赎金信
         */
//        ArrayList<Character> record = new ArrayList<>();
//        for (char ch : magazine.toCharArray()) {
//            record.add(ch);
//        }
//        for (char ch : ransomNote.toCharArray()) {
//            if (record.contains(ch)) {
//                record.remove(record.indexOf(ch));
//            } else {
//                return false;
//            }
//        }
//        return true;
        /**
         * 2、哈希解法、数组做哈希表
         */
        int[] record = new int[26];
        for (char ch : magazine.toCharArray()) {
            record[ch - 'a']++;
        }
        for (char ch : ransomNote.toCharArray()) {
            if (--record[ch - 'a'] < 0) {
                return false;
            }
        }
        return true;
    }

    // 1047
    public String removeDuplicates(String s) {
        ArrayDeque<Character> deque = new ArrayDeque<>();
        //Deque的两个实现类   ArrayDeque LinkedList
        char ch;
        int length = s.length();
        for (int i = 0;i < length;i++) {
            ch = s.charAt(i);
            if (deque.isEmpty() || ch != deque.peek()) {
                deque.push(ch);
            } else {
                deque.poll();
            }
        }
        String ans = "";
        while (!deque.isEmpty()) {
            ans = deque.poll() + ans;
        }
        return ans;
    }

    // 1672
    public int maximumWealth(int[][] accounts) {
//        int sum = 0, ans = 0;
//        for (int[] num : accounts) {
//            for (int i : num) {
//                sum += i;
//            }
//            ans = Math.max(sum, ans);
//            sum = 0;
//        }
//        return ans;
        int ans = 0;
        for (int[] num : accounts) {
            ans = Math.max(ans, Arrays.stream(num).sum());       // 官方瞎搞
        }
        return ans;
    }

    // offer 62
    public int lastRemaining(int n, int m) {
        /**
         * 1、递归
         */
//        if (n == 1) return 0;
//        int x = lastRemaining(n - 1, m);
//        return (m + x) % n;
        /**
         * 2、迭代
         */
        int f = 0;
        for (int i = 2;i < n + 1;i++) {
            f = (m + f) % i;
        }
        return f;
    }

    // offer 50
    public char firstUniqChar(String s) {
        /**
         * 第一个只出现一次的字符
         */
        HashMap<Character, Integer> hashMap = new HashMap<>();
        for (char ch : s.toCharArray()) {
            hashMap.put(ch, hashMap.getOrDefault(ch, 0) + 1);
        }
        for (char ch : s.toCharArray()) {
            if (hashMap.get(ch) == 1) {
                return ch;
            }
        }
        return  ' ';
    }

    // 110 & offer 55-2
    public boolean isBalanced(TreeNode root) {
        /**
         * 自顶向下  height重复调用导致时间复杂度升高
         * 自底向上  首先判断是否平衡 平衡再返回深度
         */
        if (root == null) return true;
        return Math.abs(height(root.left) - height(root.right)) <= 1
                && isBalanced(root.left) && isBalanced(root.right);
    }
    public int height(TreeNode root) {
        if (root == null) return 0;
        return Math.max(height(root.left), height(root.right)) + 1;
    }

    // offer 58-1
    public String reverseWords58(String s) {
        // 除去开头和末尾的空白字符
        s = s.trim();
        // 正则匹配连续的空白字符作为分隔符分割
        List<String> wordList = Arrays.asList(s.split("\\s+"));
        Collections.reverse(wordList);
        return String.join(" ", wordList);
    }

    // 257
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> ans = new ArrayList<>();
        if (root == null) return ans;
        constructPaths(root, "", ans);
        return ans;

    }
    public void constructPaths(TreeNode root, String path, List<String> ans) {
        if (root != null) {
            StringBuffer sb = new StringBuffer(path);
            sb.append(root.val);
            if (root.left == null && root.right == null) {
                ans.add(sb.toString());
            } else {
                sb.append("->");
                constructPaths(root.left, sb.toString(), ans);
                constructPaths(root.right, sb.toString(), ans);
            }
        }
    }

    public int sumOfLeftLeaves(TreeNode root) {
        /**
         * 求左叶子之和
         */
        if (root == null) return 0;
        int left = sumOfLeftLeaves(root.left);
        int right = sumOfLeftLeaves(root.right);
        if (root.left != null && root.left.right == null && root.left.left == null) {   // 此处左孩子右孩子均为空不能说明该节点一点为左叶子
            return left + right + root.left.val;
        }
        return left + right;
    }

    // 819
    public String mostCommonWord(String paragraph, String[] banned) {
        HashSet<String> bannedSet = new HashSet<>();
        for (String string : banned) {
            bannedSet.add(string);
        }
        int length = paragraph.length(), maxFre = 0;
        StringBuilder builder = new StringBuilder();
        HashMap<String, Integer> fre = new HashMap<>();
        for (int i = 0;i <= length;i++) {
            if (i < length && Character.isLetter(paragraph.charAt(i))) {
                builder.append(Character.toLowerCase(paragraph.charAt(i)));
            } else if (builder.length() > 0) {
                if (!bannedSet.contains(builder)) {
                    String word = builder.toString();
                    int indexFre = fre.getOrDefault(word, 0) + 1;
                    fre.put(word, indexFre);
                    maxFre = Math.max(maxFre, indexFre);
                }
                builder.setLength(0);
            }
        }
        String ans = "";
        for (Map.Entry<String, Integer> entry : fre.entrySet()) {
            /**
             * 知识点：entrySet    将映射搞成集合    方便分别操作键和值
             */
            String word = entry.getKey();
            int frequency = entry.getValue();
            if (frequency == maxFre) {
                ans = word;
                break;
            }
        }
        return ans;
    }

    // 821
    public int[] shortestToChar(String s, char c) {
        int length = s.length();
        int[] ans = new int[length];
        for (int i = 0, index = -length;i < length;i++) {
            if (s.charAt(i) == c) {
                index = i;
            }
            ans[i] = i - index;
        }
        for (int i = length - 1, index = 2 * length;i >= 0;i--) {
            if (s.charAt(i) == c) {
                index = i;
            }
            ans[i] = Math.min(ans[i], index - i);
        }
        return ans;
    }

    // offer 56-2
    public int singleNumber2(int[] nums) {
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i : nums) {
            hashMap.put(i, hashMap.getOrDefault(i, 0) + 1);
        }
        for (Map.Entry<Integer, Integer> entry : hashMap.entrySet()) {
            if (entry.getValue() == 1) {
                return entry.getKey();
            }
        }
        return 0;
    }

    // offer 56-1
    public int[] singleNumbers(int[] nums) {
        int res = 0;
        for (int i : nums) {
            res ^= i;
        }
        int index = 1;
        while ((res & index) == 0) {
            index <<= 1;
        }
        int a = 0, b = 0;
        for (int n : nums) {
            if ((n & index) != 0) {
                a ^= n;
            } else {
                b ^= n;
            }
        }
        return new int[] {a, b};
    }

    // 944
    public int minDeletionSize(String[] strs) {
        int ans = 0, row = strs.length, col = strs[0].length();
        for (int i = 0;i < col;i++) {
            for (int j = 1;j < row;j++) {
                if (strs[j].charAt(i) < strs[j - 1].charAt(i)) {
                    ans++;
                    break;
                }
            }
        }
        return ans;
    }

    // 977
    public int[] sortedSquares(int[] nums) {
        int left = 0, right = nums.length - 1;
        int[] ans = new int[right + 1];
        for (int i = right;i >= 0;i--) {
            int res1 = nums[left] * nums[left], res2 = nums[right] * nums[right];
            if (res1 > res2) {
                ans[i] = res1;
                left++;
            }
            else {
                ans[i] = res2;
                right--;
            }
        }
        return ans;
    }

    // 209
    public int minSubArrayLen(int target, int[] nums) {
        /**
         * 滑动窗口 最小连续子数组
         */
        int ans = Integer.MAX_VALUE;
        int sum = 0;
        int left = 0;
        for (int right = 0;right < nums.length;right++) {
            sum += nums[right];
            while (sum >= target) {
                ans = Math.min(ans, right - left + 1);
                sum -= nums[left++];
            }
        }
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }

    // 450
    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null) return null;
        if (root.val > key) {
            root.left = deleteNode(root.left, key);
            return root;
        }
        if (root.val < key) {
            root.right = deleteNode(root.right, key);
            return root;
        }
        if (root.val == key) {
            if (root.left == null && root.right == null) return null;
            if (root.left == null) return root.right;
            if (root.right == null) return root.left;
            TreeNode index = root.right;
            while (index.left != null) {
                index = index.left;
            }
            root.right = deleteNode(root.right, index.val);
            index.right = root.right;
            index.left = root.left;
            return index;
        }
        return root;
    }

    // 829
    public int consecutiveNumbersSum(int n) {
//        int count = 0;
//        for (int i = 1;i <= n;i++) {
//            int sum = 0;
//            for (int j = i;j <= n;j++) {
//                sum += i;
//                if (sum == n) {
//                    count++;
//                    break;
//                }
//            }
//        }
//        return count;
        int ans = 0;
        for (int i = 1;i * i < 2 * n;i++) {
            if (2 * n % i == 0 && (2 * n / i - i + 1) % 2== 0)
                ans++;
        }
        return ans;
    }

    // 929
    public int numUniqueEmails(String[] emails) {
        HashSet<String> ans = new HashSet<>();
        for (String string : emails) {
            int i = string.indexOf('@');
            String left = string.substring(0, i).split("\\+")[0].replace(".", "");
            ans.add(left + string.substring(i));
        }
        return ans.size();
    }

    // 875
    public int minEatingSpeed(int[] piles, int h) {
        int low = 1;
        int high = 0;
        for (int pile : piles) {
            high = Math.max(pile, high);
        }
        int k = high;
        while (low < high) {
            int speed = (high - low) / 2 + low;
            int time = getTime(piles, speed);
            if (time <= h) {
                k = speed;
                high = speed;
            } else {
                low = speed + 1;
            }
        }
        return k;
    }
    int getTime(int[] piles, int speed) {
        int time = 0;
        for (int pile : piles) {
            int curTime = (pile + speed - 1) / speed;        //  解决结果等于0的问题  如3/6=0  根据题意应等于1
            time += curTime;
        }
        return time;
    }

    // 1037
    public boolean isBoomerang(int[][] points) {
        /**
         * 向量叉乘不等于0 == 三点相同且不在一条直线
         */
        int[] v1 = {points[1][0] - points[0][0], points[1][1] - points[0][1]};
        int[] v2 = {points[2][0] - points[0][0], points[2][1] - points[0][1]};
        return v1[0] * v2[1] - v1[1] * v2[0] != 0;
    }

    // 28
    public int strStr(String haystack, String needle) {
        if (needle.length() == 0) {
            return 0;
        }
        int[] next = new int[needle.length()];
        getNext(next, needle);
        int j = -1;
        for (int i = 0;i < haystack.length();i++) {
            while (j >= 0 && haystack.charAt(i) != needle.charAt(j + 1)) {
                j = next[j];
            }
            if (haystack.charAt(i) == needle.charAt(j + 1)) {
                j++;
            }
            if (j == needle.length() - 1) {
                return (i - needle.length() + 1);
            }
        }
        return -1;
    }
    public void getNext(int[] next, String s) {
        int j = -1;
        next[0] = j;
        for (int i = 1;i < s.length();i++) {
            while (j >= 0 && s.charAt(i) != s.charAt(j + 1)) {
                j = next[j];
            }
            if (s.charAt(i) == s.charAt(j + 1)) {
                j++;
            }
            next[i] = j;
        }
    }

    // 111
    public int minDepth(TreeNode root) {
//        if (root == null) {
//            return 0;
//        }
//        int leftDepth = minDepth(root.left);
//        int rightDepth = minDepth(root.right);
//        if (root.left == null) {
//            return 1 + rightDepth;
//        }
//        if (root.right == null) {
//            return 1 + leftDepth;
//        }
//        return Math.min(leftDepth, rightDepth) + 1;
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        }
        int minDepth = Integer.MAX_VALUE;
        if (root.left != null) {
            minDepth = Math.min(minDepth(root.left), minDepth);
        }
        if (root.right != null) {
            minDepth = Math.min(minDepth, minDepth(root.right));
        }
        return minDepth + 1;
    }

    // 513
    public int findBottomLeftValue(TreeNode root) {
        Deque<TreeNode> deque = new LinkedList<>();
        deque.offer(root);
        int res = 0;
        while (!deque.isEmpty()) {
            int size = deque.size();
            for (int i = 0;i < size;i++) {
                TreeNode poll = deque.poll();
                if (i == 0) {
                    res = poll.val;
                }
                if (poll.left != null) {
                    deque.offer(poll.left);
                }
                if (poll.right != null) {
                    deque.offer(poll.right);
                }
            }
        }
        return res;
    }

    // 112
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null) {
            return root.val == targetSum;
        }
        return hasPathSum(root.left, targetSum - root.val) ||
                hasPathSum(root.right, targetSum - root.val);
    }

    // 654
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return constructMaximumBinaryTree1(nums, 0, nums.length);
    }
    public TreeNode constructMaximumBinaryTree1(int[] nums, int leftIndex, int rightIndex) {
        if (rightIndex - leftIndex < 1) {
            return null;
        }
        if (rightIndex - leftIndex == 1) {
            return new TreeNode(nums[leftIndex]);
        }
        int maxIndex = leftIndex;
        int maxValue = nums[maxIndex];
        for (int i = leftIndex + 1;i < rightIndex;i++) {
            if (nums[i] > maxValue) {
                maxValue = nums[i];
                maxIndex = i;
            }
        }
        TreeNode root = new TreeNode(nums[maxIndex]);
        root.left = constructMaximumBinaryTree1(nums, leftIndex, maxIndex);
        root.right = constructMaximumBinaryTree1(nums, maxIndex + 1, rightIndex);
        return root;
    }

    // 501
    ArrayList<Integer> resList;
    int count;
    int maxCount;
    TreeNode pre;
    public int[] findMode(TreeNode root) {
        /*
         * @description:找出二叉树中的众数  二叉树中双指针
         * @author: pwz
         * @date: 2022/7/22 10:38
         * @param: [root]
         * @return: int[]
         **/
        resList = new ArrayList<>();
        pre = null;
        maxCount = 0;
        count = 0;
        findMode1(root);
        int[] res = new int[resList.size()];
        for (int i = 0;i < resList.size();i++) {
            res[i] = resList.get(i);
        }
        return res;
    }
    public void findMode1(TreeNode root) {
        if (root == null) {
            return;
        }
        findMode1(root.left);
        int rootValue = root.val;
        if (pre == null || rootValue != pre.val) {
            count = 1;
        } else {
            count++;
        }

        if (count > maxCount) {
            resList.clear();
            resList.add(rootValue);
            maxCount = count;
        } else if (count == maxCount) {
            resList.add(rootValue);
        }
        pre = root;
        findMode1(root.right);
    }


    public static void main(String[] args) {
        LCodeSolution LCodeSolution = new LCodeSolution();
        int[] test = {1, 2, 3};
        String testString = "234";
        LCodeSolution.reverseWords58("the sky is blue");
//        System.out.println(LCodeSolution.minEatingSpeed(new int[]{3, 6, 7, 11}, 8));
//        LCodeSolution.canConstruct("aa", "aab");        383
//        LCodeSolution.numberOfLines(new int[] {3,4,10,4,8,7,3,3,4,9,8,2,9,6,2,
//                8,4,9,9,10,2,4,9,10,8,2},"mqblbtpvicqhbrejb");             806
//        System.out.println(LCodeSolution.climbStairs(3));         70
//        System.out.println(LCodeSolution.hammingDistance(3, 1));     461
//        System.out.println(Arrays.toString(LCodeSolution.countBits(10)));     338
//        System.out.println(LCodeSolution.findDisappearedNumbers(new int[]{2, 2, 3, 4, 2}).toString());     448
//        System.out.println(LCodeSolution.majorityElement(new int[]{1, 2, 3, 3, 3}));                      169
//        System.out.println(LCodeSolution.singleNumber(new int[]{2, 3, 3, 1, 1}));                 136
//        LCodeSolution.moveZeroes(test);                                                       283
//        System.out.println(Arrays.toString(test));
//        System.out.println(LCodeSolution.maxProfit(new int[]{5, 1, 5}));             121
//        System.out.println(LCodeSolution.subsets(new int[]{1, 2}));                78
//        System.out.println(LCodeSolution.permute(new int[]{1, 2, 3}));           46
//        System.out.println(LCodeSolution.generateParenthesis(2));                   22
//        System.out.println(LCodeSolution.combinationSum(test, 5));             39
//        System.out.println(LCodeSolution.letterCombinations(testString));           17
//        System.out.println(LCodeSolution.findTargetSumWays(new int[]{1, 2, 2, 1}, 6));         494
//        System.out.println(LCodeSolution.exist(new char[][]{{'a', 'b'}, {'a', 'c'}}, "bc"));      79
//        System.out.println(Arrays.toString(LCodeSolution.dailyTemperatures(new int[]{23, 32, 43, 54})));      739
//        LCodeSolution.groupAnagrams(new String[] {"abc", "bac", "das"});       148
//        System.out.println(LCodeSolution.threeSum(new int[]{-1, 0, 1, 2, -1, 4}));               15
//        System.out.println(LCodeSolution.reverseWords(new String("Let's take LeetCode contest")));   557
//        System.out.println(LCodeSolution.rotateString("abcde", "abcdg"));       796
//        System.out.println(LCodeSolution.exchange(new int[]{1, 2, 3, 4}));     offer 21


    }
}
