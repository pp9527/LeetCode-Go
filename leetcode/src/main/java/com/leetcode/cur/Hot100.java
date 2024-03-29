package com.leetcode.cur;

import java.util.*;

/**
 * @author: pwz
 * @create: 2022/10/10 9:46
 * @Description: hot100题目
 * @FileName: Hot100
 */
public class Hot100 {

    /**
     * @Description: 3. 无重复字符的最长子串
     * @author pwz
     * @date 2022/10/10 10:36
     */
    public int lengthOfLongestSubstring(String s) { // 滑动窗口59/42
        HashSet<Object> set = new HashSet<>();
        int length = s.length(), rk = -1, res = 0;
        for (int i = 0; i < length; i++) {
            if (i != 0) {
                set.remove(s.charAt(i - 1));
            }
            while (rk + 1 < length && !set.contains(s.charAt(rk + 1))) {
                set.add(s.charAt(rk + 1));
                rk++;
            }
            res = Math.max(res, rk - i + 1);
        }
        return res;
    }

    public int lengthOfLongestSubstring_1(String s) { // 滑动窗口100/82
        // 记录字符上一次出现的位置
        int[] last = new int[128];
        int n = s.length();
        int res = 0;
        int start = 0; // 窗口开始位置
        for (int i = 0; i < n; i++) {
            int index = s.charAt(i);
            start = Math.max(start, last[index] + 1);
            res = Math.max(res, i - start + 2);
            last[index] = i + 1;
        }
        return res;
    }

    /**
     * @Description: 5. 最长回文子串
     * @author pwz
     * @date 2022/10/11 10:25
     */
    public String longestPalindrome(String s) { // 动态规划28/18
        boolean[][] dp = new boolean[s.length()][s.length()];
        int left = 0, maxLen = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            for (int j = i; j < s.length(); j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    if (j - i <= 1) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }
                if (dp[i][j] && j - i + 1 > maxLen) {
                    left = i;
                    maxLen = j - i + 1;
                }
            }
        }
        return s.substring(left, left + maxLen);
    }

    public String longestPalindrome_1(String s) { // 双指针、中心扩散法71/60
        String s1 = "", s2 = "", res = "";
        for (int i = 0; i < s.length(); i++) {
            s1 = extend(s, i, i);
            res = Math.max(s1.length(), res.length()) == res.length() ? res : s1;
            s2 = extend(s, i, i + 1);
            res = Math.max(s2.length(), res.length()) == res.length() ? res : s2;
        }
        return res;
    }

    private String extend(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return s.substring(left + 1, right);
    }

    public String longestPalindrome_2(String s) { // 双指针、中心扩散法 优化 100/98
        int[] range = new int[2];
        char[] str = s.toCharArray();
        for (int i = 0; i < str.length; i++) {
            // 跳过重复字符的遍历，减少时间
            i = findLongest(str, i, range);
        }
        return s.substring(range[0], range[1] + 1);
    }

    private int findLongest(char[] str, int low, int[] range) {
        int high = low;
        while (high < str.length - 1 && str[high + 1] == str[low]) {
            high++;
        }
        int ans = high;
        while (low > 0 && high < str.length - 1 && str[low - 1] == str[high + 1]) {
            low--;
            high++;
        }
        if (high - low > range[1] - range[0]) {
            range[0] = low;
            range[1] = high;
        }
        return ans;
    }

    /**
     * @Description: 11. 盛最多水的容器
     * @author pwz
     * @date 2022/10/18 11:05
     */
    public int maxArea(int[] height) { // 双指针
        int left = 0, right = height.length - 1;
        int res = 0;
        while (left < right) {
            res = height[left] <= height[right] ?
                    Math.max(res, (right - left) * height[left++]) :
                    Math.max(res, (right - left) * height[right--]);
        }
        return res;
    }

    /**
     * @Description: 31. 下一个排列
     * @author pwz
     * @date 2022/10/19 10:20
     */
    public void nextPermutation(int[] nums) {
        int start = nums.length - 2;
        while (start >= 0 && nums[start] >= nums[start + 1]) {
            start--;
        }
        if (start >= 0) {
            int end = nums.length - 1;
            while (end >= 0 && nums[end] <= nums[start]) {
                end--;
            }
            swap(nums, start, end);
        }
        reverse(nums, start + 1);
    }

    private void swap(int[] nums, int a, int b) {
        int temp = nums[a];
        nums[a] = nums[b];
        nums[b] = temp;
    }

    private void reverse(int[] nums, int start) {
        int j = nums.length - 1;
        while (start < j) {
            swap(nums, start, j);
            start++;
            j--;
        }
    }

    /**
     * @Description: 33. 搜索旋转排序数组
     * @author pwz
     * @date 2022/10/20 10:19
     */
    public int search(int[] nums, int target) { // 部分有序  二分查找
        int i = 0, j = nums.length - 1;
        while (i <= j) {
            int mid = (i + j) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[mid] >= nums[i]) {
                if (target < nums[mid] && target >= nums[i]) {
                    j = mid - 1;
                } else {
                    i = mid + 1;
                }
            } else {
                if (target > nums[mid] && target <= nums[j]) {
                    i = mid + 1;
                } else {
                    j = mid - 1;
                }
            }
        }
        return -1;
    }

    /**
     * @Description: 64. 最小路径和
     * @author pwz
     * @date 2022/10/21 10:05
     */
    public int minPathSum(int[][] grid) {
        int row = grid.length;
        int col = grid[0].length;
        for (int i = 1; i < row; i++) {
            grid[i][0] = grid[i - 1][0] + grid[i][0];
        }
        for (int j = 1; j < col; j++) {
            grid[0][j] = grid[0][j - 1] + grid[0][j];
        }
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                grid[i][j] = Math.min(grid[i - 1][j], grid[i][j - 1]) + grid[i][j];
            }
        }
        return grid[row - 1][col - 1];
    }

    /**
     * @Description: 75. 颜色分类
     * @author pwz
     * @date 2022/10/25 20:54
     */
    public void sortColors(int[] nums) {
        int p0 = 0, p1 = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 1) {
                int temp = nums[i];
                nums[i] = nums[p1];
                nums[p1] = temp;
                p1++;
            } else if (nums[i] == 0) {
                int temp = nums[i];
                nums[i] = nums[p0];
                nums[p0] = temp;
                if (p1 > p0) {
                    int tem = nums[i];
                    nums[i] = nums[p1];
                    nums[p1] = tem;
                }
                p0++;
                p1++;
            }
        }
    }

    /**
     * @Description: 128. 最长连续序列
     * 把数放入哈希表set中，去重，遍历set中的每一个数，若当前遍历的数减一已经存在set中，
     * 则此数开头的序列必不可能最长，直接跳过进行下一次遍历，若此数减一不存在set中，则进行加一，
     * 再次判断是否在set中，直至跳出循环，更新长度
     * @author pwz
     * @date 2022/10/26 11:56
     */
    public int longestConsecutive(int[] nums) {
        Set<Integer> num = new HashSet<>();
        int curLen = 0, longest = 0;
        for (int i : nums) {
            num.add(i);
        }
        for (int n : num) {
            if (!num.contains(n - 1)) {
                curLen = 1;
                int curLenNum = n;
                while (num.contains(curLenNum + 1)) {
                    curLenNum++;
                    curLen++;
                }
            }
            longest = Math.max(longest, curLen);
        }
        return longest;
    }

    /**
     * @Description: 152. 乘积最大子数组
     * @author pwz
     * @date 2022/10/29 10:26
     */
    public int maxProduct(int[] nums) {
        int indexMax = 1, indexMin = 1;
        int res = Integer.MIN_VALUE;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < 0) {
                indexMax ^= indexMin;
                indexMin ^= indexMax;
                indexMax ^= indexMin;
            }
            indexMax = Math.max(nums[i], nums[i] * indexMax);
            indexMin = Math.min(nums[i], nums[i] * indexMin);
            res = Math.max(res, indexMax);
        }
        return res;
    }

    /**
     * @Description: 200. 岛屿数量
     * @author pwz
     * @date 2022/10/31 10:16
     */
    public int numIslands(char[][] grid) {
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    dfs(grid, i, j);
                    res++;
                }
            }
        }
        return res;
    }

    private void dfs(char[][] grid, int i, int j) {
        int r = grid.length, c = grid[0].length;
        if (i < 0 || j < 0 || i >= r || j >= c || grid[i][j] == '0') return;
        grid[i][j] = '0';
        dfs(grid, i - 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i + 1, j);
        dfs(grid, i, j - 1);
    }

    /**
     * @Description: 207. 课程表
     * @author pwz
     * @date 2022/11/1 10:40
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<List<Integer>> edges = new ArrayList<List<Integer>>();
        boolean valid = true;
        for (int i = 0; i < numCourses; ++i) {
            edges.add(new ArrayList<Integer>());
        }
        int[] visited = new int[numCourses];
        for (int[] info : prerequisites) {
            edges.get(info[1]).add(info[0]);
        }
        for (int i = 0; i < numCourses && valid; ++i) {
            if (visited[i] == 0) {
                valid = dfs(edges, visited, i);
            }
        }
        return valid;
    }

    private boolean dfs(List<List<Integer>> edges, int[] visited, int index) {
        visited[index] = 1;
        for (int v : edges.get(index)) {
            if (visited[v] == 0) {
                if (!dfs(edges, visited, v)) {
                    return false;
                }
            } else if (visited[v] == 1) {
                return false;
            }
        }
        visited[index] = 2;
        return true;
    }

    /**
     * @Description: 215. 数组中的第K个最大元素
     * @author pwz
     * @date 2022/11/2 10:55
     */
    public int findKthLargest(int[] nums, int k) {
        int left = 0, len = nums.length, right = len - 1;
        while (true) {
            int randomIndex = partition(nums, left, right);
            if (randomIndex == len - k) {
                return nums[randomIndex];
            } else if (randomIndex > len - k) {
                right = randomIndex - 1;
            } else {
                left = randomIndex + 1;
            }
        }
    }
    private int partition(int[] nums, int left, int right) {
        Random random = new Random();
        int randomIndex = left + random.nextInt(right - left + 1);
        int le = left + 1, ge = right;
        swap(nums, left, randomIndex);
        int pivot = nums[left];
        while (true) {
            while (le <= ge && nums[le] < pivot) {
                le++;
            }
            while (le <= ge && nums[ge] > pivot) {
                ge--;
            }
            if (le >= ge) {
                break;
            }
            swap (nums, le, ge);
            le++;
            ge--;
        }
        swap(nums, left, ge);
        return ge;
    }

    /**
     * @Description: 221. 最大正方形
     * @author pwz
     * @date 2022/11/3 10:46
     */
    public int maximalSquare(char[][] matrix) {
        int[][] dp = new int[matrix.length][matrix[0].length];
        int max = 0;
        for (int i = 0; i < matrix.length; i++) {
            dp[i][0] = matrix[i][0] == '1' ? 1 : 0;
        }
        for (int j = 0; j < matrix[0].length; j++) {
            dp[0][j] = matrix[0][j] == '1' ? 1 : 0;
        }
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                    }
                    max = Math.max(max, dp[i][j]);
                }
            }
        }
        return max * max;
    }

    /**
     * @Description: 4. 寻找两个正序数组的中位数
     * @author pwz
     * @date 2022/11/4 10:30
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int len1 = nums1.length, len2 = nums2.length;
        int target = (len1 + len2) / 2;
        int p1 = 0, p2 = 0;
        double left = -1, right = -1;
        for (int i = 0; i <= target; i++) {
            left = right;
            if (p1 < len1 && (p2 >= len2 || nums1[p1] < nums2[p2])) {
                right = nums1[p1++];
            } else {
                right = nums2[p2++];
            }
        }
        if ((len1 + len2) % 2 == 0) {
            return (left + right) / 2;
        }
        return right;
    }




}