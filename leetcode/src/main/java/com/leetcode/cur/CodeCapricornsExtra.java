package com.leetcode.cur;

/**
 * @author: pwz
 * @create: 2022/10/12 10:27
 * @Description:
 * @FileName: CodeCapricornsExtra
 */
public class CodeCapricornsExtra {

    /**
     * @Description: 35. 搜索插入位置
     * @author pwz
     * @date 2022/10/12 10:23
     * @param nums
     * @param target
     * @return int
     */
    public int searchInsert(int[] nums, int target) { // 二分查找
        int left = 0, right = nums.length - 1, mid;
        while (left <= right) {
            mid = ((right - left) >> 2) + left;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return right + 1;
    }

    /**
     * @Description: 34. 在排序数组中查找元素的第一个和最后一个位置
     * @author pwz
     * @date 2022/10/12 10:45
     * @param nums
     * @param target
     * @return int[]
     */
    public int[] searchRange(int[] nums, int target) { // 对撞指针
        int left = 0, right = nums.length - 1;
        while (left <= right && nums[left] != target) left++;
        while (right >= 0 && nums[right] != target) right--;
        if (left > right) return new int[] {-1, -1};
        else return new int[] {left, right};
    }

    /**
     * @Description: 69. x 的平方根
     * @author pwz
     * @date 2022/10/12 11:07
     * @param x
     * @return int
     */
    public int mySqrt(int x) { // 二分查找
        int low = 1, high = x, ans = -1, mid;
        while (low <= high) {
            mid = ((high - low) >> 1) + low;
            if (mid <= x / mid) {
                low = mid + 1;
                ans = mid;
            } else {
                high = mid - 1;
            }
        }
        return ans;
    }

    /**
     * @Description: 26. 删除有序数组中的重复项
     * @author pwz
     * @date 2022/10/12 11:01
     * @param nums
     * @return int
     */
    public int removeDuplicates(int[] nums) { // 快慢指针
        if (nums.length == 0) return 0;
        int fast = 1, slow = 1;
        for (int i = 1;i < nums.length;i++) {
            if (nums[i] != nums[i - 1]) {
                nums[slow] = nums[fast];
                slow++;
            }
            fast++;
        }
        return slow;
    }

}