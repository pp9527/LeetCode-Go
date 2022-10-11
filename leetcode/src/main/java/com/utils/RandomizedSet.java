package com.utils;

import java.util.*;

/**
 * @author: pwz
 * @create: 2022/10/11 9:49
 * @Description:
 * @FileName: RandomizedSet
 */
public class RandomizedSet {
    /**
     * O(1)时间插入、删除、获取随机元素
     * 变长数组可以在 O(1)的时间内完成获取随机元素操作
     * 哈希表可以在 O(1)的时间内完成插入和删除操作以及判断元素是否存在
     */
    List<Integer> nums;
    Map<Integer, Integer> indices;
    Random random;

    public RandomizedSet() {
        nums = new ArrayList<>();
        indices = new HashMap<>();
        random = new Random();
    }

    public boolean insert(int val) {
        if (indices.containsKey(val)) {
            return false;
        }
        int index = nums.size();
        nums.add(val);
        indices.put(val, index);
        return true;
    }

    public boolean remove(int val) {
        if (!indices.containsKey(val)) {
            return false;
        }
        nums.set(indices.get(val), nums.get(nums.size() - 1));
        indices.put(nums.get(nums.size() - 1), indices.get(val));
        nums.remove(nums.size() - 1);
        indices.remove(val);
        return true;
    }

    public int getRandom() {
        return nums.get(random.nextInt(nums.size()));
    }
}