package com.leetcode.cur;


/**
 * @author: pwz
 * @create: 2022/10/27 12:16
 * @Description: 每日一题
 * @FileName: DailyTopic
 */
public class DailyTopic {

    /**
     * @Description: 1822. 数组元素积的符号
     * @author pwz
     * @date 2022/10/27 12:29
     */
    public int arraySign(int[] nums) {
        int flag = 1;
        for (int i : nums) {
            if (i == 0) return 0;
            else if (i < 0) flag = -flag;
        }
        return flag;
    }
}