package com.leetcode.cur;


import java.util.List;

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

    /**
     * @Description: 1773. 统计匹配检索规则的物品数量
     * @author pwz
     * @date 2022/10/29 9:37
     */
    public int countMatches(List<List<String>> items, String ruleKey, String ruleValue) {
        int count = 0;
        for (List<String> item : items) {
            if (ruleKey.equals("type")) {
                if (item.get(0).equals(ruleValue)) count++;
            } else if (ruleKey.equals("color")) {
                if (item.get(1).equals(ruleValue)) count++;
            } else {
                if (item.get(2).equals(ruleValue)) count++;
            }
        }
        return count;
    }







}