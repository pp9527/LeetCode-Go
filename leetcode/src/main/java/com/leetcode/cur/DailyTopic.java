package com.leetcode.cur;


import sun.text.resources.cldr.ti.FormatData_ti;

import java.util.*;

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

    /**
     * @Description: 784. 字母大小写全排列
     * @author pwz
     * @date 2022/10/30 13:10
     */
    public List<String> letterCasePermutation(String s) {
        List<String> res = new ArrayList<>();
        dfs(s.toCharArray(), 0, res);
        return res;
    }

    private void dfs(char[] arr, int index, List<String> res) {
        while (index < arr.length && Character.isDigit(arr[index])) {
            index++;
        }
        if (index == arr.length) {
            res.add(new String(arr));
            return;
        }
//        arr[index] ^= 32;
        dfs(arr, index + 1, res);
        arr[index] ^= 32;
        dfs(arr, index + 1, res);
    }

    /**
     * @Description: 481. 神奇字符串
     * @author pwz
     * @date 2022/10/31 9:50
     */
    public int magicalString(int n) {
        if (n < 4) return 1;
        int[] arr = new int[n];
        arr[0] = 1;
        arr[1] = 2;
        arr[2] = 2;
        int i = 2, j = 3, res = 0;
        while (j < n) {
            int num = 3 - arr[j - 1];
            int size = arr[i];
            while (size > 0 && j < n) {
                arr[j++] = num;
                if (num == 1) {
                    res++;
                }
                size--;
            }
            i++;
        }
        return res;
    }

    /**
     * @Description: 1662. 检查两个字符串数组是否相等
     * @author pwz
     * @date 2022/11/1 9:53
     */
    public boolean arrayStringsAreEqual(String[] word1, String[] word2) {
        StringBuilder sb1 = new StringBuilder();
        for (String s : word1) {
            sb1.append(s);
        }
        StringBuilder sb2 = new StringBuilder();
        for (String s : word2) {
            sb2.append(s);
        }
        return sb1.toString().compareTo(sb2.toString()) == 0;
    }

    /**
     * @Description: 1620. 网络信号最好的坐标
     * @author pwz
     * @date 2022/11/2 10:09
     */
    public int[] bestCoordinate(int[][] towers, int radius) {
        int xMax = Integer.MIN_VALUE, yMax = Integer.MIN_VALUE;
        int xMin = Integer.MAX_VALUE, yMin = Integer.MAX_VALUE;
        for (int[] i : towers) {
            xMax = Math.max(i[0], xMax);
            yMax = Math.max(i[1], yMax);
            xMin = Math.min(i[0], xMin);
            yMin = Math.min(i[1], yMin);
        }
        int cx = 0, cy = 0;
        double maxPower = 0;
        for (int i = xMin; i <= xMax; i++) {
            for (int j = yMin; j <= yMax; j++) {
                int[] coordinate = {i, j};
                double power = 0;
                for (int[] tower : towers) {
                    double distance = getDistance(tower, coordinate);
                    if (distance <= radius * radius) {
                        power += Math.floor(tower[2] / (1 + Math.sqrt(distance)));
                    }
                }
                if (power > maxPower) {
                    cx = i;
                    cy = j;
                    maxPower = power;
                }
            }
        }
        return new int[]{cx, cy};
    }

    private int getDistance(int[] tower, int[] coordinate) {
        return (tower[0] - coordinate[0]) * (tower[0] - coordinate[0])
                + (tower[1] - coordinate[1]) * (tower[1] - coordinate[1]);
    }

    /**
     * @Description: 1668. 最大重复子字符串
     * @author pwz
     * @date 2022/11/3 10:25
     */
    public int maxRepeating(String sequence, String word) {
        int res = 0;
        String str = word;
        while (sequence.contains(str)) {
            str += word;
            res++;
        }
        return res;
    }

    /**
     * @Description: 754. 到达终点数字
     * @author pwz
     * @date 2022/11/4 10:23
     */
    public int reachNumber(int target) {
        int sum = 0;
        target = Math.abs(target);
        int k = 0;
        while (sum < target || (sum - target) % 2 != 0) {
            k++;
            sum += k;
        }
        return k;
    }

    /**
     * @Description: 864. 获取所有钥匙的最短路径
     * @author pwz
     * @date 2022/11/10 10:41
     */
    public int shortestPathAllKeys(String[] grid) {
        int N = 30, K = 6, INF = 0x3f3f3f3f;
        int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        int[][][] dist = new int[N][N][1 << K];
        int n = grid.length, m = grid[0].length(), cnt = 0;
        Deque<int[]> deque = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                Arrays.fill(dist[i][j], INF);
                char ch = grid[i].charAt(j);
                if (ch == '@') {
                    deque.addLast(new int[]{i, j, 0});
                    dist[i][j][0] = 0;
                } else if (ch >= 'a' && ch <= 'z') cnt++;
            }
        }
        while (!deque.isEmpty()) {
            int[] info = deque.pollFirst();
            int x = info[0], y = info[1], cur = info[2], step = dist[x][y][cur];
            for (int[] dir : dirs) {
                int nx = x + dir[0], ny = y + dir[1];
                if (nx < 0 || nx >= n || ny < 0 || ny >= m) continue;
                char ch = grid[nx].charAt(ny);
                if (ch == '#') continue;
                if ((ch >= 'A' && ch <= 'Z') && (cur >> (ch - 'A') & 1) == 0) continue;
                int nCur = cur;
                if (ch >= 'a' && ch <= 'z') nCur |= 1 << (ch - 'a');
                if (nCur == (1 << cnt) - 1) return step + 1;
                if (step + 1 >= dist[nx][ny][nCur]) continue;
                dist[nx][ny][nCur] = step + 1;
                deque.addLast(new int[]{nx, ny, nCur});
            }
        }
        return -1;
    }
}