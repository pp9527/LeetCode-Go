package com.leetcode.pre;

/**
 * @ClassName Sorts
 * @Author PanWZ
 * @Data 2022/2/12 - 15:46
 * @Version: 1.8
 */
public class Sorts {
    public void selectionSort(int arr[]) {
        if (arr == null || arr.length < 2) return;
        for (int i = 0;i < arr.length;i++) {
            int max = i;
            for (int j = i;j < arr.length;j++) {
                if (arr[max] < arr[j]) max = j;
            }
            swp(arr, max, i);
        }
    }

    public void bubble(int arr[]) {
        if (arr == null || arr.length < 2) return;
        for (int i = 0;i < arr.length - 1;i++) {
            for (int j = 0;j < arr.length - i - 1;j++) {
                if (arr[j] < arr[j + 1]) swp1(arr, j,j + 1);
            }
        }
    }

    public void swp(int arr[], int i, int j) {
        int tem = arr[i];
        arr[i] = arr[j];
        arr[j] = tem;
    }

    public void swp1(int arr[], int i, int j) {
        arr[i] = arr[i] ^ arr[j];
        arr[j] = arr[i] ^ arr[j];
        arr[i] = arr[i] ^ arr[j];
    }
}
