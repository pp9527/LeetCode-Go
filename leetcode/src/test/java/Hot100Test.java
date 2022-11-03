import com.leetcode.cur.Hot100;
import org.junit.Test;

/**
 * @author: pwz
 * @create: 2022/10/10 10:12
 * @Description:
 * @FileName: Hot100Test
 */
public class Hot100Test {

    private Hot100 hot100;

    {
        hot100 = new Hot100();
    }

    @Test
    public void testLengthOfLongestSubstring() {
        hot100.lengthOfLongestSubstring_1("abcddd");
    }


    @Test
    public void testCanFinish() {
        System.out.println(hot100.canFinish(3, new int[][]{{1, 0}, {2, 1}, {0, 2}}));
    }

    @Test
    public void testFindKthLargest() {
        hot100.findKthLargest(new int[]{3, 2, 1, 5, 6, 4}, 2);
    }

    @Test
    public void testMaximalSquare() {
        char[][] chars = {
                {'1', '0', '1', '0', '0'},
                {'1', '0', '1', '1', '1'},
                {'1', '1', '1', '1', '1'},
                {'1', '0', '0', '1', '0'}
        };
        int i = hot100.maximalSquare(chars);
    }
}