import com.leetcode.CodeCapricorns;
import com.leetcode.Hot100;
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
    public void testMaxArea() {
        hot100.maxArea(new int[] {1,8,6,2,5,4,8,3,7});
    }
}