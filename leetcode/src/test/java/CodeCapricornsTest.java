import com.leetcode.cur.CodeCapricorns;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

/**
 * @author: pwz
 * @create: 2022/10/9 17:10
 * @Description:
 * @FileName: SolutionTest
 */
public class CodeCapricornsTest {

    private CodeCapricorns codeCapricorns = new CodeCapricorns();

    @Test
    public void testFourSumCount() {
        codeCapricorns.fourSumCount(new int[]{-1, -1}, new int[]{-1, 1}, new int[]{-1, 1}, new int[]{1, -1});
    }

    @Test
    public void testFourSum1() {
        codeCapricorns.fourSum(new int[]{2, 2, 2, 2, 2}, 8);
    }
}