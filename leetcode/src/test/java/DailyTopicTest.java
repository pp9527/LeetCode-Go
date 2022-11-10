import com.leetcode.cur.DailyTopic;
import org.junit.Test;

/**
 * @author: pwz
 * @create: 2022/10/27 12:59
 * @Description:
 * @FileName: DailyTopicTest
 */
public class DailyTopicTest {

    private DailyTopic dailyTopic = new DailyTopic();

    @Test
    public void testCanConstruct() {
//        dailyTopic.canConstruct("aaa", "abaca");
    }

    @Test
    public void testLetterCasePermutation() {
        dailyTopic.letterCasePermutation("a1b1");
    }

    @Test
    public void testArrayStringsAreEqual() {
        dailyTopic.arrayStringsAreEqual(new String[]{"ab", "c"}, new String[]{"a", "bc"});
    }

    @Test
    public void test() {
        dailyTopic.shortestPathAllKeys(new String[]{"@..aA","..B#.","....b"});
    }
}