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
        dailyTopic.canConstruct("aaa", "abaca");
    }
}