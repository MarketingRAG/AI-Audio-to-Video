import unittest
import sys
import os

# Adjust path to import from the parent directory (project root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from audio_to_youtube_shorts import (
    format_timedelta,
    parse_timestamp,
    clean_json_response,
    sanitize_filename
)

class TestTimeUtils(unittest.TestCase):
    def test_format_timedelta_zero(self):
        self.assertEqual(format_timedelta(0), "00:00:00,000")

    def test_format_timedelta_whole_seconds(self):
        self.assertEqual(format_timedelta(10), "00:00:10,000")

    def test_format_timedelta_seconds_milliseconds(self):
        self.assertEqual(format_timedelta(15.75), "00:00:15,750")

    def test_format_timedelta_minutes_seconds_milliseconds(self):
        self.assertEqual(format_timedelta(90.5), "00:01:30,500") # 1 minute, 30 seconds, 500 ms

    def test_format_timedelta_hours_minutes_seconds_milliseconds(self):
        self.assertEqual(format_timedelta(3661.5), "01:01:01,500") # 1 hour, 1 minute, 1 second, 500 ms
        self.assertEqual(format_timedelta(7200), "02:00:00,000") # 2 hours

    def test_parse_timestamp_valid(self):
        self.assertEqual(parse_timestamp("00:01:30,500"), 90.5)
        self.assertEqual(parse_timestamp("01:01:01,000"), 3661.0)
        self.assertEqual(parse_timestamp("00:00:00,000"), 0.0)

    def test_parse_timestamp_invalid_format(self):
        self.assertIsNone(parse_timestamp("00:01:30")) # Missing milliseconds
        self.assertIsNone(parse_timestamp("00-01-30,500")) # Wrong separator for time parts
        self.assertIsNone(parse_timestamp("00:01:30.500")) # Wrong separator for milliseconds
        self.assertIsNone(parse_timestamp("00:01")) # Too few parts
        self.assertIsNone(parse_timestamp("00:01:02:03,500")) # Too many parts
        self.assertIsNone(parse_timestamp("abc")) # Completely invalid

    def test_parse_timestamp_invalid_values(self):
        self.assertIsNone(parse_timestamp("00:01:60,000")) # Seconds > 59
        self.assertIsNone(parse_timestamp("00:60:30,000")) # Minutes > 59
        self.assertIsNone(parse_timestamp("00:00:00,1000")) # Milliseconds > 999
        self.assertIsNone(parse_timestamp("-01:00:00,000")) # Negative hours (though int() would parse, our logic might not intend this)
        # The current parse_timestamp implementation checks for 0-99 hours, 0-59 min/sec, 0-999 ms.
        # So negative values, if not caught by int conversion, are caught by range check.

    def test_parse_timestamp_not_string(self):
        self.assertIsNone(parse_timestamp(123)) # Non-string input
        self.assertIsNone(parse_timestamp(None))


class TestStringUtils(unittest.TestCase):
    def test_clean_json_response_with_markdown(self):
        self.assertEqual(clean_json_response('```json\n{"key": "value"}\n```'), '{"key": "value"}')
        self.assertEqual(clean_json_response('```json{"key": "value"}```'), '{"key": "value"}')
        self.assertEqual(clean_json_response('{"key": "value"}'), '{"key": "value"}') # No markdown

    def test_clean_json_response_trailing_comma_object(self):
        self.assertEqual(clean_json_response('{"key": "value",}'), '{"key": "value"}')
        self.assertEqual(clean_json_response('{"k1": "v1", "k2": "v2",}'), '{"k1": "v1", "k2": "v2"}')

    def test_clean_json_response_trailing_comma_array(self):
        self.assertEqual(clean_json_response('[1, 2, 3,]'), '[1, 2, 3]')
        self.assertEqual(clean_json_response('["a", "b",]'), '["a", "b"]')
        
    def test_clean_json_response_multiple_issues(self):
        self.assertEqual(clean_json_response('```json\n{"key": "value", "arr": [1,2,],}\n```'), '{"key": "value", "arr": [1,2]}')

    def test_clean_json_response_already_clean(self):
        self.assertEqual(clean_json_response('{"key": "value"}'), '{"key": "value"}')
        self.assertEqual(clean_json_response('[1, 2, 3]'), '[1, 2, 3]')

    def test_clean_json_response_nested_trailing_commas(self):
        self.assertEqual(clean_json_response('{"outer": {"inner": "val",}, "arr": [1, {"a": "b",},],}'), '{"outer": {"inner": "val"}, "arr": [1, {"a": "b"}]}')
    
    def test_sanitize_filename_safe(self):
        self.assertEqual(sanitize_filename("This is a safe filename 123.mp4"), "This is a safe filename 123.mp4")

    def test_sanitize_filename_unsafe_chars(self):
        self.assertEqual(sanitize_filename('a<b>c:d/e\\f|g?h*i.txt'), "a_b_c_d_e_f_g_h_i.txt")

    def test_sanitize_filename_leading_trailing_whitespace(self):
        self.assertEqual(sanitize_filename("  filename with spaces  "), "filename with spaces")

    def test_sanitize_filename_truncation(self):
        long_name = "a" * 150
        expected_name = "a" * 100
        self.assertEqual(sanitize_filename(long_name), expected_name)
        self.assertEqual(len(sanitize_filename(long_name)), 100)

    def test_sanitize_filename_empty(self):
        self.assertEqual(sanitize_filename(""), "")
        self.assertEqual(sanitize_filename("   "), "") # Only whitespace should result in empty after strip

    def test_sanitize_filename_mixed(self):
        self.assertEqual(sanitize_filename("  a<b>c:d/e\\f|g?h*i with spaces.txt  "), "a_b_c_d_e_f_g_h_i with spaces.txt")


if __name__ == '__main__':
    unittest.main()
