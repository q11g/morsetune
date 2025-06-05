from unittest import TestCase
import numpy as np

from morsetune import *

class TestMorseTune(TestCase):
    def test_is_tune(self):
        tune = morsetune.MorseTune()
        audio = tune.convert('...---...')
        self.assertEqual(26400, len(audio))

    def test_convert(self):
        message = 'hello'
        morse = encode(message)
        audio = convert(morse)
        # 验证音频数据是numpy数组
        self.assertIsInstance(audio, np.ndarray)
        # 验证音频长度大于0
        self.assertGreater(len(audio), 0)

    def test_encode_single_character(self):
        """测试单个字符的编码"""
        self.assertEqual(encode('A'), '.-')
        self.assertEqual(encode('B'), '-...')
        self.assertEqual(encode('E'), '.')
        self.assertEqual(encode('T'), '-')
        self.assertEqual(encode('S'), '...')
        self.assertEqual(encode('O'), '---')

    def test_encode_numbers(self):
        """测试数字的编码"""
        self.assertEqual(encode('1'), '.----')
        self.assertEqual(encode('2'), '..---')
        self.assertEqual(encode('5'), '.....')
        self.assertEqual(encode('0'), '-----')

    def test_encode_word(self):
        """测试单词的编码"""
        self.assertEqual(encode('SOS'), '... --- ...')
        self.assertEqual(encode('HELLO'), '.... . .-.. .-.. ---')
        self.assertEqual(encode('WORLD'), '.-- --- .-. .-.. -..')

    def test_encode_case_insensitive(self):
        """测试大小写不敏感"""
        self.assertEqual(encode('hello'), encode('HELLO'))
        self.assertEqual(encode('SoS'), encode('SOS'))
        self.assertEqual(encode('WoRlD'), encode('WORLD'))

    def test_encode_with_spaces(self):
        """测试带空格的编码"""
        self.assertEqual(encode('HELLO WORLD'), '.... . .-.. .-.. --- / .-- --- .-. .-.. -..')
        self.assertEqual(encode('A B'), '.- / -...')

    def test_encode_unknown_characters(self):
        """测试未知字符的处理"""
        # 未知字符应该被忽略（返回空字符串）
        result = encode('A@B')
        self.assertEqual(result, '.-  -...')  # @字符被忽略，留下两个空格

    def test_encode_empty_string(self):
        """测试空字符串"""
        self.assertEqual(encode(''), '')

    def test_morse_tune_class_initialization(self):
        """测试MorseTune类的初始化"""
        tune = MorseTune()
        self.assertEqual(tune.SPS, 8000)
        self.assertEqual(tune.FREQ, 750)
        self.assertEqual(tune.WPM, 12)
        self.assertEqual(tune.AUDIO_PADDING, 0.5)
        self.assertEqual(tune.CLICK_SMOOTH, 2)

    def test_morse_tune_convert_simple(self):
        """测试MorseTune类的convert方法"""
        tune = MorseTune()
        audio = tune.convert('.-')  # 字母A
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_morse_tune_convert_complex(self):
        """测试复杂摩尔斯代码的转换"""
        tune = MorseTune()
        audio = tune.convert('... --- ...')  # SOS
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_audio_properties(self):
        """测试音频属性"""
        tune = MorseTune()
        audio = tune.convert('.')
        # 验证音频数据类型
        self.assertTrue(audio.dtype in [np.float32, np.float64])
        # 验证音频值在合理范围内
        self.assertTrue(np.all(np.abs(audio) <= 1.0))

    def test_different_morse_lengths(self):
        """测试不同长度的摩尔斯代码"""
        tune = MorseTune()
        
        # 测试点（短）
        audio_dot = tune.convert('.')
        # 测试划（长）
        audio_dash = tune.convert('-')
        
        # 划的音频应该比点的音频长
        self.assertGreater(len(audio_dash), len(audio_dot))

    def test_audio_consistency(self):
        """测试音频一致性"""
        tune = MorseTune()
        
        # 相同的摩尔斯代码应该产生相同的音频
        audio1 = tune.convert('...')
        audio2 = tune.convert('...')
        
        np.testing.assert_array_equal(audio1, audio2)

    def test_global_convert_function(self):
        """测试全局convert函数"""
        audio = convert('.-')
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_play_function_return_type(self):
        """测试play函数的返回类型"""
        audio = convert('.')
        result = play(audio)
        # play函数应该返回IPython.display.Audio对象
        self.assertIsNotNone(result)

    def test_full_alphabet(self):
        """测试完整字母表的转换"""
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        morse = encode(alphabet)
        audio = convert(morse)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
        # 字母表的音频应该比较长
        self.assertGreater(len(audio), 100000)

    def test_full_numbers(self):
        """测试完整数字的转换"""
        numbers = '0123456789'
        morse = encode(numbers)
        audio = convert(morse)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_mixed_content(self):
        """测试混合内容（字母、数字、空格）"""
        mixed = 'HELLO 123 WORLD'
        morse = encode(mixed)
        audio = convert(morse)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_edge_cases(self):
        """测试边界情况"""
        # 单个字符
        self.assertGreater(len(convert(encode('A'))), 0)
        
        # 只有空格
        morse = encode('   ')
        self.assertIn('/', morse)
        
        # 很长的消息
        long_message = 'A' * 100
        morse = encode(long_message)
        audio = convert(morse)
        self.assertGreater(len(audio), 0)

    def test_decode_single_character(self):
        """测试解码单个字符"""
        self.assertEqual(decode('.-'), 'A')
        self.assertEqual(decode('-...'), 'B')
        self.assertEqual(decode('.'), 'E')
        self.assertEqual(decode('-'), 'T')
        self.assertEqual(decode('...'), 'S')
        self.assertEqual(decode('---'), 'O')

    def test_decode_numbers(self):
        """测试解码数字"""
        self.assertEqual(decode('.----'), '1')
        self.assertEqual(decode('..---'), '2')
        self.assertEqual(decode('.....'), '5')
        self.assertEqual(decode('-----'), '0')

    def test_decode_word(self):
        """测试解码单词"""
        self.assertEqual(decode('... --- ...'), 'SOS')
        self.assertEqual(decode('.... . .-.. .-.. ---'), 'HELLO')
        self.assertEqual(decode('.-- --- .-. .-.. -..'), 'WORLD')

    def test_decode_with_spaces(self):
        """测试解码带空格的文本"""
        self.assertEqual(decode('.... . .-.. .-.. --- / .-- --- .-. .-.. -..'), 'HELLO WORLD')
        self.assertEqual(decode('.- / -...'), 'A B')
        self.assertEqual(decode('... --- ... / .... . .-.. .--.'), 'SOS HELP')

    def test_decode_unknown_codes(self):
        """测试解码未知摩尔斯代码"""
        # 未知代码应该被忽略
        result = decode('.- @@ -...')
        self.assertEqual(result, 'AB')  # @@是无效的摩尔斯代码，被忽略，但在同一个单词内

    def test_decode_empty_string(self):
        """测试解码空字符串"""
        self.assertEqual(decode(''), '')

    def test_decode_only_spaces(self):
        """测试解码只有空格分隔符的字符串"""
        self.assertEqual(decode('/'), ' ')
        self.assertEqual(decode('/ /'), '  ')

    def test_decode_case_consistency(self):
        """测试解码结果的大小写一致性"""
        # decode函数应该返回大写字母
        self.assertEqual(decode('.- -...'), 'AB')
        self.assertEqual(decode('.... . .-.. .-.. ---'), 'HELLO')

    def test_decode_multiple_words(self):
        """测试解码多个单词"""
        morse = '... --- ... / .... . .-.. .--. / -- .'
        expected = 'SOS HELP ME'
        self.assertEqual(decode(morse), expected)

    def test_decode_mixed_content(self):
        """测试解码混合内容（字母、数字、空格）"""
        original = 'ABC 123 XYZ'
        encoded = encode(original)
        decoded = decode(encoded)
        self.assertEqual(decoded, original)

    def test_decode_edge_cases(self):
        """测试解码边界情况"""
        # 单个点和划
        self.assertEqual(decode('.'), 'E')
        self.assertEqual(decode('-'), 'T')
        
        # 最长的摩尔斯代码
        self.assertEqual(decode('-----'), '0')  # 5个划
        self.assertEqual(decode('.----'), '1')  # 1个点4个划
        
        # 多个连续空格分隔符
        self.assertEqual(decode('.- / / -...'), 'A  B')

    def test_decode_full_alphabet(self):
        """测试解码完整字母表"""
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        encoded = encode(alphabet)
        decoded = decode(encoded)
        self.assertEqual(decoded, alphabet)

    def test_decode_full_numbers(self):
        """测试解码完整数字"""
        numbers = '0123456789'
        encoded = encode(numbers)
        decoded = decode(encoded)
        self.assertEqual(decoded, numbers)

    def test_encode_decode_round_trip(self):
        """测试编码和解码的往返转换"""
        # 单词测试
        original = 'HELLO'
        encoded = encode(original)
        decoded = decode(encoded)
        self.assertEqual(decoded, original)

        # 句子测试
        original = 'HELLO WORLD'
        encoded = encode(original)
        decoded = decode(encoded)
        self.assertEqual(decoded, original)

        # 数字测试
        original = '12345'
        encoded = encode(original)
        decoded = decode(encoded)
        self.assertEqual(decoded, original)

        # 混合测试
        original = 'SOS 123'
        encoded = encode(original)
        decoded = decode(encoded)
        self.assertEqual(decoded, original)

    def test_encode_function_consistency(self):
        """测试encode函数的一致性"""
        message = 'HELLO WORLD'
        
        # 往返转换测试
        encoded = encode(message)
        decoded = decode(encoded)
        self.assertEqual(decoded, message)

