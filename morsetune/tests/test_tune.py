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
        morse = translate(message)
        audio = convert(morse)
        # 验证音频数据是numpy数组
        self.assertIsInstance(audio, np.ndarray)
        # 验证音频长度大于0
        self.assertGreater(len(audio), 0)

    def test_translate_single_character(self):
        """测试单个字符的翻译"""
        self.assertEqual(translate('A'), '.-')
        self.assertEqual(translate('B'), '-...')
        self.assertEqual(translate('E'), '.')
        self.assertEqual(translate('T'), '-')
        self.assertEqual(translate('S'), '...')
        self.assertEqual(translate('O'), '---')

    def test_translate_numbers(self):
        """测试数字的翻译"""
        self.assertEqual(translate('1'), '.----')
        self.assertEqual(translate('2'), '..---')
        self.assertEqual(translate('5'), '.....')
        self.assertEqual(translate('0'), '-----')

    def test_translate_word(self):
        """测试单词的翻译"""
        self.assertEqual(translate('SOS'), '... --- ...')
        self.assertEqual(translate('HELLO'), '.... . .-.. .-.. ---')
        self.assertEqual(translate('WORLD'), '.-- --- .-. .-.. -..')

    def test_translate_case_insensitive(self):
        """测试大小写不敏感"""
        self.assertEqual(translate('hello'), translate('HELLO'))
        self.assertEqual(translate('SoS'), translate('SOS'))
        self.assertEqual(translate('WoRlD'), translate('WORLD'))

    def test_translate_with_spaces(self):
        """测试带空格的翻译"""
        self.assertEqual(translate('HELLO WORLD'), '.... . .-.. .-.. --- / .-- --- .-. .-.. -..')
        self.assertEqual(translate('A B'), '.- / -...')

    def test_translate_unknown_characters(self):
        """测试未知字符的处理"""
        # 未知字符应该被忽略（返回空字符串）
        result = translate('A@B')
        self.assertEqual(result, '.-  -...')  # @字符被忽略，留下两个空格

    def test_translate_empty_string(self):
        """测试空字符串"""
        self.assertEqual(translate(''), '')

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
        morse = translate(alphabet)
        audio = convert(morse)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
        # 字母表的音频应该比较长
        self.assertGreater(len(audio), 100000)

    def test_full_numbers(self):
        """测试完整数字的转换"""
        numbers = '0123456789'
        morse = translate(numbers)
        audio = convert(morse)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_mixed_content(self):
        """测试混合内容（字母、数字、空格）"""
        mixed = 'HELLO 123 WORLD'
        morse = translate(mixed)
        audio = convert(morse)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_edge_cases(self):
        """测试边界情况"""
        # 单个字符
        self.assertGreater(len(convert(translate('A'))), 0)
        
        # 只有空格
        morse = translate('   ')
        self.assertIn('/', morse)
        
        # 很长的消息
        long_message = 'A' * 100
        morse = translate(long_message)
        audio = convert(morse)
        self.assertGreater(len(audio), 0)

