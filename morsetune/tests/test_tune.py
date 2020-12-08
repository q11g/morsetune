from unittest import TestCase

import morsetune

class TestMorseTune(TestCase):
    def test_is_tune(self):
        tune = morsetune.MorseTune()
        audio = tune.convert('...---...')
        self.assertEqual(26400, len(audio))