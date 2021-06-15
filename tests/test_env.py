from unittest import TestCase

from environment.sql_env import SQLEnv


class TestEnv(TestCase):

    def test_no_html(self):
        env = SQLEnv(html=False)
        env.reset()
        state, r, d, info = env.step("SELECT * ALL")
        self.assertIn("ERROR", state)

    def test_solution(self):
        env = SQLEnv()
        env.reset()
        _, _, done, _ = env.step("1 UNION SELECT a FROM p")
        self.assertTrue(done)

    def test_html(self):
        env = SQLEnv(html=True)
        env.reset()
        state, _, done, _ = env.step("SELECT * 234")
        self.assertIn("html", state)
