from unittest import TestCase

import ppo.model
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
        cols = env.query_template.split(" FROM ")[0].count(',')
        if "firstname='{input}'" in env.query_template:
            escape = "'"
        elif "nationality=\"{input}\"" in env.query_template:
            escape = '"'
        else:
            escape = ''

        solution = ["1", escape, " UNION SELECT ", *([" NULL, "] * cols), "a", " FROM ", "p", " -- "]

        for s in solution:
            self.assertIn(s, ppo.model.Policy.output_vocab)

        _, _, done, _ = env.step("".join(solution))
        self.assertTrue(done)

    def test_html(self):
        env = SQLEnv(html=True)
        env.reset()
        state, _, done, _ = env.step("SELECT * 234")
        self.assertIn("html", state)
