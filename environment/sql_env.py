import http.client
import http.server
import sqlite3

import gym.envs
import numpy as np
import torch.distributions

import config
import constants

torch.manual_seed(1)
from strsimpy.normalized_levenshtein import NormalizedLevenshtein


class TextSpace(gym.spaces.Space):
    def __init__(self, vocab, sequence_length, shape):
        self.vocab = vocab

        if vocab is None:
            self.vocab_length = None
        else:
            self.vocab = sorted(vocab)
            self.vocab_length = len(vocab)

        self.sequence_length = sequence_length
        # if length is None:
        #     self.sequence_length = 1
        #     shape = (1,)
        # else:
        #     self.sequence_length = length
        #     shape = (1, length)
        super().__init__(shape=shape, dtype=np.object_)

    def contains(self, x):
        raise NotImplemented

    def sample(self):
        raise NotImplemented


class SQLEnv(gym.Env):
    max_columns = constants.max_columns
    normalized_levenshtein = NormalizedLevenshtein()

    def render(self, mode='human'):
        pass

    def __init__(self):
        http.server.HTTPServer.allow_reuse_address = True
        self.connection = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)

        self.cursor = self.connection.cursor()
        self.cursor.execute("CREATE TABLE users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, firstname TEXT, surname TEXT, age INT, nationality TEXT, created_at TEXT)")
        self.cursor.execute("CREATE TABLE p(id INTEGER PRIMARY KEY AUTOINCREMENT, userid INT, a TEXT)")
        self.cursor.execute("INSERT INTO p(id, userid, a) VALUES(NULL, 1, 'accountnr:123456!')")

        data = []
        # To tell the agent what kind of outputs it can expect (XXX so far this is not an exhaustive list)
        output_vocab = {
            # "near", "syntax", "error", "no", "such", "column", "incomplete", "input", "unrecognized", "token",
            # 'You', 'can', 'only', 'execute', 'one', 'statement', 'at', 'a', 'time.',
            # *"SELECTs to the left and right of UNION do not have the same number of result columns".split(),
            "columns1",
            "columns2",
            "columns3",
            # "columns",
            # *"Incorrect number of bindings supplied".split(),
            # *"no such table".split(),
            "success",
            "UNK",
            "syntax_error",
            "firstname",
            "nationality",
            "age",
        }

        self.observation_space = TextSpace(output_vocab, 2, (1, ))

        self.target_query_length = config.complexity + self.max_columns - 1
        assert self.target_query_length > 1, "lvl1 is bugged"

        self.action_space = TextSpace(None, self.target_query_length, (1, self.target_query_length))

        self.cursor.executemany("INSERT INTO users(id, username, firstname, surname, age, nationality, created_at) VALUES(NULL, ?, ?, ?, ?, ?, ?)", data)
        self.cursor.execute("CREATE TABLE comments(id INTEGER PRIMARY KEY AUTOINCREMENT, comment TEXT, time TEXT)")

        self.query_template = None
        self.hidden_parameter = None
        self.reset()

    def step(self, user_query: str):
        assert isinstance(user_query, str)
        # We can use the same database as long as we change the hidden query

        solution = self.get_solution(user_query)
        solution_query = "".join(solution)
        # completed_input_query = input_query + precompleted_query

        http_code = http.client.OK

        content, found_flag = self.query_db(user_query)
        _, found_flag_ = self.query_db(solution_query)
        assert found_flag_

        terminal = False

        reward = -0.
        # if http_code == http.client.INTERNAL_SERVER_ERROR:
        # else:
        #     reward = -.1
        if found_flag:
            reward = 1.
            terminal = True

        similarity = self.normalized_levenshtein.similarity(user_query, " ".join(solution))
        # reward += 0.1 * similarity

        if ": syntax error" in content and "near " in content:
            content = "syntax_error"
            # content = "UNK"

        elif "no such column" in content:
            content = "no such column"
            content = "UNK"

        elif "no such table" in content:
            content = "no such table"
            content = "UNK"

        elif "unrecognized token" in content:
            content = "unrecognized token"
            content = "UNK"

        elif "SELECTs to the left and right of UNION do not have the same number of result columns" in content:
            query_columns = user_query.split(" FROM ")[0].count(',') + 1
            content = f"columns{query_columns}"
            # content = f"columns"
            # content = "UNK"

        elif "incomplete input" in content:
            content = "UNK"

        elif "Incorrect number of bindings supplied" in content:
            content = "Incorrect number of bindings supplied"
            content = "UNK"

        if not content:
            content = "success"

        out_tokens = content.split(" ")

        if set(out_tokens).difference(self.observation_space.vocab):
            if not terminal:
                print("Query: ", user_query)
                print("returns: ", content)
            content = "UNK"

        response = " ".join([self.hidden_parameter.split("=")[0], content])
        return response, reward, terminal, {
            'similarity': similarity,
            'columns': self.query_template.split(" FROM ")[0].count(','),
            'template': self.query_template,
            'solved': found_flag,
        }

    def query_db(self, user_query):
        query = self.query_template.format(input=user_query)
        content = ""
        found_flag = False
        try:
            self.cursor.execute(query)
            for some in self.cursor.fetchall():
                for f in some:
                    # content += str(f) + ";"  # "{'-' if f is None else f}\n"
                    f = str(f)
                    if 'account' in f and '!' in f:
                        found_flag = True
        except Exception as ex:
            content += str(ex)
            http_code = http.client.INTERNAL_SERVER_ERROR
        return content, found_flag

    def get_similarity(self, input_query, solution):
        similarity = 0
        for i, s in zip(input_query.split(), solution[:-self.target_query_length]):
            similarity += float(i.strip() == s.strip()) / self.target_query_length
        return similarity / self.target_query_length

    def get_solution(self, input_query):
        cols = self.query_template.split(" FROM ")[0].count(',')
        if "firstname='{input}'" in self.query_template:
            escape = "'"
        elif "nationality=\"{input}\"" in self.query_template:
            escape = '"'
        else:
            escape = '1'
        if escape in input_query:
            input_query.format(escape=escape)
        solution = [escape, " UNION SELECT ", *([" NULL, "] * cols), " a ", " FROM ", " p ", " -- "]
        return solution

    def reset(self):
        columns = np.random.randint(1, self.max_columns + 1)
        selected_columns = ", ".join(constants.columns[:columns])
        self.hidden_parameter = np.random.choice([
            "firstname='{input}'",
            "nationality=\"{input}\"",
            "age={input}",
        ])
        self.query_template = f"SELECT {selected_columns} FROM users WHERE {self.hidden_parameter}"
        state, _, _, _ = self.step("--")
        return state
