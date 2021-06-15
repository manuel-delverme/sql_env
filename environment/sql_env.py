import http.client
import http.server
import sqlite3

import gym.envs
import numpy as np
import torch.distributions

import constants

torch.manual_seed(1)


class TextSpace(gym.spaces.Space):
    def __init__(self, vocab, length=None):
        self.vocab = vocab
        self.sequence_length = length
        self.vocab_length = len(vocab)
        super().__init__(shape=(1,), dtype=np.object_)

    def contains(self, x):
        raise NotImplemented

    def sample(self):
        raise NotImplemented


class SQLEnv(gym.Env):
    max_columns = constants.max_columns

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
            "near", "syntax", "error", "no", "such", "column", "incomplete", "input", "unrecognized", "token",
            'You', 'can', 'only', 'execute', 'one', 'statement', 'at', 'a', 'time.',
            *"SELECTs to the left and right of UNION do not have the same number of result columns".split(),
            *"Incorrect number of bindings supplied".split(),
            *"no such table".split(),
            "", "UNK"
        }

        self.observation_space = TextSpace(output_vocab)

        self.target_query_length = 9
        self.action_space = TextSpace(output_vocab, self.target_query_length)

        self.cursor.executemany("INSERT INTO users(id, username, firstname, surname, age, nationality, created_at) VALUES(NULL, ?, ?, ?, ?, ?, ?)", data)
        self.cursor.execute("CREATE TABLE comments(id INTEGER PRIMARY KEY AUTOINCREMENT, comment TEXT, time TEXT)")

        self.query_template = None
        self.reset()

    def step(self, input_query: str):
        assert isinstance(input_query, str)
        # We can use the same database as long as we change the hidden query

        cols = self.query_template.split(" FROM ")[0].count(',')
        if "firstname='{input}'" in self.query_template:
            escape = "'"
        elif "nationality=\"{input}\"" in self.query_template:
            escape = '"'
        else:
            escape = ''
        solution = ["1", escape, " UNION SELECT ", *([" NULL, "] * cols), "a", " FROM ", "p", " -- "]
        completed_input_query = "".join(solution[:-self.target_query_length]) + input_query

        http_code = http.client.OK
        content = ""
        found_flag = False
        query = self.query_template.format(input=completed_input_query)

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

        terminal = False

        reward = 0.
        # if http_code == http.client.INTERNAL_SERVER_ERROR:
        # else:
        #     reward = -.1
        if found_flag:
            reward = 1.
            terminal = True

        #for i, s in zip(input_query.split(), solution[-self.target_query_length:]):
        #    reward += 0.1 * float(i.isupper() == s.isupper())

        if ": syntax error" in content and "near " in content:
            content = "syntax error"

        if "no such column" in content:
            content = "no such column"

        if "no such table" in content:
            content = "no such table"

        if "unrecognized token" in content:
            content = "unrecognized token"

        if "SELECTs to the left and right of UNION do not have the same number of result columns" in content:
            content = "SELECTs to the left and right of UNION do not have the same number of result columns"

        if "Incorrect number of bindings supplied" in content:
            content = "Incorrect number of bindings supplied"

        out_tokens = content.split(" ")

        if content and set(out_tokens).difference(self.action_space.vocab):
            if not terminal:
                print("Query: ", input_query)
                print("returns: ", content)
            content = "UNK"

        return content, reward, terminal, {}

    def reset(self):
        np.random.seed(0)
        columns = np.random.randint(1, self.max_columns + 1)
        selected_columns = ", ".join(constants.columns[:columns])
        hidden_parameter = np.random.choice([
            "firstname='{input}'",
            "nationality=\"{input}\"",
            "age={input}",
        ])
        self.query_template = f"SELECT {selected_columns} FROM users WHERE {hidden_parameter}"
        # 1' UNION SELECT a, NULL, NULL FROM p --
        state, _, _, _ = self.step("1")
        return state
