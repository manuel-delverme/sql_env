import http.client
import http.server
import sqlite3
import xml.etree.ElementTree

import gym.envs
import numpy as np
import torch.distributions

import constants

torch.manual_seed(1)


class TextSpace(gym.spaces.Space):
    def __init__(self, vocab):
        self.vocab = vocab
        self.n = len(vocab)
        super().__init__(shape=(1,), dtype=np.object_)

    def contains(self, x):
        raise NotImplemented

    def sample(self):
        raise NotImplemented


class SQLEnv(gym.Env):
    def render(self, mode='human'):
        pass

    def __init__(self, max_columns):
        self.max_columns = max_columns

        http.server.HTTPServer.allow_reuse_address = True
        self.connection = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)

        self.cursor = self.connection.cursor()
        self.cursor.execute("CREATE TABLE users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, firstname TEXT, surname TEXT, age INT, nationality TEXT, created_at TEXT)")
        self.cursor.execute("CREATE TABLE private(id INTEGER PRIMARY KEY AUTOINCREMENT, userid INT, account TEXT)")

        self.cursor.execute("INSERT INTO private(id, userid, account) VALUES(NULL, 1, 'accountnr:123456')")

        data = []
        # To tell the agent what kind of outputs it can expect (XXX so far this is not an exhaustive list)
        output_vocab = {"UNION", "SELECT", "*", "FROM", "users", "1", "ERROR", "", "private", "account", "None", "and"}

        for idx, row in enumerate(xml.etree.ElementTree.fromstring(constants.USERS_XML).findall("user")):
            row = row.findtext("username"), row.findtext("firstname"), row.findtext("surname"), row.findtext("age"), row.findtext("nationality"), row.findtext("created_at")
            data.append(row)
            output_vocab.update(row)
            output_vocab.update(str(idx + 1))

        self.observation_space = TextSpace(output_vocab)
        self.action_space = TextSpace(output_vocab)

        self.cursor.executemany("INSERT INTO users(id, username, firstname, surname, age, nationality, created_at) VALUES(NULL, ?, ?, ?, ?, ?, ?)", data)
        self.cursor.execute("CREATE TABLE comments(id INTEGER PRIMARY KEY AUTOINCREMENT, comment TEXT, time TEXT)")

        self.query_template = None
        self.reset()

    def step(self, input_query: str):
        assert isinstance(input_query, str)
        # We can use the same database as long as we change the hidden query

        http_code = http.client.OK
        content = ""
        query = self.query_template.format(input=input_query)
        try:
            self.cursor.execute(query)
            for some in self.cursor.fetchall():
                for f in some:
                    content += str(f) + ";"  # "{'-' if f is None else f}\n"
        except Exception as ex:
            # content += str(ex)
            content += "ERROR"
            http_code = http.client.INTERNAL_SERVER_ERROR

        terminal = False

        if http_code == http.client.INTERNAL_SERVER_ERROR:
            reward = -1.
        else:
            reward = -.1
        if 'account' in content:
            reward += 1
            terminal = True
        return content, reward, terminal, {}

    def reset(self):
        columns = np.random.randint(0, self.max_columns)
        selected_columns = ", ".join(constants.columns[:columns])
        hidden_parameter = np.random.choice([
            "firstname='{input}'",
            "nationality=\"{input}\"",
            "age={input}",
        ])
        self.query_template = f"SELECT {selected_columns} FROM users WHERE = {hidden_parameter}"
        state, _, _, _ = self.step("1")
        return state
