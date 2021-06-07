import http.client
import http.server
import re
import sqlite3
import subprocess
import time
import traceback
import urllib.parse
import urllib.request
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

    def shape(self):
        # shapes[key] = box.shape
        return None


def fancy_split(query):
    query_ = []
    for q in query.split():
        if q.islower():
            query_.extend(q)
        else:
            query_.append(q)
    return query_


class SQLEnv(gym.Env):
    def render(self, mode='human'):
        pass

    def __init__(self, html, second_order=False):
        self.get_help_reward = True
        self.html = html
        self.second_order = second_order
        http.server.HTTPServer.allow_reuse_address = True
        self.connection = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)

        self._slept = None

        def fake_sleep(time):
            print(f"sleep {time}")
            self._slept = time

        self.connection.create_function("sleep", narg=1, func=fake_sleep)
        self.cursor = self.connection.cursor()
        self.cursor.execute("CREATE TABLE users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, name TEXT, surname TEXT, password TEXT)")

        data = []
        output_vocab = {"UNION", "SELECT", "*", "FROM", "users", "1", "ERROR", ""}

        for idx, row in enumerate(xml.etree.ElementTree.fromstring(constants.USERS_XML).findall("user")):
            row = row.findtext("username"), row.findtext("name"), row.findtext("surname"), row.findtext("password")
            data.append(row)
            output_vocab.update(row)
            output_vocab.update(str(idx + 1))

        self.observation_space = TextSpace(output_vocab)
        self.action_space = TextSpace(output_vocab)
        # output_vocab = {"UNION", "SELECT", "*", "FROM", "users", "1", "ERROR", ""}

        self.cursor.executemany("INSERT INTO users(id, username, name, surname, password) VALUES(NULL, ?, ?, ?, ?)", data)
        self.cursor.execute("CREATE TABLE comments(id INTEGER PRIMARY KEY AUTOINCREMENT, comment TEXT, time TEXT)")

    def step(self, query: str):
        assert isinstance(query, str)
        code = http.client.OK
        content = ""
        self._slept = None
        if self.second_order:
            self.cursor.execute("""
            SELECT birthday 
            FROM users
            WHERE strftime('%m-%d',date('now', 'localtime')) = bday
            """)



        # TODO Cursor doesn't owkr.' fix me.
        try:
            # self.cursor.execute("SELECT id, username, name, surname FROM users WHERE id=" + query)
            start = time.time()
            self.cursor.execute(query)
            runtime = time.time() - start

            content += "<div><span>Result(s):</span></div><table><thead><th>id</th><th>username</th><th>name</th><th>surname</th></thead>" if self.html else ""
            for user in self.cursor.fetchall():
                content += f"<tr>" if self.html else ""
                for f in user:
                    content += f"<td>{'-' if f is None else f}</td>" if self.html else f"{'-' if f is None else f}\n"
                content += f"</tr>" if self.html else ""
            content += f"</table>" if self.html else ""
            runtime += self._slept
            content += f"query executed in {runtime}" if self.html else f"{runtime}"
        except Exception as ex:
            content += ex.output if isinstance(ex,
                                               subprocess.CalledProcessError) else traceback.format_exc() if self.html else "ERROR"
            code = http.client.INTERNAL_SERVER_ERROR

        html_response = (constants.HTML_PREFIX + content + constants.HTML_POSTFIX) if self.html else content

        terminal = False

        if code == http.client.INTERNAL_SERVER_ERROR:
            reward = -0.
        else:
            reward = 0.
        if "7en8aiDoh!" in content in content:
            reward += 1
            terminal = True
        if self.get_help_reward:
            reward += self._get_help_reward(query)[0]
        return html_response, reward, terminal, {}

    def _get_help_reward(self, query: str):
        optimal_query = "SELECT * FROM".split(" ") + list("users")
        query_ = fancy_split(query)
        reward = sum(q == oq for q, oq in zip(query_, optimal_query))
        return reward, reward == 4

    def reset(self):
        state, _, _, _ = self.step("1")
        return state

    def get_params(self, query):
        params = {}
        for match in re.finditer(r"((\A|[?&])(?P<parameter>[\w\[\]]+)=)([^&]+)", query):
            val = urllib.parse.unquote(','.join(re.findall(r"(?:\A|[?&])%s=([^&]+)" % match.group("parameter"), query)))
            params[match.group("parameter")] = val
        return params


gym.envs.register(
    id='MyEnv-v0',
    entry_point='gym.envs.classic_control:MyEnv',
    max_episode_steps=1000,
)
