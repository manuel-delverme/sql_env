import http.client
import http.server
import random
import re
import sqlite3
import subprocess
import traceback
import urllib.parse
import urllib.request
import xml.etree.ElementTree

import gym.envs
import torch
import torch.distributions
import torch.nn as nn
import tqdm

import constants

torch.manual_seed(1)

EMBEDDING_DIM = 10
QUERY_LEN = 4
OBSERVATION_LEN = 20


class SQLEnv(gym.Env):
    def render(self, mode='human'):
        pass

    def __init__(self, html):

        self.html = html
        http.server.HTTPServer.allow_reuse_address = True
        self.connection = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
        self.cursor = self.connection.cursor()
        self.cursor.execute("CREATE TABLE users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, name TEXT, surname TEXT, password TEXT)")

        data = []
        query_vocab = {"UNION", "SELECT", "*", "FROM", "users", "1", "ERROR", ""}
        output_vocab = {"UNION", "SELECT", "*", "FROM", "users", "1", "ERROR", ""}

        for idx, row in enumerate(xml.etree.ElementTree.fromstring(constants.USERS_XML).findall("user")):
            row = row.findtext("username"), row.findtext("name"), row.findtext("surname"), row.findtext("password")
            data.append(row)
            output_vocab.update(row)
            output_vocab.update(str(idx + 1))

        self.cursor.executemany("INSERT INTO users(id, username, name, surname, password) VALUES(NULL, ?, ?, ?, ?)", data)
        self.cursor.execute("CREATE TABLE comments(id INTEGER PRIMARY KEY AUTOINCREMENT, comment TEXT, time TEXT)")

        query_vocab = sorted(query_vocab)
        output_vocab = sorted(output_vocab)
        self.query_word_to_idx = {word: idx for idx, word in enumerate(query_vocab)}
        self.output_word_to_idx = {word: idx for idx, word in enumerate(output_vocab)}

        self.observation_space = gym.Space((OBSERVATION_LEN, EMBEDDING_DIM), )
        self.action_space = gym.spaces.MultiDiscrete(QUERY_LEN * [len(query_vocab), ])

        self.embeddings = nn.Embedding(len(output_vocab), EMBEDDING_DIM)
        self.embeddings.weight.requires_grad = False
        self.query_vocab = query_vocab
        self.output_vocab = output_vocab

    def _decode(self, action):
        return " ".join([self.query_vocab[w] for w in action])

    def _encode(self, state):
        words = state.split()
        assert len(words) <= self.observation_space.shape[0]

        words = words + [""] * self.observation_space.shape[0]
        words = words[:self.observation_space.shape[0]]
        try:
            retr = torch.tensor([self.output_word_to_idx[w] for w in words], dtype=torch.long)
        except Exception as e:
            print(state)
            raise e
        return retr

    def get_obs(self, text):
        word_idxes = self._encode(text)
        embeds = self.embeddings(word_idxes)
        assert embeds.shape == self.observation_space.shape
        return embeds

    def step(self, action):
        assert self.action_space.shape == action.shape

        query = self._decode(action)

        code = http.client.OK
        content = ""

        try:
            # self.cursor.execute("SELECT id, username, name, surname FROM users WHERE id=" + query)
            self.cursor.execute(query)
            content += "<div><span>Result(s):</span></div><table><thead><th>id</th><th>username</th><th>name</th><th>surname</th></thead>" if self.html else ""
            for user in self.cursor.fetchall():
                content += f"<tr>" if self.html else ""
                for f in user:
                    content += f"<td>{'-' if f is None else f}</td>" if self.html else f"{'-' if f is None else f}\n"
                content += f"</tr>" if self.html else ""
            content += f"</table>" if self.html else ""
        except Exception as ex:
            content += ex.output if isinstance(ex, subprocess.CalledProcessError) else traceback.format_exc() if self.html else "ERROR"
            code = http.client.INTERNAL_SERVER_ERROR

        html_response = (constants.HTML_PREFIX + content + constants.HTML_POSTFIX) if self.html else content

        terminal = False
        if code == http.client.INTERNAL_SERVER_ERROR:
            reward = -0.
        else:
            reward = 0.
        if "7en8aiDoh!" in content:
            reward += 1
            terminal = True

        obs = self.get_obs(html_response)
        assert obs.shape == self.observation_space.shape
        return obs, reward, terminal, {}

    def reset(self):
        query = ["1", ] + (QUERY_LEN - 1) * [""]
        action = [self.query_word_to_idx[w] for w in query]
        action = torch.tensor(action, dtype=torch.long)
        return self.step(action)[0]

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

if __name__ == "__main__":
    env = SQLEnv()
    env.reset()
    action = ""
    for _ in tqdm.trange(500_000):
        s, r, t, i = env.step(str(random.random()))
