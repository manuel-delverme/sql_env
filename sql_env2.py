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

    def __init__(self):
        #We can use the same database as long as we change the hidden query
        http.server.HTTPServer.allow_reuse_address = True
        self.connection = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)

        self.cursor = self.connection.cursor()
        self.cursor.execute("CREATE TABLE users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, name TEXT, surname TEXT, password TEXT, address TEXT)")
        self.cursor.execute("CREATE TABLE flagtable(id INTEGER PRIMARY KEY AUTOINCREMENT, flag TEXT)")

        self.cursor.execute("INSERT INTO flagtable(id, flag) VALUES(NULL, 'flag')")

        data = []

        for idx, row in enumerate(xml.etree.ElementTree.fromstring(constants.USERS_XML).findall("user")):
            row = row.findtext("username"), row.findtext("name"), row.findtext("surname"), row.findtext("password")
            data.append(row)

        self.cursor.executemany("INSERT INTO users(id, username, name, surname, password) VALUES(NULL, ?, ?, ?, ?)", data)
        self.cursor.execute("CREATE TABLE comments(id INTEGER PRIMARY KEY AUTOINCREMENT, comment TEXT, time TEXT)")

        #We create the hidden_query in the reset function
        self.reset()


    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self, inquery: str):
        assert isinstance(inquery, str)
        code = http.client.OK
        content = ""
        query = self.hidden_query.format(input = inquery)
        try:
            self.cursor.execute(query)
            for user in self.cursor.fetchall():
                for f in user:
                    content += f"{'-' if f is None else f}\n"
            content += f"{runtime}"
        except Exception as ex:
            content += "ERROR"
            code = http.client.INTERNAL_SERVER_ERROR

        terminal = False

        if code == http.client.INTERNAL_SERVER_ERROR:
            reward = -1.
        else:
            reward = -.1
        if('flag' in content):
            reward += 100
            terminal = True
        return content, reward, terminal, {}


    def reset(self):
        columns = ["id", "username", "name", "surname", "address"]
        escape_characters = ["'", '"',""]

        #Picking the number of columns for the hidden query
        self.colnum = np.random.choice(5)+1
        #Picking the escape_character for the hidden query
        self.escape = np.random.choice(3)
        #constructing the hidden query
        self.hidden_query = "SELECT "+", ".join(columns[:self.colnum])+" FROM users WHERE id={0}{1}{0}".format(escape_characters[self.escape], "{input}")

        state, _, _, _ = self.step("1")
        return state

    def get_params(self, query):
        params = {}
        for match in re.finditer(r"((\A|[?&])(?P<parameter>[\w\[\]]+)=)([^&]+)", query):
            val = urllib.parse.unquote(','.join(re.findall(r"(?:\A|[?&])%s=([^&]+)" % match.group("parameter"), query)))
            params[match.group("parameter")] = val
        return params


if(__name__ == "__main__"):
    s = SQLEnv()
    s.step("test")
    print(s.get_params("trsdd"))
