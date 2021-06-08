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
"""
XXX We might need a new state as it has to remember some things from the previous queries, like the escape character (and maybe the number of rows, but it can also just go directly.)
"""

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


class SQLEnv(gym.Env):
    def render(self, mode='human'):
        pass

    def __init__(self, max_columns = 3):
        self.max_columns = max_columns
        #We can use the same database as long as we change the hidden query
        http.server.HTTPServer.allow_reuse_address = True
        self.connection = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)

        self.cursor = self.connection.cursor()
        self.cursor.execute("CREATE TABLE users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, firstname TEXT, surname TEXT, age INT, nationality TEXT, created_at TEXT)")
        self.cursor.execute("CREATE TABLE private(id INTEGER PRIMARY KEY AUTOINCREMENT, userid INT, account TEXT)")

        self.cursor.execute("INSERT INTO private(id, userid, account) VALUES(NULL, 1, 'accountnr:123456')")

        data = []
        #To tell the agent what kind of outputs it can expect (XXX so far this is not an exhaustive list)
        output_vocab = {"UNION", "SELECT", "*", "FROM", "users", "1", "ERROR", "", "private", "account", "None", "and"}

        for idx, row in enumerate(xml.etree.ElementTree.fromstring(constants.USERS_XML).findall("user")):
            row = row.findtext("username"), row.findtext("firstname"), row.findtext("surname"), row.findtext("age"), row.findtext("nationality"), row.findtext("created_at")
            data.append(row)
            output_vocab.update(row)
            output_vocab.update(str(idx + 1))

        self.observation_space = TextSpace(output_vocab)
        #Not truly the action space, but mainly to tell the agent what to expect?
        self.action_space = TextSpace(output_vocab)

        self.cursor.executemany("INSERT INTO users(id, username, firstname, surname, age, nationality, created_at) VALUES(NULL, ?, ?, ?, ?, ?, ?)", data)
        self.cursor.execute("CREATE TABLE comments(id INTEGER PRIMARY KEY AUTOINCREMENT, comment TEXT, time TEXT)")

        #We create the hidden_query in the reset function
        self.reset()


    def step(self, inquery: str):
        assert isinstance(inquery, str)
        code = http.client.OK
        content = ""
        query = self.hidden_query.format(input = inquery)
        try:
            self.cursor.execute(query)
            for some in self.cursor.fetchall():
                for f in some:
                    content += str(f) + ";"#"{'-' if f is None else f}\n"
        except Exception as ex:
            #print(ex)
            content += "ERROR"
            code = http.client.INTERNAL_SERVER_ERROR

        terminal = False

        if code == http.client.INTERNAL_SERVER_ERROR:
            reward = -1.
        else:
            reward = -.1
        if('account' in content):
            reward += 100
            terminal = True
        #return content, reward, terminal, {}
        return content, reward, terminal, {'episode': {'r': reward}}


    def reset(self):
        columns = constants.columns
        escape_characters = ["'", '"',""]

        #Picking the number of columns for the hidden query
        self.colnum = np.random.choice(self.max_columns)+1
        #Picking the escape_character for the hidden query
        self.escape = np.random.choice(3)
        #constructing the hidden query
        #Simulating that we are logged in as the dummy user Bob for this attack
        if(self.escape == 0):
            self.hidden_query = "SELECT "+", ".join(columns[:self.colnum])+" FROM users WHERE firstname={0}{1}{0}".format(escape_characters[self.escape], "{input}")
        elif(self.escape == 1):
            self.hidden_query = "SELECT "+", ".join(columns[:self.colnum])+" FROM users WHERE nationality={0}{1}{0}".format(escape_characters[self.escape], "{input}")
        else:
            self.hidden_query = "SELECT "+", ".join(columns[:self.colnum])+" FROM users WHERE age={0}{1}{0}".format(escape_characters[self.escape], "{input}")

        state, _, _, _ = self.step("1")
        return state



if(__name__ == "__main__"):
    s = SQLEnv()
    s.step("test")
