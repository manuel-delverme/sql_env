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

import tqdm

import constants


class SQLEnv:
    def __init__(self):
        http.server.HTTPServer.allow_reuse_address = True
        self.connection = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
        self.cursor = self.connection.cursor()
        self.cursor.execute("CREATE TABLE users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, name TEXT, surname TEXT, password TEXT)")
        self.cursor.executemany("INSERT INTO users(id, username, name, surname, password) VALUES(NULL, ?, ?, ?, ?)",
                                ((_.findtext("username"), _.findtext("name"), _.findtext("surname"), _.findtext("password")) for _ in
                                 xml.etree.ElementTree.fromstring(constants.USERS_XML).findall("user")))
        self.cursor.execute("CREATE TABLE comments(id INTEGER PRIMARY KEY AUTOINCREMENT, comment TEXT, time TEXT)")

    def step(self, query):
        code = http.client.OK
        content = ""

        try:
            self.cursor.execute("SELECT id, username, name, surname FROM users WHERE id=" + query)
            content += "<div><span>Result(s):</span></div><table><thead><th>id</th><th>username</th><th>name</th><th>surname</th></thead>"
            for user in self.cursor.fetchall():
                content += f"<tr>"
                for f in user:
                    content += f"<td>{'-' if f is None else f}</td>"
                content += f"</tr>"
            content += f"</table>"
        except Exception as ex:
            content += ex.output if isinstance(ex, subprocess.CalledProcessError) else traceback.format_exc()
            code = http.client.INTERNAL_SERVER_ERROR

        html_response = constants.HTML_PREFIX + content + constants.HTML_POSTFIX

        terminal = False
        if code == http.client.INTERNAL_SERVER_ERROR:
            reward = -0.1
        else:
            reward = 0
        if "7en8aiDoh!" in content:
            reward += 10
            terminal = True

        return html_response, reward, terminal, {}

    def reset(self):
        pass

    def get_params(self, query):
        params = {}
        for match in re.finditer(r"((\A|[?&])(?P<parameter>[\w\[\]]+)=)([^&]+)", query):
            val = urllib.parse.unquote(','.join(re.findall(r"(?:\A|[?&])%s=([^&]+)" % match.group("parameter"), query)))
            params[match.group("parameter")] = val
        return params


if __name__ == "__main__":
    env = SQLEnv()
    env.reset()
    action = ""
    for _ in tqdm.trange(500_000):
        s, r, t, i = env.step(str(random.random()))
