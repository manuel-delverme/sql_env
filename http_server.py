import http.server

import environment  # noqa

env = environment.sql_env.SQLEnv()


class WebServerHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)

    def do_GET(self):
        query = self.path.split("/?id=", 1)[1]
        obs, reward, done, infos = env.step(query)

        if done:
            env.reset()

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        output = obs
        self.wfile.write(output._text_to_token_idx())
        # self.send_error(404, 'File Not Found: %s' % self.path)


def main():
    server_address = ('', 8000)
    httpd = http.server.HTTPServer(server_address, WebServerHandler)
    httpd.serve_forever()


if __name__ == "__main__":
    print('run python ~/sqlmap-dev/sqlmap.py --technique=U -u "localhost:8000/?id=123" --dump')
    main()
