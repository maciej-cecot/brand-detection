import falcon


class Ping:
    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.body = 'pong'
