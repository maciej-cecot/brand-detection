import json
import falcon
from wi_bd_api.lib.predict_crf import predict_only


class BrandDetection:
    """
    Example of BrandDetection body("title" and "root_cat" is required):
    body = {"title":["NEW MERRELL EAGLE ORIGINS PURPLE LEATHER LACE UP HIKING ANKLE BOOTS NEW",
                    "Kush 100% Hemp Rose pink King Size Slim Rolling Papers 5 & 10 pack deals"],
            "root_cat":[11450, 1]}
    """
    def on_post(self, req, resp):
        body = req.stream.read()
        if not body:
            raise falcon.HTTPBadRequest('Empty request body, JSON required')
        try:
            payload = json.loads(body.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            raise falcon.HTTPError(falcon.HTTP_753)

        brand = predict_only(titles = payload, model_filepath = '/wi-bd-api/wi_bd_api/lib/crf_model.sav')
        resp.body = json.dumps(brand)
        resp.status = falcon.HTTP_200
