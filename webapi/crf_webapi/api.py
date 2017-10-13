from webapi.resources.ping import Ping
from webapi.resources.brand_detection import BrandDetection
import falcon

#creating endpoints
application = falcon.API()
application.add_route('/alert/ping', Ping())
application.add_route('/brands', BrandDetection())
