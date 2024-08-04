import redis
import rq

import test

redis_conn = redis.Redis(host="redis", port=6379)
worker = rq.Worker(["segmentation_queue"], connection=redis_conn)
worker.work()