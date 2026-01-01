import init
init.init()

import redis
import rq

import inference

redis_conn = redis.Redis(host="segmentation_redis", port=6379)
worker = rq.Worker(["segmentation_queue"], connection=redis_conn)
worker.work()
