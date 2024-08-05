import redis
import rq
import logging

logger = logging.getLogger(__name__)

class TaskQueue:
    def __init__(self, redis_host, redis_port, queue_name):
        self.redis_conn = redis.Redis(host=redis_host, port=redis_port)
        self.task_queue = rq.Queue(queue_name, connection=self.redis_conn)
        logger.info("Init task queue.")

    def format_job_status(self, job):
        ret = {}
        job_status = job.get_status(refresh=True)
        if job_status == 'queued':
            ret['done'] = False
            ret['description'] = f"Position in queue: {job.get_position()+1}"
            return ret, 200
        elif job_status == 'started':
            ret['done'] = False
            ret['description'] = "Now processing"
            return ret, 200
        elif job_status == 'finished':
            ret['done'] = True
            ret['result'] = job.result
            return ret, 200
        ret['done'] = False
        ret['description'] = "Something went wrong, try again later"
        logger.error("Task failed.")
        return ret, 500

    def add_task(self, function_name, *args):
        logger.info(f"Added task with args: {' '.join(map(str, args))}.")
        created_job = self.task_queue.enqueue(function_name, *args, result_ttl=60, failure_ttl=0)
        return created_job.id

    def get_job_status(self, job_id):
        job = rq.job.Job.fetch(job_id, connection=self.redis_conn)
        return self.format_job_status(job)
