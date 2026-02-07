from celery import Celery
celery_app = Celery("deepfake_detector", broker="redis://redis:6379/2", backend="redis://redis:6379/3")
celery_app.conf.update(task_serializer="json", accept_content=["json"], result_serializer="json", timezone="UTC", enable_utc=True, task_track_started=True, task_time_limit=900, worker_prefetch_multiplier=1, worker_max_tasks_per_child=100)
celery_app.autodiscover_tasks(["app.workers"])
