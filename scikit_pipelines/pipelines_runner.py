from sklearn.pipeline import Pipeline
from typing import List


class PipelinesRunner(object):
    __pipelines: List[Pipeline]

    def __init__(self):
        self.__pipelines = []

    def reset(self):
        self.__pipelines.clear()

    def add_pipeline(self, pipeline: Pipeline):
        self.__pipelines.append(pipeline)

    def call_method(self, method_name: str, *args, **kwargs):
        for pipeline in self.__pipelines:
            getattr(pipeline, method_name)(*args, **kwargs)


__all__ = ["PipelinesRunner"]
