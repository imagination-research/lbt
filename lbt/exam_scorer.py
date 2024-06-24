# -*- coding: utf-8 -*-
from abc import abstractmethod
from lbt.base import Component


class BaseExamScorer(Component):
    REGISTRY = "exam_scorer"

    @abstractmethod
    def score_exam_result(self, exam_gt_item, exam_result_item):
        """
        Return score.
        """
        pass


class ModelExamScorer(BaseExamScorer):
    NAME = "model_based"

    def __init__(self, model_type, model_cfg):
        super().__init__()

        self.model = Component.init_from_cfg(
            {"model_type": model_type, "model_cfg": model_cfg}, registry_name="model"
        )

    def score_exam_result(self, exam_gt_item, exam_result_item):
        # TODO: (1) make judge prompt (prompt template); (2) text_generator; (3) parse score from results (parsing)
        pass
