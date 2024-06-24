from typing import Dict, List, Any, Tuple
import numpy as np

from tqdm import tqdm
from datasets import Dataset

from lbt.qa_item import QAItem
from lbt.models.base import BaseModel
from lbt.exam_maker import ExamPrompter
from lbt.exam_scorer import BaseExamScorer


def aggregate_scores(scores: List[float]) -> float:
    scores = np.array(scores)
    return scores.mean()


def test_single_student(
    student: BaseModel,
    exam_prompter: ExamPrompter,
    exam_scorer: BaseExamScorer,
    teaching_items: List[QAItem],
    exam_dataset: Dataset,
    sample_cfg: Dict[str, Any],
    debug: bool = False,
) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    def exam_prompt_generator(debug=False, rationales=False, answers=False):
        for exam_item in exam_dataset:
            if debug:
                exam_item["rationale"] = rationales[i]
                exam_item["answer"] = answers[i]
            if student.fastchat:
                yield exam_prompter.make_exam_prompt_fastchat(
                    teaching_items, exam_item, student.conv_template_type
                )
            else:
                yield exam_prompter.make_exam_prompt_chat_template(
                    teaching_items, exam_item
                )

    single_student_exam_rationales = []
    single_student_exam_answers = []
    single_student_exam_scores = []

    for i, single_question_exam_rationales in enumerate(
        tqdm(
            student.text_generator(
                exam_prompt_generator(),
                return_full_text=False,
                **sample_cfg,
            ),
            total=len(exam_dataset),
        )
    ):
        exam_gt_item = exam_dataset[i]

        single_question_exam_rationales = [
            _["generated_text"] for _ in single_question_exam_rationales
        ]
        exam_result_items = [
            QAItem(question=exam_gt_item["question"], rationale=rationale, task_id=exam_gt_item.get("task_id", None))
            for rationale in single_question_exam_rationales
        ]

        scores = [
            exam_scorer.score_exam_result(exam_gt_item, exam_result_item)
            for exam_result_item in exam_result_items
        ]
        single_question_exam_answers = [
            exam_result_item["answer"] for exam_result_item in exam_result_items
        ]

        single_student_exam_rationales.append(single_question_exam_rationales)
        single_student_exam_answers.append(single_question_exam_answers)
        single_student_exam_scores.append(scores)
    
    # add a debug loop, check each question's answer
    if debug:
        single_student_exam_answers_debug = []
        single_student_exam_scores_debug = []
        for i, single_question_exam_rationales_debug in enumerate(
            tqdm(
                student.text_generator(
                    exam_prompt_generator(debug=debug, rationales=single_student_exam_rationales, answers=single_student_exam_answers),
                    return_full_text=False,
                    **sample_cfg,
                ),
                total=len(exam_dataset),
            )
        ):
            exam_gt_item = exam_dataset[i]

            # if there are no error, skip
            if '# ' not in single_student_exam_answers[i][0][:5]:
                single_student_exam_answers_debug.append(single_student_exam_answers[i])
                single_student_exam_scores_debug.append(single_student_exam_scores[i])
                continue

            single_question_exam_rationales_debug = [
                _["generated_text"] for _ in single_question_exam_rationales_debug
            ]
            exam_result_items = [
                QAItem(question=exam_gt_item["question"], rationale=rationale, task_id=exam_gt_item.get("task_id", None))
                for rationale in single_question_exam_rationales_debug
            ]

            scores = [
                exam_scorer.score_exam_result(exam_gt_item, exam_result_item)
                for exam_result_item in exam_result_items
            ]
            single_question_exam_answers = [
                exam_result_item["answer"] for exam_result_item in exam_result_items
            ]
            
            single_student_exam_answers_debug.append(single_question_exam_answers)
            single_student_exam_scores_debug.append(scores)
        # rename the debug information
        single_student_exam_answers = single_student_exam_answers_debug
        single_student_exam_scores = single_student_exam_scores_debug

    return (
        single_student_exam_rationales,
        single_student_exam_answers,
        single_student_exam_scores,
    )
