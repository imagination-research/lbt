# from opencompass.registry import ICL_EVALUATORS  # , TEXT_POSTPROCESSORS
import re
import random
import signal
import contextlib
import numpy as np

from datasets import concatenate_datasets
from datasets import Dataset

from lbt.exam_scorer import BaseExamScorer
from lbt.exam_maker import FixedExamMaker, ExamPrompter

from fastchat.model import get_conversation_template
from fastchat.conversation import get_conv_template


### Different ExamScorer


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class CodeExamScorer(BaseExamScorer):
    NAME = "code"

    def __init__(self, recall_mode=False):
        super().__init__()
        self.recall_mode = recall_mode

    def score_exam_result(self, exam_gt_item, exam_result_item):
        extracted_code = self.post_process(exam_result_item["rationale"])

        exam_result_item["answer"] = extracted_code

        # evaluate the code
        try:
            # run the code
            with time_limit(10):
                exec(extracted_code + "\n" + exam_gt_item["test"], {})
            return 1.0
        except TimeoutException:
            exam_result_item["answer"] = (
                "# Code execution took too long and was terminated.\n"
                + exam_result_item["answer"]
            )
            print("Code execution took too long and was terminated.")
            return 0.0
        except AssertionError as e:
            exam_result_item["answer"] = (
                f"# AssertionError: {e}\n" + exam_result_item["answer"]
            )
            print(f"AssertionError: {e}")
            return 0.0
        except BaseException as e:
            exam_result_item["answer"] = (
                f"# An error occurred: {e} \n" + exam_result_item["answer"]
            )
            print(f"An error occurred: {e}")
            return 0.0

    # Copy and modified from opencompass.datasets.math
    @staticmethod
    def post_process(text: str) -> str:
        if "[[Final Code]]:\n" in text:
            text = text.split("[[Final Code]]:\n")[-1]
        if "[[DEBUG Code]]:\n" in text:
            text = text.split("[[DEBUG Code]]:\n")[-1]
        if "$" in text:
            split_test = text.split("$")
            for sample_split in split_test[::-1]:
                if "class Solution" in sample_split and "\n    def" in sample_split:
                    text = sample_split
                    break
        if "```" in text:
            split_test = text.split("```")
            for sample_split in split_test[::-1]:
                if "class Solution" in sample_split and "\n    def" in sample_split:
                    text = sample_split
                    break
        if "'''" in text:
            split_test = text.split("'''")
            for sample_split in split_test[::-1]:
                if "class Solution" in sample_split and "\n    def" in sample_split:
                    text = sample_split
                    break
        if '"""' in text:
            split_test = text.split('"""')
            for sample_split in split_test[::-1]:
                if "class Solution" in sample_split and "\n    def" in sample_split:
                    text = sample_split
                    break

        # remove fault code
        if "from typing import str" in text:
            text = text.replace("from typing import str", "")
        if "from typing import int" in text:
            text = text.replace("from typing import int", "")
        if "from typing import bool" in text:
            text = text.replace("from typing import bool", "")

        # add necessary imports
        text = (
            "from typing import List\nfrom collections import Counter\nimport collections\nimport math\nimport bisect\n"
            + text
        )

        # deal with divide operation
        if "\/" in text:
            text = text.replace("\/", "/")

        # deal with known useless strings
        text = re.sub("python", "", text, flags=re.IGNORECASE)

        return text


from lbt.datasets_adapter.leetcode_sub.types import (
    LeetCodeSubmission,
    ProgrammingLanguage,
)
from lbt.datasets_adapter.leetcode_sub.environment import LeetCodeEnv


class CodeSubmitScorer(CodeExamScorer):
    NAME = "code_submit"

    def __init__(self, recall_mode=False):
        super().__init__()
        self.recall_mode = recall_mode

    def score_exam_result(self, exam_gt_item, exam_result_item):
        extracted_code = self.post_process(exam_result_item["rationale"])

        exam_result_item["answer"] = extracted_code

        # evaluate the code
        try:
            # run the code
            sub = LeetCodeSubmission(
                code=extracted_code,
                lang=ProgrammingLanguage.PYTHON3,
                question_slug=exam_result_item["task_id"],
                timeout=5,
            )
            env = LeetCodeEnv()
            status, reward, done, submission_result = env.step(sub)
            if done:
                return 1.0
            else:
                return 0.0
        except BaseException as e:
            exam_result_item["answer"] = (
                f"# An error occurred: {e} \n" + exam_result_item["answer"]
            )
            print(f"An error occurred: {e}")
            return 0.0


from lbt.datasets_adapter.apps_utils.testing_util import run_test


class CodeAppsExamScorer(CodeExamScorer):
    NAME = "code_apps"

    def __init__(self, recall_mode=False):
        super().__init__()
        self.recall_mode = recall_mode

    def score_exam_result(self, exam_gt_item, exam_result_item):
        extracted_code = self.post_process(exam_result_item["rationale"])

        exam_result_item["answer"] = extracted_code

        # evaluate the code
        try:
            with time_limit(0):
                results = run_test(extracted_code, exam_gt_item["test"])
                results = np.array(results).astype(np.float32)
                return np.average(results)
        except TimeoutException:
            exam_result_item["answer"] = (
                "# Code execution took too long and was terminated.\n"
                + exam_result_item["answer"]
            )
            print("Code execution took too long and was terminated.")
            return 0.0
        except BaseException as e:
            exam_result_item["answer"] = (
                f"# An error occurred: {e} \n" + exam_result_item["answer"]
            )
            print(f"An error occurred: {e}")
            return 0.0

    # Copy and modified from opencompass.datasets.math
    @staticmethod
    def post_process(text: str) -> str:
        if "[[Final Code]]:\n" in text:
            text = text.split("[[Final Code]]:\n")[-1]
        if "[[DEBUG Code]]:\n" in text:
            text = text.split("[[DEBUG Code]]:\n")[-1]
        if "$" in text:
            split_test = text.split("$")
            for sample_split in split_test[::-1]:
                if "input()" in sample_split:
                    text = sample_split
                    break
        if "```" in text:
            split_test = text.split("```")
            for sample_split in split_test[::-1]:
                if "input()" in sample_split:
                    text = sample_split
                    break
        if "'''" in text:
            split_test = text.split("'''")
            for sample_split in split_test[::-1]:
                if "input()" in sample_split:
                    text = sample_split
                    break
        if '"""' in text:
            split_test = text.split('"""')
            for sample_split in split_test[::-1]:
                if "input()" in sample_split:
                    text = sample_split
                    break

        # remove fault code
        if "from typing import str" in text:
            text = text.replace("from typing import str", "")
        if "from typing import int" in text:
            text = text.replace("from typing import int", "")
        if "from typing import bool" in text:
            text = text.replace("from typing import bool", "")

        # deal with divide operation
        if "\/" in text:
            text = text.replace("\/", "/")

        # deal with known useless strings
        text = re.sub("python", "", text, flags=re.IGNORECASE)

        return text


class CodeAppsExamScorer(CodeExamScorer):
    NAME = "code_choose"

    def __init__(self, recall_mode=False):
        super().__init__()
        self.recall_mode = recall_mode

    def score_exam_result(self, exam_gt_item, exam_result_item):
        response_number = self.post_process(exam_result_item["rationale"])

        exam_result_item["answer"] = response_number
        return 0.0

    # Copy and modified from opencompass.datasets.math
    @staticmethod
    def post_process(text: str) -> str:
        # find the last digital number in a string
        response_number = re.findall("\d+", text)
        if response_number is not None and len(response_number) > 0:
            response_number = int(response_number[-1])
        else:
            print(f"Got unparsable result")
            response_number = -1

        return response_number


### Different ExamMaker


class CodeMetaInfoMaker(FixedExamMaker):
    NAME = "code_metainfo"

    def __init__(
        self,
        exam_bank_dataset,
        selected_indexes=None,
        same_subject=True,
        level_controls=[
            "="
        ],  # Number: the corresponding level; =: the same level as the teaching question; >: higher/harder levels ...; <: lower/easier levels ...
        num_exam_questions=16,
        random=False,  # False: choose the first `num_exam_questions` that satisfy the meta-info control; True: random choose
    ):
        super().__init__(exam_bank_dataset, selected_indexes)

        self.same_subject = same_subject
        self.level_controls = level_controls
        self.num_exam_questions = num_exam_questions
        self.random = random
        if self.same_subject:
            assert "tags" in self.exam_selected_dataset.features

    def make_exam_questions(self, teaching_items):
        num_list = self._get_num_exam_items(teaching_items, self.num_exam_questions)
        final_selected_dataset = None
        for t_item, num_exam in zip(teaching_items, num_list):
            if num_exam == 0:
                continue

            # Filter according to item["tags"]
            if self.same_subject:
                if "tags" not in t_item:
                    self.logger.warn(
                        "The `level` feature of the teaching item is not set. Level"
                        # f" control `{control}` not supported."
                    )
                    continue
                exam_selected_dataset = self.exam_selected_dataset.filter(
                    lambda exam_item: t_item["tags"] in exam_item["tags"]
                )

            if exam_selected_dataset.num_rows > num_exam:
                # Select `num_exam` exam items from the filtered dataset
                if self.random:
                    # Random `num_exam` items
                    all_indexes = list(range(exam_selected_dataset.num_rows))
                    indexes = random.sample(population=all_indexes, k=num_exam)
                else:
                    # First `num_exam` items
                    indexes = list(range(num_exam))
                exam_selected_dataset = exam_selected_dataset.select(indexes)
            elif exam_selected_dataset.num_rows < num_exam:
                # Only warning
                # FIXME: rewrite this function to ensure returning `num_exam` exam items
                self.logger.warn(
                    "The size of the returned exam dataset would be smaller than the"
                    f" set `num_exam_questions`: {self.num_exam_questions}"
                )

            if final_selected_dataset is None:
                final_selected_dataset = exam_selected_dataset
            else:
                final_selected_dataset = concatenate_datasets(
                    [final_selected_dataset, exam_selected_dataset]
                )

        return final_selected_dataset


class CodeFixedMetaInfoMaker(FixedExamMaker):
    NAME = "code_fixed_exam"

    def __init__(
        self,
        exam_bank_dataset,
        selected_indexes=None,
        same_subject=True,
        level_controls=[
            "="
        ],  # Number: the corresponding level; =: the same level as the teaching question; >: higher/harder levels ...; <: lower/easier levels ...
        num_exam_questions=16,
        random=False,  # False: choose the first `num_exam_questions` that satisfy the meta-info control; True: random choose
    ):
        super().__init__(exam_bank_dataset, selected_indexes)

        self.same_subject = same_subject
        self.level_controls = level_controls
        self.num_exam_questions = num_exam_questions
        self.random = random
        if self.same_subject:
            assert "tags" in self.exam_selected_dataset.features

    def make_exam_questions(self, teaching_items):
        teacher_task_id = teaching_items[0]["task_id"]
        final_selected_dataset = None

        for exam in self.exam_selected_dataset:
            if exam["task_id"] == teacher_task_id:
                final_selected_dataset = [exam]
                break

        final_selected_dataset = Dataset.from_list(final_selected_dataset)

        return final_selected_dataset


### Different Prompter


class CodeDebugExamPrompter(ExamPrompter):
    NAME = "code_debug"

    def __init__(
        self,
        demo_template,
        exam_template,
        debug_template=None,
        instruction="",
        use_multi_round_conv=False,
        stub_teaching_items=None,
    ):
        super().__init__(
            demo_template,
            exam_template,
            debug_template,
            instruction,
            use_multi_round_conv,
            stub_teaching_items,
        )

    def make_exam_prompt(
        self, teaching_items, exam_item, conv_template_type, debug=False
    ) -> str:
        try:
            # exact match conversation name
            conv = get_conv_template(conv_template_type)
        except KeyError:
            # get through model adapter
            conv = get_conversation_template(conv_template_type)
            # For base model, use conv_template type "raw"

        assert self.use_multi_round_conv is True
        demo_items = self.stub_teaching_items + teaching_items

        exam_template_user, exam_template_assistant = self.exam_template.split(
            self.PROMPT_ROLE_SWITCH_STR
        )

        exam_user = exam_template_user.format(
            question=demo_items[0]["question"],
            rationale=demo_items[0]["rationale"],
            answer=demo_items[0]["answer"],
        )
        exam_assistant = exam_template_assistant

        conv.append_message(conv.roles[0], self.instruction + exam_user)
        conv.append_message(conv.roles[1], None)
        return (conv, exam_assistant)


class CodePipeline4_ExamPrompter(ExamPrompter):
    NAME = "code_pipeline4"

    def __init__(
        self,
        demo_template,
        exam_template,
        debug_template=None,
        instruction="",
        use_multi_round_conv=False,
        stub_teaching_items=None,
    ):
        super().__init__(
            demo_template,
            exam_template,
            debug_template,
            instruction,
            use_multi_round_conv,
            stub_teaching_items,
        )

    def make_exam_prompt(
        self, teaching_items, exam_item, conv_template_type, debug=False
    ) -> str:
        try:
            # exact match conversation name
            conv = get_conv_template(conv_template_type)
        except KeyError:
            # get through model adapter
            conv = get_conversation_template(conv_template_type)
            # For base model, use conv_template type "raw"

        assert self.use_multi_round_conv is True
        demo_items = self.stub_teaching_items + teaching_items

        answers = "\n\n"
        for i, demo_item in enumerate(demo_items):
            answers += (
                f"Answer Code {i} is :"
                + "\n```\n"
                + demo_item["answer"]
                + "\n```\n"
                + "\n\n"
            )

        exam_template_user, exam_template_assistant = self.exam_template.split(
            self.PROMPT_ROLE_SWITCH_STR
        )

        exam_user = exam_template_user.format(
            question=demo_items[0]["question"], answer=answers
        )
        exam_assistant = exam_template_assistant

        conv.append_message(conv.roles[0], self.instruction + exam_user)
        conv.append_message(conv.roles[1], None)
        return (conv, exam_assistant)


if __name__ == "__main__":
    from pprint import pprint
    import json

    # load the coding dataset in a list of dict
    t_data = []
    with open("examples/leetcode/leetcode-3-algorithms-python3.jsonl", "r") as file:
        for line in file:
            t_data.append(json.loads(line))

    # transform the coding dataset list into a huggingface dataset
    t_dataset = Dataset.from_list(t_data)

    # create the InfoMaker
    top16_maker = CodeMetaInfoMaker(
        "examples/rationale/data/math_12k/",
        level_controls=["=", 5],
        num_exam_questions=16,
        random=False,
    )
    random16_maker = CodeMetaInfoMaker(
        "examples/rationale/data/math_12k/",
        level_controls=["=", 5],
        num_exam_questions=16,
        random=True,
    )
    for t_set in [t_dataset[:1], t_dataset[:2], t_dataset[:3]]:
        print(f"num teaching items: {len(t_set)}")
        pprint(t_set)
        e_set_top16 = top16_maker.make_exam_questions(t_set)
        e_set_random16 = random16_maker.make_exam_questions(t_set)
        print("top16:")
        pprint(e_set_top16.select_columns(["level", "subject", "unique_id"]).to_list())
        print("random16:")
        pprint(
            e_set_random16.select_columns(["level", "subject", "unique_id"]).to_list()
        )
