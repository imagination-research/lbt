# -*- coding: utf-8 -*-
import os.path as osp
from abc import abstractmethod
from typing import Union, Optional, List

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from fastchat.model import get_conversation_template
from fastchat.conversation import get_conv_template

from lbt.base import Component


### ---- exam makers ----
class BaseExamMaker(Component):
    REGISTRY = "exam_maker"

    def _get_num_exam_items(self, teaching_items, total_num_exam):
        num_t = len(teaching_items)
        if total_num_exam % num_t != 0:
            if num_t > total_num_exam:
                # Top-1 neighbor for the front teaching_items
                num_list = [1] * total_num_exam + [0] * (num_t - total_num_exam)
            else:
                # Averagely allocate neighbor num quota to teaching items,
                # the first teaching item get extra quota
                num_other = total_num_exam // num_t
                num_first = total_num_exam - num_other * (num_t - 1)
                num_list = [num_first] + [num_other] * (num_t - 1)
            self.logger.warn(
                f"Want to fetch {total_num_exam} exam questions that are"
                f" similar to {num_t} teaching items. {total_num_exam} %"
                f" {num_t} != 0."
            )
        else:
            num_list = [total_num_exam // num_t] * num_t
        return num_list

    @abstractmethod
    def make_exam_questions(self, teaching_items):
        # (1) choose from `exam_bank_dataset`
        # (2) use a strong `exam_proposal_model` to propose questions
        pass


class FixedExamMaker(BaseExamMaker):
    NAME = "fixed"

    def __init__(
        self,
        exam_bank_dataset: Union[str, Dataset],
        selected_indexes: Optional[Union[str, List]] = None,
    ):
        super().__init__()

        if isinstance(exam_bank_dataset, str):
            assert osp.exists(exam_bank_dataset)
            self.exam_bank_dataset_path = exam_bank_dataset
            self.exam_bank_dataset = load_from_disk(self.exam_bank_dataset_path)
        else:
            assert isinstance(exam_bank_dataset, Dataset)
            self.exam_bank_dataset = exam_bank_dataset

        if selected_indexes is None:
            self.exam_selected_dataset = self.exam_bank_dataset
        else:
            if isinstance(selected_indexes, str):
                selected_indexes = eval(selected_indexes)
            else:
                assert isinstance(selected_indexes, (tuple, list))
            self.exam_selected_dataset = self.exam_bank_dataset.select(selected_indexes)
            self.selected_indexes = selected_indexes

    def make_exam_questions(self, teaching_items):
        return self.exam_selected_dataset


class QuesSimilarityExamMaker(FixedExamMaker):
    """
    Ref: opencompass icl_topk_retriever
    """

    NAME = "ques_similarity"

    def __init__(
        self,
        exam_bank_dataset: Union[str, Dataset],
        selected_indexes: Optional[Union[str, List]] = None,
        sentence_transformers_model_name: Optional[str] = "all-mpnet-base-v2",
        knn_pickle_path: str = "knn16.pkl",
        num_exam_questions: int = 1,
        num_repetitions: int = 8,
    ):
        super().__init__(exam_bank_dataset, selected_indexes)

        self.num_exam_questions = num_exam_questions
        self.num_repetitions = num_repetitions

        if knn_pickle_path:
            import pickle

            with open(knn_pickle_path, "rb") as f:
                self.knn_pkl = pickle.load(f)
        else:
            from sentence_transformers import SentenceTransformer

            self.knn_pkl = None

            self.model = SentenceTransformer(sentence_transformers_model_name)
            self.model = self.model.to("cuda")
            self.model.eval()
            self.emb_dim = self.model.get_sentence_embedding_dimension()

            self.emb_index = self._create_index()

    def _create_index(self):
        import faiss

        avail_exam_questions = self.exam_selected_dataset["question"]
        all_embs = []
        for question in avail_exam_questions:
            emb = self._embed(question)
            all_embs.append(emb)
        self.all_embs = np.stack(all_embs).astype("float32")

        size = len(avail_exam_questions)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.emb_dim))
        id_list = np.array(list(range(size)))
        index.add_with_ids(self.all_embs, id_list)
        return index

    def _knn_search(self, question, knn_num):
        emb = self._embed(question)
        self.logger.info(
            f"Retrieving {knn_num}-NN exam question indexes for teaching question"
            f' """{question}""" ...'
        )
        emb = np.expand_dims(emb, axis=0).astype("float32")
        near_ids = self.emb_index.search(emb, knn_num)[1][0].tolist()
        return near_ids

    def _embed(self, question):
        with torch.no_grad():
            emb = self.model.encode(question, show_progress_bar=False)
        return emb

    def make_exam_questions(self, teaching_items):
        knn_num_list = self._get_num_exam_items(teaching_items, self.num_exam_questions)
        all_exam_question_ids = []
        for t_item, knn_num in zip(teaching_items, knn_num_list):
            if knn_num == 0:
                continue

            if self.knn_pkl:
                for idx, question in self.knn_pkl[t_item["question"]][:knn_num]:
                    assert self.exam_selected_dataset[idx]["question"] == question
                    all_exam_question_ids.extend([idx] * self.num_repetitions)
            else:
                exam_question_ids = self._knn_search(t_item["question"], knn_num)
                all_exam_question_ids.extend(exam_question_ids * self.num_repetitions)
        return self.exam_selected_dataset.select(all_exam_question_ids)


class FunctionalExamMaker(FixedExamMaker):
    NAME = "func"

    def __init__(
        self,
        exam_bank_dataset: Union[str, Dataset],
        selected_indexes: Optional[Union[str, List]] = None,
        num_exam_questions: int = 3,
        num_repetitions: int = 3,
    ):
        super().__init__(exam_bank_dataset, selected_indexes)

        self.num_exam_questions = num_exam_questions
        self.num_repetitions = num_repetitions

    def make_exam_questions(self, teaching_items):
        all_exam_question_ids = []
        for t_item in teaching_items:
            exam_question_ids = []
            for i, e_item in enumerate(self.exam_selected_dataset):
                if t_item["unique_id"] == e_item["unique_id"]:
                    exam_question_ids.append(i)
            all_exam_question_ids.extend(exam_question_ids * self.num_repetitions)

        assert (
            len(all_exam_question_ids) == self.num_exam_questions * self.num_repetitions
        )
        return self.exam_selected_dataset.select(all_exam_question_ids)


### ---- exam prompters ----
class ExamPrompter(Component):
    REGISTRY = "exam_prompter"
    NAME = "basic"

    PROMPT_ROLE_SWITCH_STR = "[ROLESWITCHING assistant:]"

    def __init__(
        self,
        demo_template,
        exam_template,
        debug_template=None,
        instruction="",
        use_multi_round_conv=False,
        stub_teaching_items=None,
    ):
        super().__init__()
        self.demo_template = demo_template
        self.exam_template = exam_template
        self.debug_template = debug_template
        self.instruction = instruction
        self.use_multi_round_conv = use_multi_round_conv
        self.stub_teaching_items = stub_teaching_items or []

        if self.use_multi_round_conv:
            assert self.PROMPT_ROLE_SWITCH_STR in self.demo_template, (
                "`use_multi_round_conv==True`: Using multiple conversation rounds to"
                " present the teaching demostrations. Must specify the conversation"
                " switching point in `demo_template`."
            )
        else:
            assert self.PROMPT_ROLE_SWITCH_STR not in self.demo_template

    def make_exam_prompt_fastchat(
        self, teaching_items, exam_item, conv_template_type, debug=False
    ) -> str:
        try:
            # exact match conversation name
            conv = get_conv_template(conv_template_type)
        except KeyError:
            # get through model adapter
            conv = get_conversation_template(conv_template_type)
            assert (
                conv.name != "one_shot"
            ), f"`{conv_template_type}` not supported in `fastchat`."
            # For base model, use conv_template type "raw"

        if not debug:
            _exam_item = exam_item.copy()
            demo_items = self.stub_teaching_items + teaching_items

        if not debug:
            _exam_item = exam_item.copy()
            demo_items = self.stub_teaching_items + teaching_items

            if self.use_multi_round_conv:
                demo_template_user, demo_template_assistant = self.demo_template.split(
                    self.PROMPT_ROLE_SWITCH_STR
                )
                for t_item in demo_items:
                    demo_user = demo_template_user.format(**t_item)
                    demo_assistant = demo_template_assistant.format(**t_item)
                    conv.append_message(conv.roles[0], demo_user)
                    conv.append_message(conv.roles[1], demo_assistant)
            else:
                demo = "\n\n\n".join(
                    [self.demo_template.format(**t_item) for t_item in (demo_items)]
                )
                _exam_item["demo"] = demo
            if self.use_multi_round_conv:
                demo_template_user, demo_template_assistant = self.demo_template.split(
                    self.PROMPT_ROLE_SWITCH_STR
                )
                for t_item in demo_items:
                    demo_user = demo_template_user.format(**t_item)
                    demo_assistant = demo_template_assistant.format(**t_item)
                    conv.append_message(conv.roles[0], demo_user)
                    conv.append_message(conv.roles[1], demo_assistant)
            else:
                demo = "\n\n\n".join(
                    [self.demo_template.format(**t_item) for t_item in (demo_items)]
                )
                _exam_item["demo"] = demo

        if self.PROMPT_ROLE_SWITCH_STR in self.exam_template:
            # has partial answer
            exam_template_user, exam_template_assistant = self.exam_template.split(
                self.PROMPT_ROLE_SWITCH_STR
            )
            exam_user = exam_template_user.format(**_exam_item)
            exam_assistant = exam_template_assistant.format(**_exam_item)
        else:
            exam_user = self.exam_template.format(**_exam_item)
            exam_assistant = None

        conv.append_message(conv.roles[0], self.instruction + exam_user)
        conv.append_message(conv.roles[1], None)
        return (conv, exam_assistant)

    def make_exam_prompt_chat_template(self, teaching_items, exam_item) -> str:
        conv = [{"role": "system", "content": "You are a helpful assistant."}]

        _exam_item = exam_item.copy()
        demo_items = self.stub_teaching_items + teaching_items

        if self.use_multi_round_conv:
            demo_template_user, demo_template_assistant = self.demo_template.split(
                self.PROMPT_ROLE_SWITCH_STR
            )
            for t_item in demo_items:
                demo_user = demo_template_user.format(**t_item)
                demo_assistant = demo_template_assistant.format(**t_item)
                conv.append({"role": "user", "content": demo_user})
                conv.append({"role": "assistant", "content": demo_assistant})
        else:
            demo = "\n\n\n".join(
                [self.demo_template.format(**t_item) for t_item in (demo_items)]
            )
            _exam_item["demo"] = demo

        if self.PROMPT_ROLE_SWITCH_STR in self.exam_template:
            # has partial answer
            exam_template_user, exam_template_assistant = self.exam_template.split(
                self.PROMPT_ROLE_SWITCH_STR
            )
            exam_user = exam_template_user.format(**_exam_item)
            exam_assistant = exam_template_assistant.format(**_exam_item)
        else:
            exam_user = self.exam_template.format(**_exam_item)
            exam_assistant = None

        conv.append({"role": "user", "content": self.instruction + exam_user})
        return (conv, exam_assistant)


if __name__ == "__main__":
    from pprint import pprint
    from termcolor import colored

    teaching_items = load_from_disk(
        "../NLP-playground/examples/rationale/data/math_solution_worstRationale_10"
    )
    name_mapping = {"problem": "question", "solution": "rationale"}
    teaching_items = teaching_items.select_columns(
        ["solution", "answer", "problem"]
    ).rename_columns(name_mapping)
    teaching_items = teaching_items.to_list()[:2]
    print(colored("Teaching items:\n----", "green"))
    pprint(teaching_items)

    exam_band_dataset = load_from_disk(
        "../NLP-playground/examples/rationale/data/math_1500"
    )
    exam_item = exam_band_dataset[0]
    exam_item = {new_n: exam_item[old_n] for old_n, new_n in name_mapping.items()}
    print(colored("Exam items:\n----", "green"))
    pprint(exam_item)

    # OpenCompass ICL template for Math
    single_conv_prompter = ExamPrompter(
        demo_template="""Question:\n{question}\n\nSolution:\n{rationale}\n\nFinal Answer:\nThe final answer is $${answer}$$.\n""",
        exam_template=(
            "{demo}\n\n\nQuestion:\n{question}\n\n[ROLESWITCHING assistant:]Solution:\n"
        ),
        use_multi_round_conv=False,
    )
    multi_conv_prompter = ExamPrompter(
        demo_template="""Question:\n{question}\n\n[ROLESWITCHING assistant:]Solution:\n{rationale}\n\nFinal Answer:\nThe final answer is $${answer}$$.\n""",
        exam_template="Question:\n{question}\n\n[ROLESWITCHING assistant:]Solution:\n",
        use_multi_round_conv=True,
    )

    for conv_template_type in ["llama-2", "qwen-7b-chat", "chatglm3"]:
        single_conv_prompt = single_conv_prompter.make_exam_prompt(
            teaching_items, exam_item, conv_template_type
        )
        print(
            colored(
                f"[{conv_template_type}] Single-round conversation prompt string\n----",
                "green",
            )
        )
        print(single_conv_prompt)

        multi_conv_prompt = multi_conv_prompter.make_exam_prompt(
            teaching_items, exam_item, conv_template_type
        )
        print(
            colored(
                f"[{conv_template_type}] Multi-round conversation prompt string\n----",
                "green",
            )
        )
        print(multi_conv_prompt)
