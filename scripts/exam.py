import os
import copy
import yaml
import argparse

from tqdm import tqdm
from datasets import Dataset, load_from_disk

from lbt.base import Component
from lbt.test import test_single_student, aggregate_scores
from lbt.utils.log import getLogger


LOGGER = getLogger("exam")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file", type=str, help="Path to the config file")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--teaching-dataset-file",
        type=str,
        help="The items in this dataset file will be used as few-shot demonstrations.",
        default=None,
    )
    parser.add_argument("--exam-dataset-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.cfg_file, "r") as rf:
        cfg = yaml.safe_load(rf)

    teaching_plans = cfg.get("teaching_plans", "every")
    assert isinstance(teaching_plans, (list, tuple)) or teaching_plans in {
        "every",
        "no demo",
    }

    # Load datasets
    if teaching_plans != "no demo":
        # Initialize teaching and exam datasets
        assert args.teaching_dataset_file is not None, (
            'Only when `teaching_plans == "no demo"`, `--teaching-dataset-file` can be'
            " omitted."
        )
        teaching_dataset = load_from_disk(args.teaching_dataset_file)
        LOGGER.info(
            f"Loaded teaching dataset from {args.teaching_dataset_file}, fields:"
            f" {teaching_dataset.features}"
        )
        # the columns are: question, rationale, solution, answer
        if "rationale" not in teaching_dataset.features:
            LOGGER.info(
                f"Use the `solution` field in {args.teaching_dataset_file} as the"
                " demonstration."
            )
            teaching_dataset = teaching_dataset.rename_columns(
                {"solution": "rationale"}
            )
        else:
            LOGGER.info(
                f"Use the `rationale` field in {args.teaching_dataset_file} as the"
                " demonstration."
            )
        # the columns are: question, rationale, answer

    exam_dataset = load_from_disk(args.exam_dataset_file)
    LOGGER.info(
        f"Loaded exam dataset from {args.exam_dataset_file}, fields:"
        f" {exam_dataset.features}"
    )
    if "rationale" not in exam_dataset.features:
        LOGGER.info(
            f"Use the `solution` field in {args.exam_dataset_file} as the GT to measure"
            " scores."
        )
        exam_dataset = exam_dataset.rename_columns({"solution": "rationale"})
    else:
        LOGGER.info(
            f"Use the `rationale` field in {args.exam_dataset_file} as the GT to"
            " measure scores."
        )
    # the columns are: question, rationale, answer

    # Unify teaching plan as a list
    if teaching_plans == "every":
        # Take `num_rows` exams, each with one row from the teaching dataset as the demonstration
        teaching_plans = [[index] for index in range(teaching_dataset.num_rows)]
    elif teaching_plans == "no demo":
        # Take 1 exam, with no demonstrations from the teaching dataset
        teaching_plans = [[]]
    else:
        # Take `len(teaching_plans)` exams,
        # each item in list is a list of indexes, which are the teaching-dataset indexes
        # that will be used as the demonstrations in one exam
        assert (
            max([num for num in sum(teaching_plans, []) if isinstance(num, int)])
            < teaching_dataset.num_rows
        )  # do a check

    # Initialize exam_maker, exam_prompter, exam_scorer, student_models
    exam_maker = Component.init_from_cfg(
        cfg, "exam_maker", exam_bank_dataset=exam_dataset
    )
    exam_prompter = Component.init_from_cfg(cfg, "exam_prompter")
    exam_scorer = Component.init_from_cfg(cfg, "exam_scorer")
    student_pool = [
        Component.init_from_cfg(s_m_cfg, "model")
        for s_m_cfg in cfg["student_model_cfgs"]
    ]
    student_sample_cfgs = [
        s_m_cfg.get("sample_cfg", {}) for s_m_cfg in cfg["student_model_cfgs"]
    ]

    # Prepare output directory, dump the config
    os.makedirs(args.output_path, exist_ok=True)
    cfg["teaching_dataset_file"] = args.teaching_dataset_file
    cfg["exam_dataset_file"] = args.exam_dataset_file
    with open(os.path.join(args.output_path, "config.yaml"), "w") as wf:
        yaml.safe_dump(cfg, wf)

    # Loop: Iterate over the teaching plans
    # The output dataset has fields: teaching_items: List, exam_questions: List[str],
    # exam_gt_rationales List[str]: exam_gt_answers: List[str],
    # scores: Dict[str, float], exam_details: Dict[str, List]
    output_items = []
    for teaching_plan in tqdm(teaching_plans):
        teaching_item_question_only = False

        if teaching_plan:
            if teaching_plan[0] == "question-only":
                teaching_item_question_only = True
                teaching_plan = teaching_plan[1:]
            teaching_items = [teaching_dataset[index] for index in teaching_plan]
        else:
            teaching_items = []

        output_item = {
            "teaching_items": teaching_items,
            "exam_questions": [],
            "exam_gt_rationales": [],
            "exam_gt_answers": [],
            "exam_details": {student.name: None for student in student_pool},
            "scores": {student.name: None for student in student_pool},
        }

        # Decide the exam questions
        s_exam_dataset = exam_maker.make_exam_questions(teaching_items)
        # Record the exam questions and gt answers for this teaching question - rationale pair
        output_item["exam_questions"] = s_exam_dataset["question"]
        output_item["exam_gt_rationales"] = s_exam_dataset["rationale"]
        output_item["exam_gt_answers"] = s_exam_dataset["answer"]

        # Loop: Evaluate each student
        for student, stu_sample_cfg in zip(student_pool, student_sample_cfgs):
            # Loop: Evaluate each question
            sample_cfg = copy.deepcopy(
                cfg.get("general_student_sample_cfg", {})
            )  # general sample config
            sample_cfg.update(stu_sample_cfg)  # update with per-student sample config
            (
                single_student_rationales,
                single_student_answers,
                single_student_scores,
            ) = test_single_student(
                student=student,
                exam_prompter=exam_prompter,
                exam_scorer=exam_scorer,
                teaching_items=(
                    teaching_items if not teaching_item_question_only else []
                ),
                exam_dataset=s_exam_dataset,
                sample_cfg=sample_cfg,
            )

            # judges & exam_rationales: a nested list of shape `num_exam_questions x num_exam_answer_per_question`,
            # where every item is a score or a string
            output_item["exam_details"][student.name] = {
                "rationales": single_student_rationales,
                "answers": single_student_answers,
                "scores": single_student_scores,
            }
            score = aggregate_scores(single_student_scores)
            output_item["scores"][student.name] = score

        output_items.append(output_item)

    # Save the results
    output_dataset = Dataset.from_list(output_items)
    LOGGER.info(f"Dumping results to {args.output_path}   ...")
    output_dataset.save_to_disk(args.output_path)
    output_dataset.to_csv(os.path.join(args.output_path, "dataset.csv"))
    output_dataset.to_json(os.path.join(args.output_path, "dataset.json"), indent=2)
