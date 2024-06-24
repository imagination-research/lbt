from datasets import load_from_disk

from lbt.base import Component
from lbt.exam_maker import QuesSimilarityExamMaker

teaching_dataset = (
    load_from_disk(
        "../NLP-playground/examples/rationale/data/math_solution_worstRationale_10/"
    )
    .select_columns(["problem", "answer", "solution"])
    .rename_columns({"problem": "question", "solution": "rationale"})
)

exam_dataset = load_from_disk("../NLP-playground/examples/rationale/data/math_1500/")
# the columns are: problem, solution, solution
exam_dataset = exam_dataset.select_columns(
    ["problem", "answer", "solution"]
).rename_columns({"problem": "question", "solution": "rationale"})

exam_maker = QuesSimilarityExamMaker(
    exam_bank_dataset=exam_dataset,
    selected_indexes="range(0, 16)",
    num_exam_questions=4,
)

s_exam_dataset_1t = exam_maker.make_exam_questions([teaching_dataset[0]])
print(s_exam_dataset_1t["question"])
s_exam_dataset_2t = exam_maker.make_exam_questions(teaching_dataset.to_list()[:2])
print(s_exam_dataset_2t["question"])
s_exam_dataset_3t = exam_maker.make_exam_questions(teaching_dataset.to_list()[3:6])
print(s_exam_dataset_3t["question"])
