from datasets import load_from_disk

from lbt.base import Component
from lbt.exam_maker import ExamPrompter, FixedExamMaker
from lbt.datasets_adapter.math_dataset import MathExamScorer

# output_dataset = load_from_disk("results/try_filter/").select_columns(
#     ["question", "answer", "solution"]
# ).rename_columns({"solution": "rationale"})
output_dataset = (
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
# exam_maker = Component.init_from_cfg(cfg, "exam_maker", exam_bank_dataset=exam_dataset)
exam_maker = FixedExamMaker(
    exam_bank_dataset=exam_dataset, selected_indexes="range(0, 16)"
)
exam_dataset = exam_maker.make_exam_questions(None)

exam_prompter = ExamPrompter(
    demo_template="""Question:\n{question}\n\n[ROLESWITCHING assistant:]Solution:\n{rationale}\n\nFinal Answer:\n${answer}$\n""",
    exam_template="Question:\n{question}\n\n[ROLESWITCHING assistant:]Solution:\n",
    use_multi_round_conv=True,
)

stub_teacher_items = [
    {
        "question": "What is 10+8-4?",
        "rationale": "10+8=18, 18-4=14. So the answer value is 14.",
        "answer": "14",
    },
    {
        "question": "What is the result of $\frac{6 \times 3}{2} ?",
        "rationale": (
            "$6 \times 3 = 18$, $18/2=9$. $9$ should be the result of $\frac{6 \times"
            " 3}{2}."
        ),
        "answer": "9",
    },
]
teach_index = 0
teaching_items = stub_teacher_items + [output_dataset[teach_index]]
exam_index = 3
exam_item = exam_dataset[exam_index]
conv_template_type = "Qwen/Qwen-14B-Chat"
prompt = exam_prompter.make_exam_prompt(teaching_items, exam_item, conv_template_type)

exam_scorer = MathExamScorer()
