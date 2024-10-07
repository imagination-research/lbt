# from opencompass.registry import ICL_EVALUATORS  # , TEXT_POSTPROCESSORS
import re
import random

from datasets import concatenate_datasets

from lbt.exam_scorer import BaseExamScorer
from lbt.exam_maker import FixedExamMaker


class MATHEvaluator:
    """
    Copied from opencompass, as directly using opencompass interfere with logging.
    """

    def _fix_fracs(self, string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(self, string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except AssertionError:
            return string

    def _remove_right_units(self, string):
        # "\\text{ " only ever occurs (at least in the val set) when describing
        # units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(self, string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    def _strip_string(self, string):
        # linebreaks
        string = string.replace("\n", "")

        # remove inverse spaces
        string = string.replace("\\!", "")

        # replace \\ with \
        string = string.replace("\\\\", "\\")

        # replace tfrac and dfrac with frac
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")

        # remove \( and \)
        string = string.replace("\\(", "")
        string = string.replace("\\)", "")

        # remove \left and \right
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")

        # Remove circ (degrees)
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")

        # remove dollar signs
        string = string.replace("\\$", "")

        # remove units (on the right)
        string = self._remove_right_units(string)

        # remove percentage
        string = string.replace("\\%", "")
        string = string.replace("\%", "")  # noqa: W605

        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively,
        # add "0" if "." is the start of the string
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]

        # fix sqrt3 --> sqrt{3}
        string = self._fix_sqrt(string)

        # remove spaces
        string = string.replace(" ", "")

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works
        # with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
        string = self._fix_fracs(string)

        # manually change 0.5 --> \frac{1}{2}
        if string == "0.5":
            string = "\\frac{1}{2}"

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix
        # in case the model output is X/Y
        string = self._fix_a_slash_b(string)

        return string

    def is_equiv(self, str1, str2, verbose=False):
        if str1 is None and str2 is None:
            print("WARNING: Both None")
            return True
        if str1 is None or str2 is None:
            return False

        try:
            ss1 = self._strip_string(str1)
            ss2 = self._strip_string(str2)
            if verbose:
                print(ss1, ss2)
            return ss1 == ss2
        except:  # noqa
            return str1 == str2

    def can_recall(self, extracted_answer, gt_answer):
        str1 = extracted_answer
        str2 = gt_answer
        try:
            ss1 = self._strip_string(str1)
            ss2 = self._strip_string(str2)
            return ss2 in ss1
        except:
            return str2 in str1


class MathExamScorer(BaseExamScorer):
    NAME = "math"

    def __init__(self, recall_mode=False):
        super().__init__()
        self.evaluator = MATHEvaluator()
        # ICL_EVALUATORS.build(
        #     {"type": "opencompass.datasets.MATHEvaluator"}
        # )
        self.recall_mode = recall_mode

    def score_exam_result(self, exam_gt_item, exam_result_item):
        gt_answer = self.post_process(exam_gt_item["answer"])
        extracted_answer = self.post_process(exam_result_item["rationale"])
        exam_result_item["answer"] = extracted_answer
        if self.recall_mode:
            is_correct = self.evaluator.can_recall(extracted_answer, gt_answer)
        else:
            is_correct = self.evaluator.is_equiv(extracted_answer, gt_answer)
        return float(is_correct)

    @staticmethod
    def _normalize_final_answer(final_answer: str) -> str:
        """Normalize a final answer to a quantitative reasoning question."""
        RE_SUBSTITUTIONS = [
            (r"\\le(?!ft)", r"<"),  # replace \le as <, but do not change "\left"
            (r"(?<!le)(ft)", ""),  # remove ft unit, but do not remove "\left"
        ]
        SUBSTITUTIONS = [
            ("an ", ""),
            ("a ", ""),
            (".$", "$"),
            ("\\$", ""),
            (r"\ ", ""),
            (" ", ""),
            ("mbox", "text"),
            (",\\text{and}", ","),
            ("\\text{and}", ","),
            ("\\text{m}", "\\text{}"),
        ]
        REMOVED_EXPRESSIONS = [
            "square",
            "ways",
            "integers",
            "dollars",
            "mph",
            "inches",
            "hours",
            "km",
            "units",
            "\\ldots",
            "sue",
            "points",
            "feet",
            "minutes",
            "digits",
            "cents",
            "degrees",
            "cm",
            "gm",
            "pounds",
            "meters",
            "meals",
            "edges",
            "students",
            "childrentickets",
            "multiples",
            "\\text{s}",
            "\\text{.}",
            "\\text{\ns}",
            "\\text{}^2",
            "\\text{}^3",
            "\\text{\n}",
            "\\text{}",
            r"\mathrm{th}",
            r"^\circ",
            r"^{\circ}",
            r"\;",
            r",\!",
            "{,}",
            '"',
            "\\dots",
            "\n",
            "\r",
            "\f",
        ]
        # final_answer = final_answer.split('=')[-1]
        for before, after in RE_SUBSTITUTIONS:
            final_answer = re.sub(before, after, final_answer)
        for before, after in SUBSTITUTIONS:
            final_answer = final_answer.replace(before, after)
        for expr in REMOVED_EXPRESSIONS:
            final_answer = final_answer.replace(expr, "")

        # Extract answer that is in LaTeX math, is bold,
        # is surrounded by a box, etc.
        final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
        assert "\n" not in final_answer
        assert "\r" not in final_answer
        assert "\f" not in final_answer

        # if len(re.findall(r"finalansweris(.*)", final_answer)) > 0:
        #     final_answer = re.findall(r"finalansweris(.*)", final_answer)[-1]

        # if len(re.findall(r"oxed\{(.*?)\}", final_answer)) > 0:
        #     final_answer = re.findall(r"oxed\{(.*?)\}", final_answer)[-1]

        # if len(re.findall(r"\$\$(.*?)\$\$", final_answer)) > 0:
        #     final_answer = re.findall(r"\$(.*?)\$", final_answer)[-1]
        # final_answer = final_answer.strip()
        # if "rac" in final_answer and "\\frac" not in final_answer:
        #     final_answer = final_answer.replace("rac", "\\frac")

        # Normalize shorthand TeX:
        # \fracab -> \frac{a}{b}
        # \frac{abc}{bef} -> \frac{abc}{bef}
        # \fracabc -> \frac{a}{b}c
        final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)

        final_answer = re.sub(
            r"(?<!\\)(sqrt)", r"\\sqrt", final_answer
        )  # change sqrt to \sqrt
        final_answer = re.sub(
            r"(sqrt)\(([^)]+)\)", "sqrt{\\2}", final_answer
        )  # change sqrt(...) to sqrt{...}
        # \sqrta -> \sqrt{a}
        # \sqrtab -> sqrt{a}b
        final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)

        final_answer = final_answer.replace("$", "")

        # Normalize 100,000 -> 100000
        if final_answer.replace(",", "").isdigit():
            final_answer = final_answer.replace(",", "")

        return final_answer

    # Copy and modified from opencompass.datasets.math
    @staticmethod
    def post_process(text: str) -> str:
        matches = re.findall(r"\[\[Final Answer\]\]:\n([^\n]+)\n", text)
        if not matches:
            answer = text.strip().split("\n")[-1]
        else:
            answer = matches[0]
        return MathExamScorer._normalize_final_answer(answer)
        # for maybe_ans in text.split("."):
        #     if "final answer" in maybe_ans.lower():
        #         return normalize_final_answer(maybe_ans)
        # return normalize_final_answer(text.split(".")[0])


class MathMetaInfoMaker(FixedExamMaker):
    NAME = "math_metainfo"

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
        if self.level_controls:
            assert "level" in self.exam_selected_dataset.features
        if self.same_subject:
            assert "subject" in self.exam_selected_dataset.features

    @staticmethod
    def _is_int(num):
        try:
            int(num)
        except ValueError:
            return False
        return True

    def _parse_permit_levels(self, teaching_level):
        permit_levels = []
        for control in self.level_controls:
            if self._is_int(control):
                permit_levels.append(int(control))
            else:
                assert control in ["=", ">", "<"]
                if teaching_level is None:
                    self.logger.warn(
                        "The `level` feature of the teaching item is not set. Level"
                        f" control `{control}` not supported."
                    )
                    continue
                if control == "=":
                    permit_levels.append(teaching_level)
                elif control == ">":
                    permit_levels += list(range(teaching_level + 1, 6))
                elif control == ">":
                    permit_levels += list(range(1, teaching_level))
        return permit_levels

    def make_exam_questions(self, teaching_items):
        num_list = self._get_num_exam_items(teaching_items, self.num_exam_questions)
        final_selected_dataset = None
        for t_item, num_exam in zip(teaching_items, num_list):
            if num_exam == 0:
                continue

            # Filter according to item["subject"]
            if self.same_subject:
                if "subject" not in t_item:
                    self.logger.warn(
                        "The `level` feature of the teaching item is not set. Level"
                        f" control `{control}` not supported."
                    )
                    continue
                exam_selected_dataset = self.exam_selected_dataset.filter(
                    lambda exam_item: exam_item["subject"] == t_item["subject"]
                )

            # Filter according to item["level"]
            permit_levels = self._parse_permit_levels(t_item.get("level", None))
            exam_selected_dataset = exam_selected_dataset.filter(
                lambda exam_item: exam_item["level"] in permit_levels
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


if __name__ == "__main__":
    from pprint import pprint
    from datasets import load_from_disk

    t_dataset = load_from_disk("examples/rationale/data/math_500").to_list()
    top16_maker = MathMetaInfoMaker(
        "examples/rationale/data/math_12k/",
        level_controls=["=", 5],
        num_exam_questions=16,
        random=False,
    )
    random16_maker = MathMetaInfoMaker(
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
