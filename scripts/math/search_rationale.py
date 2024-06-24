import argparse
from glob import glob

from tqdm import tqdm
from datasets import load_from_disk

from lbt.datasets_adapter.math_dataset import MathExamScorer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=str, default="./outputs")
    parser.add_argument("--teacher_exp", type=str)
    parser.add_argument("--teacher_name", type=str)
    parser.add_argument("--student_exp", type=str)
    parser.add_argument("--student_name", type=str)
    parser.add_argument("--dataname", type=str, default="math200")
    parser.add_argument("--num_rationales", type=int, default=256)
    parser.add_argument("--num_exam_questions", type=int, default=3)
    parser.add_argument("--num_repetitions", type=int, default=3)
    args = parser.parse_args()

    print(args)

    scorer = MathExamScorer(False)
    exam_dataset = load_from_disk(f"./examples/datasets/math/snapshots")
    dataset_paths = glob(f"./examples/datasets/math/math_splits/{args.dataname}_r{args.num_rationales}s*")

    num_prev_problems = 0
    accuracy = [0] * 4

    for i in tqdm(range(len(dataset_paths))):
        path = f"./examples/datasets/math/math_splits/{args.dataname}_r{args.num_rationales}s{i}"
        split = path.split("/")[-1]
        dataset_split = load_from_disk(path)
        teacher_dataset = load_from_disk(f"{args.outputs}/{args.teacher_exp}/rationales/{split}")
        student_dataset = load_from_disk(f"{args.outputs}/{args.student_exp}/{args.teacher_exp}_exams/{split}")

        teacher_rationales = teacher_dataset[0]['exam_details'][args.teacher_name]['rationales']
        teacher_answers = teacher_dataset[0]['exam_details'][args.teacher_name]['answers']
        gt_rationales = teacher_dataset[0]['exam_gt_rationales']
        gt_answers = teacher_dataset[0]['exam_gt_answers']

        num_problems = len(dataset_split) // args.num_rationales
        for j in range(num_problems):
            all_answers = {}
            rationale_start = args.num_rationales * j
            for k in range(rationale_start, rationale_start + args.num_rationales):
                student_rationales = student_dataset[k]["exam_details"][args.student_name]["rationales"]
                answer = teacher_answers[k][0]

                lbt_score = 0
                for l in range(args.num_exam_questions * args.num_repetitions):
                    question = student_dataset[k]["exam_questions"][l]
                    snapshot_start = args.num_exam_questions * (num_prev_problems + j)
                    snapshot_end = args.num_exam_questions * (num_prev_problems + j + 1)
                    for index in range(snapshot_start, snapshot_end):
                        if exam_dataset[index]["question"] == question:
                            break
                    assert exam_dataset[index]["question"] == question

                    exam_gt_rationale = exam_dataset[index]['solution']
                    exam_gt_answer = exam_dataset[index]['answer']
                    student_rationale = student_rationales[l][0]
                    gt = {"answer": f"[[Solution]]:\nLet's think step by step.\n\n{exam_gt_rationale}\n\n[[Final Answer]]:\n${exam_gt_answer}$\n"}
                    exam = {"rationale": student_rationale}
                    lbt_score += scorer.score_exam_result(gt, exam)

                if answer in all_answers:
                    all_answers[answer][0] += 1
                    all_answers[answer][1] = max(lbt_score, all_answers[answer][1])
                    all_answers[answer][2] += lbt_score
                    all_answers[answer][3] = all_answers[answer][2] / all_answers[answer][0]
                else:
                    all_answers[answer] = [1, lbt_score, lbt_score, lbt_score]

            for mode in ["MAJ", "MAX", "SUM", "AVG"]:
                if mode == "MAJ":
                    dim = 0
                elif mode == "MAX":
                    dim = 1
                elif mode == 'SUM':
                    dim = 2
                elif mode == "AVG":
                    dim = 3

                winner = sorted(all_answers.items(), key=lambda item: (item[1][dim], item[1][0]))[-1][0]

                for k in range(rationale_start, rationale_start + args.num_rationales):
                    gt_rationale = gt_rationales[k]
                    gt_answer = gt_answers[k]
                    teacher_rationale = teacher_rationales[k][0]
                    teacher_answer = teacher_answers[k][0]

                    if teacher_answer == winner:
                        gt = {"answer": f"[[Solution]]:\nLet's think step by step.\n\n{gt_rationale}\n\n[[Final Answer]]:\n${gt_answer}$\n"}
                        exam = {"rationale": teacher_rationale}
                        accuracy[dim] += scorer.score_exam_result(gt, exam)
                        break

        num_prev_problems += num_problems

    for mode in ["MAJ", "MAX", "SUM", "AVG"]:
        if mode == "MAJ":
            dim = 0
        elif mode == "MAX":
            dim = 1
        elif mode == 'SUM':
            dim = 2
        elif mode == "AVG":
            dim = 3

        accuracy[dim] *= (100 / num_prev_problems)

        print(f"{mode}: {accuracy[dim]:.2f}")
