import argparse
from glob import glob

from datasets import Dataset, load_from_disk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=str, default="./outputs")
    parser.add_argument("--teacher_exp", type=str)
    parser.add_argument("--teacher_name", type=str)
    parser.add_argument("--dataname", type=str, default="math200")
    parser.add_argument("--num_rationales", type=int, default=256)
    args = parser.parse_args()

    dataset_paths = glob(f"./examples/datasets/math/math_splits/{args.dataname}_r{args.num_rationales}s*")

    for path in dataset_paths:
        split = path.split("/")[-1]
        dataset_split = load_from_disk(path)
        teacher_dataset = load_from_disk(
            f"{args.outputs}/{args.teacher_exp}/rationales/{split}"
        )

        questions = teacher_dataset[0]["exam_questions"]
        rationales = teacher_dataset[0]['exam_details'][args.teacher_name]['rationales']
        answers = teacher_dataset[0]['exam_details'][args.teacher_name]['answers']

        assert len(dataset_split) == len(questions)

        teaching_dataset = []
        for j in range(len(dataset_split)):
            assert dataset_split[j]["question"] == questions[j]

            problem = {}
            problem["question"] = dataset_split[j]["question"]
            index = rationales[j][0].find("[[Final Answer]]")
            problem["solution"] = rationales[j][0][:index].rstrip()
            problem["answer"] = answers[j][0]
            problem["subject"] = dataset_split[j]["subject"]
            problem["level"] = dataset_split[j]["level"]
            problem["unique_id"] = dataset_split[j]["unique_id"]
            teaching_dataset.append(problem)

        Dataset.from_list(teaching_dataset).save_to_disk(f"{args.outputs}/{args.teacher_exp}/teaching/{split}")
