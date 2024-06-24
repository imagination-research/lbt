import json
import argparse
from glob import glob

from datasets import Dataset


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval[7:-1]

def process_json_file(json_file, key_mapping = {"problem": "question", "type": "subject"}):
    with open(json_file) as f:
        problem = json.load(f)

    answer = last_boxed_only_string(problem["solution"])
    unique_id = "/".join(json_file.split("/")[-3:])
    problem["answer"] = answer
    problem["unique_id"] = unique_id
    problem["level"] = int(problem["level"][-1])

    for old_key in key_mapping:
        problem = {key_mapping[old_key] if key == old_key else key: value for key, value in problem.items()}

    return problem

def split_dataset(dataset, dataname, num_problems=10, num_rationales=256):
    num_splits = len(dataset) // num_problems
    splits_ids = set()

    for i in range(num_splits):
        dataset_split = []
        start = i * num_problems
        if i != num_splits - 1:
            end = (i + 1) * num_problems
        else:
            end = len(dataset)

        for j in range(start, end):
            problem = dataset[j]
            dataset_split.extend([problem] * num_rationales)
            splits_ids.add(problem['unique_id'])

        Dataset.from_list(dataset_split).save_to_disk(
            f"./examples/datasets/math/math_splits/{dataname}_r{num_rationales}s{i}"
        )

    assert len(splits_ids) == len(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_problems", type=int, default=10)
    parser.add_argument("--num_rationales", type=int, default=256)
    args = parser.parse_args()

    math500_dataset = []
    math500_ids = set()
    with open("./examples/datasets/math/MATH/math500_splits/test.jsonl") as f:
        for line in f:
            problem = json.loads(line)
            problem = {"question" if key == "problem" else key: value for key, value in problem.items()}
            id = problem['unique_id']
            math500_dataset.append(problem)
            math500_ids.add(id)

    math200_dataset = []
    snapshots_dataset = []
    math200_ids = set()
    snapshots_ids = set()
    snapshot1_json_files = glob(f"./examples/datasets/math/MATH/Oct-2023/test/*/*.json")
    for snapshot1_json_path in snapshot1_json_files:
        id = "/".join(snapshot1_json_path.split("/")[-3:])
        snapshots_ids.add(id)

        math_json_path = snapshot1_json_path.replace("/Oct-2023/", "/data/")
        snapshot2_json_path = snapshot1_json_path.replace("/Oct-2023/", "/Nov-2023/")
        snapshot3_json_path = snapshot1_json_path.replace("/Oct-2023/", "/Dec-2023/")

        problem = process_json_file(math_json_path)
        snapshot1 = process_json_file(snapshot1_json_path)
        snapshot2 = process_json_file(snapshot2_json_path)
        snapshot3 = process_json_file(snapshot3_json_path)

        if id in math500_ids:
            math200_dataset.append(problem)
            snapshots_dataset.extend([snapshot1, snapshot2, snapshot3])
            math200_ids.add(id)

    assert len(math200_ids) == 181
    assert len(math500_ids) == 500
    assert len(snapshots_ids) == 1745

    Dataset.from_list(math200_dataset).save_to_disk("./examples/datasets/math/math200")
    Dataset.from_list(math500_dataset).save_to_disk("./examples/datasets/math/math500")
    Dataset.from_list(snapshots_dataset).save_to_disk("./examples/datasets/math/snapshots")

    split_dataset(math200_dataset, "math200", args.num_problems, args.num_rationales)
