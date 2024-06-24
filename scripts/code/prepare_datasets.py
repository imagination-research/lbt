import os
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset

from lbt.datasets_adapter.utils.fetch_leetcode import fetch_dataset, fetch_solutions
from lbt.datasets_adapter.utils.utils_leetcode import get_api_instance
from lbt.datasets_adapter.utils.clean_leetcode import remove_class_dependent, remove_void, remove_class_impls, remove_examples
from lbt.datasets_adapter.utils.format_leetcode import format_problems, to_jsonl

parser = argparse.ArgumentParser(description="Configuration for building uncontaminated Leetcode Hard dataset")
parser.add_argument('--langs', nargs='+', default=['python3'], help="List of languages.")
parser.add_argument('--output_dir', type=str, default="./examples/leetcode", help="Directory to save the built dataset.")
parser.add_argument('--extract_test_cases', action='store_true', help="If set, test cases will be extracted from problem descriptions using GPT.")
parser.add_argument('--remove_examples', action='store_true', help="If set, examples will be removed. Cannot be used with --extract_test_cases.")
parser.add_argument('--fetch_solutions', action='store_true', help="If set, solutions to problems will be fetched. Currently only supports lang=python3.")
parser.add_argument('--topic',  type=str, default='algorithms', choices=['algorithms'])
parser.add_argument('--difficulty', type=int, default=3, choices=[1, 2, 3], help="Get data of certain difficulty. 1: Easy, 2: Medium, 3: Hard")

args = parser.parse_args()

if __name__ == "__main__":
    # Check LEETCODE environment variables
    try:
        leetcode_session = os.environ["LEETCODE_SESSION"]
    except:
        print("Environment variable LEETCODE_SESSION is not set. Please refer to README")
        exit(1)

    # Check OPENAI environment variables
    if args.extract_test_cases:
        try:
            os.environ["OPENAI_API_KEY"]
            import openai
        except:
            print("Extra dependencies and setup are required for test case extraction. Please refer to README")
            exit(1)
        if args.remove_examples:
            print("Cannot use --remove_examples with --extract_test_cases")
            exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    api_instance = get_api_instance(leetcode_session=leetcode_session, csrf_token=os.environ["CSRF_TOKEN"])
    dataset = fetch_dataset(api_instance, topic=args.topic, difficulty=args.difficulty)

    # use pandas to save pandas format dataset
    # dataset.to_csv(os.path.join(args.output_dir, f'leetcode-{args.difficulty}-{args.topic}.csv'), index=False)

    filtered_dataset = \
        remove_class_impls(
        remove_void(
        remove_class_dependent(dataset))).reset_index(drop=True)

    if args.remove_examples:
        filtered_dataset = remove_examples(filtered_dataset)

    print(f"Filtered out {len(dataset) - len(filtered_dataset)} problem(s)")

    for lang in args.langs:
        print(f"Formatting dataset for {lang}")
        formatted_dataset = format_problems(filtered_dataset, lang)
        if args.extract_test_cases:
            print(f"Extracting test cases for {lang}")
            from lbt.datasets_adapter.utils.add_test_cases import extract_test_cases
            formatted_dataset = extract_test_cases(formatted_dataset, lang)
        if args.fetch_solutions:
            print(f"Fetching solutions for {lang}")
            formatted_dataset = fetch_solutions(formatted_dataset, lang)

        # save into the huggingface datasets in the Humaneval format
        q_templete = "Write a python code \n\"\"\"{}\"\"\"\n to solve the following problem: \n\n{} \n"
        for sample in formatted_dataset:
            # transform signature into class
            sample["signature"] = sample["signature"].replace("(", "(self, ")
            sample["signature"] = 'class Solution():\n    def ' + sample["signature"]

            # transform test cases into class format
            sample["test"] = sample["test"].replace("assert ", "assert Solution().")

            # add new column for question and rationale
            sample["question"] = q_templete.format(sample["signature"], sample["docstring"])
            if "rationale" not in sample.keys():
                sample["rationale"] = ""

        to_jsonl(formatted_dataset, os.path.join(args.output_dir, f'leetcode-{args.difficulty}-{lang}', 'dataset.jsonl'))

        t_dataset = Dataset.from_list(formatted_dataset)
        if "canonical_solution" in t_dataset.features:
            t_dataset = t_dataset.rename_columns({"canonical_solution": "answer"})
        t_dataset.save_to_disk(os.path.join(args.output_dir, f'leetcode-{args.difficulty}-{lang}'))
