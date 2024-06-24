import pandas as pd
from .utils_llm import LanguageFunction
import os
import inspect

UTILS_DIR = os.path.dirname(inspect.getabsfile(inspect.currentframe()))


def extract_test_cases(dataset: pd.DataFrame, lang: str) -> pd.DataFrame:
    """
    Add test cases to the dataset
    Adds columns: 'test_cases' (List[str])
    """
    dataset = dataset.copy()
    dataset.reset_index(inplace=True, drop=True)
    dataset["test_cases"] = None

    # extract test cases
    for ind, row in dataset.iterrows():
        print(f"Extracting test cases for problem {ind+1}/{len(dataset)}")
        examples = extract_examples(row["description"])
        function_signature = row["signature"]
        test_cases = examples_to_test_cases(examples, function_signature, lang)
        dataset.at[ind, "test_cases"] = test_cases
    return dataset


def extract_examples(description):
    """
    Extract a natural language representation of the examples from the description
    """
    inputs = [l for l in description.split("\n") if l.strip().startswith("Input")]
    outputs = [
        l.strip("Output: ")
        for l in description.split("\n")
        if l.strip().startswith("Output")
    ]

    examples = []

    for i, (input_str, output_str) in enumerate(zip(inputs, outputs)):
        example_str = f"Example {i+1}:\n{input_str}\nOutput: {output_str}"
        examples.append(example_str)
    return "\n\n".join(examples)


def examples_to_test_cases(
    examples: str, function_signature: str, language: str
) -> str:
    """
    Extract test cases from a natural language representation of the examples
    """
    lang_function = LanguageFunction.from_yaml(
        os.path.join(UTILS_DIR, "extract_tests.yaml")
    )
    response = lang_function(
        function_signature=function_signature, examples=examples, language=language
    )
    test_cases = response["response"].split("\n")
    return test_cases
