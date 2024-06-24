from datasets import load_dataset, Dataset
import json


## -------------------------------------------------------------------------------------------------
## Dataset Transformation
## -------------------------------------------------------------------------------------------------
def SynthesisDatasetTrans(code_dataset):
    Synthesis_Prompt = "Write a python function '{}' to solve the following problem: {} Please note: (1) Import the necessary Python packages, and  If the function head requires additional packages, ensure they are imported before the function head; (2) Pay attention to Python's indentation format; (3) Absolutely refrain from outputting any unnecessary explanatory text after generating the code."

    Synthesis_list = []
    for code_sample in code_dataset:
        index = code_sample["prompt"].find("def")
        code_sample_temp = code_sample["prompt"][index:].split('"""')
        if len(code_sample_temp) == 1:
            code_sample_temp = code_sample["prompt"][index:].split("'''")
        code_sample["question"] = Synthesis_Prompt.format(
            code_sample_temp[0].strip(), code_sample_temp[1].strip()
        )

        # deal with test
        index = code_sample["test"].find("def")
        code_sample["test"] = code_sample["test"][index:]
        if "def" in code_sample["test"]:
            code_sample["test"] = code_sample["test"] + "\n\ncheck({})".format(
                code_sample["entry_point"]
            )

        Synthesis_list.append(code_sample)

    return Synthesis_list


def DebugDatasetTrans(code_dataset):
    Debug_Prompt = "{}\n\nFix bugs in {}."

    Debug_list = []
    for code_sample in code_dataset:
        code_sample_temp = code_sample["prompt"].split('"""')
        if len(code_sample_temp) == 1:
            code_sample_temp = code_sample["prompt"].split("'''")

        # TODO: add some bugs here, we can use teacher to generate some buggy code.
        code_sample_temp = code_sample_temp[0] + code_sample["canonical_solution"]
        Debug_list.append(
            Debug_Prompt.format(code_sample_temp, code_sample["entry_point"])
        )
    return Debug_list


def ExplainDatasetTrans(code_dataset):
    Explain_Prompt = "{}\n\nProvide a concise natural language description of the function using at most 500 characters."

    Explain_list = []
    for code_sample in code_dataset:
        code_sample_temp = code_sample["prompt"].split('"""')
        if len(code_sample_temp) == 1:
            code_sample_temp = code_sample["prompt"].split("'''")
        code_sample_temp = code_sample_temp[0] + code_sample["canonical_solution"]
        Explain_list.append(Explain_Prompt.format(code_sample_temp))
    return Explain_list


if __name__ == "__main__":
    # We use this code to evaluate the coding ability of the instruction-tuned LLMs.
    code_dataset = load_dataset("openai_humaneval", split="test")

    # generate the synthesis data list
    synthesis_data_list = SynthesisDatasetTrans(code_dataset)
    # dump jsonl
    with open(
        "./examples/datasets/humaneval.jsonl", "w"
    ) as file:
        for sample in synthesis_data_list:
            file.write(json.dumps(sample) + "\n")

    synthesis_data_list = Dataset.from_list(synthesis_data_list)
    synthesis_data_list.save_to_disk(
        "./datasets/datasets/humaneval"
    )
