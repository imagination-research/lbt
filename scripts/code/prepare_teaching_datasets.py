import argparse
import os
import json

from datasets import load_from_disk, Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Input baseline directory')
parser.add_argument('--output', type=str, required=True, help='Output directory')
parser.add_argument('--freq', type=int, default=8, help='freq times')
args = parser.parse_args()

if __name__ == "__main__":
    dataset_list = []
    for m in range(args.freq):
        dataset_list.append(load_from_disk(os.path.join(args.input, str(m))))
    
    question_num = len(dataset_list[0][0]["exam_questions"])
    # make a dictionary to store the question items, indexed by the question num
    question_dict = {k: [] for k in range(question_num)}

    # search the dataset to find each rationale
    for dataset in dataset_list:
        for m in range(question_num):
            model_name = list(dataset[0]["exam_details"].keys())[0]

            teacher_item = {}
            teacher_item["task_id"] = dataset[0]["task_id"][m]
            teacher_item["answer"] = dataset[0]["exam_details"][model_name]['answers'][m][0]
            teacher_item["question"] = dataset[0]["exam_questions"][m]
            teacher_item["rationale"] = dataset[0]["exam_details"][model_name]['rationales'][m][0]
            if '/bs-' in args.input:
                teacher_item["tags"] = "Binary Search"
            elif '/dp-' in args.input:
                teacher_item["tags"] = "Dynamic Programming"
            elif 'code_contests' in args.input or "apps" in args.input:
                teacher_item["tags"] = "competition"
            else:
                print(args.input)
                raise NotImplementedError
            teacher_item["model_name"] = model_name
            teacher_item["score"] = dataset[0]["exam_details"][model_name]['scores'][m][0]
            
            # append into question_dict
            question_dict[m].append(teacher_item)
            

    # save the question_dict into a dataset and jsonl file
    for m in range(question_num):
        t_dataset = Dataset.from_list(question_dict[m])
        t_dataset.save_to_disk(os.path.join(args.output, str(m)))

        with open(os.path.join(args.output, str(m), "dataset.jsonl"), "w") as f:
            for item in question_dict[m]:
                f.write(json.dumps(item) + "\n")