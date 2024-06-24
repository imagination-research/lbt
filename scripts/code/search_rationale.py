from lbt.datasets_adapter.leetcode_sub.types import LeetCodeSubmission, ProgrammingLanguage
from lbt.datasets_adapter.leetcode_sub.environment import LeetCodeEnv

import argparse
import os
import json
import tqdm
import signal
import contextlib
import time

from datasets import load_from_disk, Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Input directory')
parser.add_argument('--is_submit', action='store_true', help='submit mode')
parser.add_argument('--resume', action='store_true', help='resume submission')
parser.add_argument('--num_sample', type=int, default=8, help='number of samples')
args = parser.parse_args()

class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def eval_code(is_submit, code, question_slug, visible_score):
    # if "AssertionError" in code:
    #     return 0.0
    if is_submit:
        # define the submission question
        sub = LeetCodeSubmission(code=code.replace('\/', '/'),
                lang=ProgrammingLanguage.PYTHON3,
                question_slug=question_slug,
                timeout=20)
        
        # if there are some exception, retry
        try_num=0
        while True:
            if try_num > 5:
                return -1.0
            try:
                try_num += 1
                # time.sleep(30)
                env = LeetCodeEnv()
                with time_limit(40):
                    status, reward, done, submission_result = env.step(sub)
                    print(status, reward, done, submission_result)

                # number of correct and testcases
                total_correct = submission_result['total_correct']
                total_testcases = submission_result['total_testcases']
                return total_correct / total_testcases
            # if keyboard interrupt, exit
            except KeyboardInterrupt:
                print("******** Keyboard Interrupt ********")
                exit()
            except TimeoutException:
                print("******** Timeout ********")
                continue
            except Exception as e:
                print(f"******** {e} ********")
                continue
    else:
        return visible_score

if __name__ == "__main__":
    # Check LEETCODE environment variables
    if args.is_submit:
        output_path = os.path.join(args.input, 'results-submit.jsonl')
        try:
            leetcode_session = os.environ["LEETCODE_SESSION"]
        except:
            print("Environment variable LEETCODE_SESSION is not set. Please refer to README")
            exit(1)
    else:
        output_path = os.path.join(args.input, 'results-visible.jsonl')
    
    if not args.resume:
        if os.path.exists(output_path):
            os.remove(output_path)
    else:
        if os.path.exists(output_path):
            past_data = []
            with open(output_path, 'r+') as f:
                for line in f:
                    past_data.append(json.loads(line))
            past_name = len(past_data) // args.num_sample
            past_test_id = len(past_data) % args.num_sample
        else:
            past_name = 0
            past_test_id = 0

    # find all the file names in args.input
    file_names = os.listdir(args.input)
    # remove all the file name with .json
    file_names = [file_name for file_name in file_names if '.json' not in file_name]
    for file_name in range(len(file_names)):
        if args.resume and file_name < past_name:
            continue
        datasets = load_from_disk(os.path.join(args.input, str(file_name)))
        for test_id, dataset in enumerate(datasets):
            if args.resume and test_id < past_test_id and file_name == past_name:
                continue
            # get exam information
            s_model_name = list(dataset["exam_details"].keys())[0]
            s_task_ids = dataset['task_id']
            s_answer = dataset["exam_details"][s_model_name]["answers"]
            s_visible_scores = dataset["exam_details"][s_model_name]["scores"]

            # check oracle-1
            if 'pipeline-1' in args.input:
                # if oracle-1, get teacher information
                t_task_id = dataset["teaching_items"][0]["task_id"]
                t_answer = dataset["teaching_items"][0]["answer"]
                # check if scores in dictionary
                if "score" in dataset["teaching_items"][0]:
                    t_visible_score = dataset["teaching_items"][0]["score"]
                else:
                    t_visible_score = dataset["teaching_items"][0]["scores"]
                t_final_score = eval_code(args.is_submit, t_answer, t_task_id, t_visible_score)

                # refine the exam information
                rm_idx = s_task_ids.index(t_task_id)
                s_task_ids.pop(rm_idx)
                s_answer.pop(rm_idx)
                s_visible_scores.pop(rm_idx)

            s_final_scores = []
            for i in tqdm.tqdm(range(len(s_task_ids))):
                s_final_scores.append(eval_code(args.is_submit, s_answer[i][0], s_task_ids[i], s_visible_scores[i][0]))

            # calculate the average score
            s_avg_score = sum(s_final_scores) / len(s_final_scores)

            # write the submit_scores in a json file
            with open(output_path, 'a') as f:
                item = {"t_task_id": t_task_id, "t_score": t_final_score, "exam_id": test_id, "s_score": s_final_scores, "s_avg_score": s_avg_score}
                f.write(json.dumps(item) + "\n")