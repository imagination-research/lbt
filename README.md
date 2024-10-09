# Can LLMs Learn by Teaching for Better Reasoning? A Preliminary Study


**[[NeurIPS 2024](https://openreview.net/forum?id=0ZZMUjZJYF)]**
**[[arXiv](https://arxiv.org/abs/2406.14629)]**
**[[code](https://github.com/imagination-research/lbt)]**
**[[blog](https://www.microsoft.com/en-us/research/blog/research-focus-week-of-august-26-2024/)]**

This is the code repository of our paper "Can LLMs Learn by Teaching for Better Reasoning? A Preliminary Study". 
Aiming at improving the reasoning ability of LLMs, our paper explores whether or not the current LLMs can "learn by teach (LbT)", which is a well-recognized paradigm in human learning. In addition to improving reasoning, as one can imagine, the ability of LbT could offer exciting opportunities for the models to continuously evolve by teaching other (potentially weaker) models, rather than solely relying on human-produced data or stronger teachers.

We execute the exploration by implementing the LbT idea into well-established pipelines to see if it can improve the reasoning outcomes and ability on several complex tasks (e.g., mathematical reasoning, competition-level code synthesis). Our results suggest LbT's potential for harnessing the diversity offered by different students and facilitating weak-to-strong generalization in improving reasoning.

We believe that this work merely scratches the surface of LbT's potential. As LLMs are exhibiting increasingly intelligence, education approaches beneficial for human learning may play a more crucial role in improving LLMs. To make this vision more concrete, we present a roadmap for incorporating education strategies into LLM learning in Section 6 of our paper. Besides, Appendix D in our paper discusses the detailed research rationale of our work: how we decide the high-level target, the LbT idea, the specific tasks, and the concrete LbT implementations.

If you find this repository or paper useful, you can cite

```
@inproceedings{ning2024lbt,
      title={Can LLMs Learn by Teaching? A Preliminary Study},
      author={Xuefei Ning and Zifu Wang and Shiyao Li and Zinan Lin and Peiran Yao and Tianyu Fu and Matthew B. Blaschko and Guohao Dai and Huazhong Yang and Yu Wang},
      year={2024},
      booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
      url={https://openreview.net/forum?id=0ZZMUjZJYF}
}
```

The repository is organized as follows.
* The core implementation is under [`lbt/`](lbt/).
* The scripts under [`scripts/`](scripts/) are used to prepare the dataset and run the experiments.
* The YAML configurations for M1 are given under [`examples/config/`](examples/config/).
* We put the processed data under [`examples/datasets/`](examples/datasets/). Note that we didn't directly release the dataset we use, but we give out the instructions and scripts to prepare the dataset. 


## Contents
- [Install](#install)
- [Method 1 (M1)](#method-1-m1)
- [Method 2 (M2)](#method-2-m2)
- [Method 3 (M3)](#method-3-m3)
- [Acknowledgement](#acknowledgement)


## Install
Run the following command:
```pip install -e .```

**Required environment variable setups for using the API-based models**: For all the experiments, we use **Azure OpenAI API**. The Azure endpoint and API key should be provided by `export AZURE_OPENAI_ENDPOINT=<API endpoint>`, `export AZURE_OPENAI_API_KEY=<API key>`.

Or, you can choose to **specify them in the configuration file**, for example:
```
student_model_cfgs:
 - model_type: azure_openai
 model_cfg:
 model: gpt-35-turbo
 api_endpoint: <YOUR API ENDPOINT>
 api_key: <YOUR API KEY>
 api_version: <YOUR API VERSION, default to "2024-02-15-preview">
```

## Method 1 (M1)

M1 incorporates the LbT idea into the search-based output generation pipeline. The method goes as follows (see the paper for more detailed descriptions):
* Step 1: for each teaching problem (TP), we generate `num_rationales` pairs of teaching rationales and answers (TR-TA pairs).
* Step 2: Then, each TR-TA pair is used as an in-context learning (ICL) example to guide the student model in solving a series of exam problems (EPs). With the produced Exam Rationale (ER) and Exam Answers (EAs), each student will then receive an exam score (i.e., accuracy of EAs), denoted as the LbT score.
* Step 3: The LbT score can be used as a quality assessment of the corresponding TR-TA pair. We consider two ways to select the final TA: (1) Selecting the TR-TA pair with the highest LbT score. We denote this approach as "M1 (MAX)". (2) Selecting the weighted-voted TA using the LbT score as the weight. We denote this approach as "M1 (SUM)".

### M1 Implementation
Here, we briefly go through the M1 implementation. You can skip to [the next sections](#step-0-build-the-dataset-for-mathematical-reasoning) for the step-by-step commands for running experiments.

The implementation orchestrates the running of several components, including the **exam maker** that decides the EPs given the TP-TR pair, the **exam prompter** that assembles the exam prompt, the **model** that takes the prompt and output the ERs and EAs, and the **exam scorer** that parses the model output and scores the ER and EA.

We use the `scripts/exam.py` script for the TR generation from the teacher model (Step 1), as well as the student examination on the EPs (Step 2).
The script takes one positional argument and three named arguments:
* `cfg_file`: A YAML configuration file specifying the settings for the components. 
* `--output-path`: A path to save the output result.
* `--teaching-dataset-file`: (Optional) The **teaching dataset**. The script selects teaching items from this dataset as the ICL examples. The selection is based on the `teaching_plans` specified in the configuration.
* `--exam-dataset-file`: The **exam dataset**. The **exam maker** will select EPs from this exam dataset for the model to answer. The available strategies of selecting EPs are implemented in `lbt.exam_maker`.

The YAML configuration file specifies the settings of the components. Here explains its contents:
* `student_model_cfgs`: A list of model configs. Each model config is a dict containing three keys:
  * `model_type`: A string specifying the class of **model** component. Use `azure_openai` for GPTs through the Azure API; Use `huggingface` for open-source models through the HuggingFace transformer library. See the `lbt.models.base` module for the implementation of **model** component classes.
  * `model_cfg`: A dict specifying the arguments that will be used to instantiate the **model** component.
  * `sample_cfg`: A dict specifying the sampling configurations for the model component. Can override the `general_student_sample_cfg` configurations.
* `exam_maker_type` and `exam_maker_cfg`: The class and instantiating arguments of the **exam maker** component.
* `exam_prompter_type` and `exam_prompter_cfg`: The class and instantiating arguments of the **exam prompter** component. For example, the prompt template should be provided here.
* `exam_scorer_type` and `exam_scorer_cfg`: The class and instantiating arguments of the **exam scorer** component.

There are two other configurations in the YAML file:
* `teaching_plans`: A string or a list.
  * If `teaching_plans` is a list, every item in the list specifies the teaching plan for one exam, which is a list of the indexes of the teaching items. For example, the configuration `teaching_plans: [[0], [1,2]]` means each student will take two exams: the first exam will use the 0-th teaching item in the teaching dataset as the ICL example, while the second exam will use the 1-th and 2-th items in the teaching dataset as the ICL examples.
  * If `teaching_plans` is a string, it must be one of `no demo` or `every` (default).
    * `every`: Each item in the teaching dataset will be separately used as the ICL example in one exam, i.e., equivalent to `teaching_plans: [[0], [1], [2], ..., [N-1]]`, where N is the number of items in the teaching dataset.
    * `no demo`: Don't use items from the teaching dataset as the ICL examples, and only take one exam, i.e., equivalent to `teaching_plans: [[]]`. This option is used to support generating TRs in Step 1 (where no TP-TR pair needs to be used as the ICL example). 
* `general_student_sample_cfg`: A dict. The general sampling configurations for all models.

### Step 0: Build the Dataset for Mathematical Reasoning

Download the [MATH](https://github.com/hendrycks/math/) dataset, the [MATH()](https://github.com/ConsequentAI/fneval) dataset and the [MATH500](https://github.com/openai/prm800k/tree/main/prm800k/math_splits) train-test splits, and organize them as follows:
```
examples/datasets/math/MATH
|-- data # MATH dataset
|  |-- train
|  |-- test
|-- Oct-2023 # MATH() dataset Oct-2023 snapshot
|  |-- test
|-- Nov-2023 # MATH() dataset Nov-2023 snapshot
|  |-- test
|-- Dec-2023 # MATH() dataset Dec-2023 snapshot
|  |-- test
|-- math500_splits # The train-test split of MATH dataset from the "Let's verify step by step" paper
|  |-- train.jsonl
|  |-- test.jsonl
```

Then, run the following command to build the dataset:
```bash
python scripts/math/prepare_datasets.py
```

This script filters out the 181 problems from MATH500 test split (500 problems in total) that have 3 snapshot questions in MATH(). These questions will be regarded as the TPs. This script repeats them for `num_rataionles` times, and finally splits the 181x`num_rationales` items into `num_splits` splits, where each split has `num_problems` items. This splitting is for easy diagnosis and parallel experiments: Instead of generating TRs for all the items, we can generate TRs for one split on one GPU and check them separately. This script receives two command-line arguments:
* `--num_rationales`: Repeat each TP for `num_rationales` times in the built dataset. This is for the batched generation of the TRs in the following Step 1.
* `--num_problems`: Each split has `num_problems` items.


### Step 0: Build the Dataset for Code Synthesis

Due to the copyright constraint, we don't directly open-source the data originated from [Leetcode](https://leetcode.com). To replicate the paper's results, one needs to download the data from LeetCode and build the dataset as follows.

Set the environment variable `LEETCODE_SESSION` to the cookie `LEETCODE_SESSION` from a signed-in Leetcode session. This cookie can be found by using the browser DevTools or by using a browser extension like [EditThisCookie](https://www.editthiscookie.com/). Set the environment variable `CSRF_TOKEN`.

```bash
export LEETCODE_SESSION=...
```

Then, run the following command to fetch the leetcode dataset:
```bash
python scripts/code/prepare_datasets.py --langs python3 --log_level INFO --output_dir ./examples/datasets/code/LEETCODE --csrf_token <YOUR OWN CSRFTOKEN> --difficulty 2
```
The output dir is `./examples/datasets/code/LEETCODE/leetcode-{difficulty}-{lang}`.

<!-- This script receives some other command-line arguments:
* `--topic`: Get data for topics other than `algorithms`.
* `--difficulty`: Get data of certain difficulty. 1 == easy; 2 == medium; 3 == hard. -->

> Problems are transformed into the `HumanEval` format and saved in the dataset file.

> You can select the coding problem in the [leetcode study plan](https://leetcode.com/studyplan/) for exam.

To summarize, the leetcode datasets are organized as follows:
```
examples/datasets/code/LEETCODE
|-- <DATASET NAME 1>
|-- <DATASET NAME 2>
|-- <DATASET NAME 3>
|...
```

### Step 1: Generate `num_rationales` TR-TA pairs for each TP from the teacher model

To generate `num_rationales` TR-TA pairs from the teacher, one can set the `teaching_plan` configuration in the YAML file to `no demo`, and run the following commands:

**For mathematical reasoning**:
```bash
for i in $(seq 0 NUM_SPLITS); do
  python scripts/exam.py \
    ./examples/config/math/<TEACHER NAME>_trgen.yaml \
    --output-path "<OUTPUT PATH>/<TEACHER EXP>/rationales/math200_r256s$i" \
    --exam-dataset-file "./examples/datasets/math/math_splits/math200_r256s$i"
done
```

`NUM_SPLITS` depends on how you set `--num_problems` in the previous step. The generated TR-TA pairs will be organized as `"<OUTPUT PATH>/<TEACHER EXP>/rationales/math200_r256s*"`. Then, we need to prepare the teaching dataset for Step 2's exams as the desired format.
```bash
python scripts/math/prepare_teaching_dataset.py \
  --outputs <OUTPUT PATH> \
  --teacher_exp <TEACHER EXP> \
  --teacher_name <TEACHER NAME>
```

The prepared teaching dataset will be saved as `<OUTPUT PATH>/<TEACHER EXP>/teaching/math200_r256s*`.

**For code synthesis**:
```bash
python scripts/code_exam.py \
    ./examples/config/code/<TEACHER NAME>_trgen.yaml \
    --output-path "<YOUR TR OUTPUT PATH>" \
    --exam-dataset-file "examples/datasets/code/LEETCODE/<DATASET NAME>"
```
> NOTE: The logic in `code_exam.py` is almost the same as that in `exam.py`; it just changes some dataset parsing.

Then, we need to prepare the teaching dataset for Step 2's exams as the desired format.
```bash
python scripts/code/prepare_teaching_dataset.py \
  --input "<YOUR TR OUTPUT PATH>" \
  --outputs "<YOUR TEACHING DATASET PATH>" \
  --freq 8 
```

### Step 2: Student examination to get the LbT scores
Each TR-TA pair is separately used as the ICL example for the student to answer EPs.

**For mathematical reasoning**:
```bash
for i in $(seq 0 NUM_SPLITS); do
  python scripts/exam.py \
    ./examples/config/math/<STUDENT NAME>_exam.yaml \
    --output-path "<OUTPUT PATH>/<STUDENT EXP>/<TEACHER EXP>_exams/math200_r256s$i" \
    --teaching-dataset-file "./<TEACHER EXP>/teaching/math200_r256s$i" \
    --exam-dataset-file "./examples/datasets/math/snapshots"
done
```

The generated outputs will be organized as `<OUTPUT PATH>/<STUDENT EXP>/<TEACHER EXP>_exams/math200_r256s*`.

**For code synthesis**:
```bash
python scripts/code_exam.py \
  ./examples/config/math/<STUDENT NAME>_exam.yaml \
  --output-path "<YOUR STUDENT EXAM OUTPUT PATH>" \
  --teaching-dataset-file "<YOUR TEACHING DATASET PATH>" \
  --exam-dataset-file "examples/datasets/code/LEETCODE/<DATASET NAME>"
```

### Step 3: Select the final TA and Compare with the baseline methods

Use the LbT scores to find a TR-TA pair for each TP and calculate the accuracy.

**For mathematical reasoning**:
You should organize the outputs from the previous steps as
```
<OUTPUT PATH>
|-- <TEACHER EXP>/rationales/math200_r256s*
|-- <STUDENT EXP>/<TEACHER EXP>_exams/math200_r256s*
```

Then run the following command:
```bash
python scripts/math/search_rationale.py
  --outputs <OUTPUT PATH> \
  --teacher_exp <TEACHER EXP> \
  --teacher_name <TEACHER NAME> \
  --student_exp <STUDENT EXP> \
  --student_name <STUDENT NAME>
```

**For code synthesis**:
```bash
python scripts/code/search_rationale.py
  --input "<YOUR STUDENT EXAM OUTPUT PATH>" \
  --num_sample 8
```
This code will output a `.txt` result file in `<YOUR STUDENT EXAM OUTPUT PATH>`. If you want to submit the code to LEETCODE for evaluating the submit score (S-score), you need to use `--is_submit`.

## Method 2 (M2)
In M1, each TR-TA pair receives a LbT score that measures its quality. We collect the LbT scores of many TR-TA pairs and use them to finetune the teacher model with DPO. We prepare the data and directly use this [codebase](https://github.com/eric-mitchell/direct-preference-optimization) for DPO training. Thus, we omit the code of M2 here.

## Method 3 (M3)
Using the idea of LbT, M3 optimizes the in-context learning examples for a given (binary classification) task using a teacher LLM and one or more student LLMs.

M3 is implemented based on the [codebase](https://github.com/microsoft/LMOps/tree/main/prompt_optimization) of the paper [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://aclanthology.org/2023.emnlp-main.494/). We provide a patch to the original codebase to support the LbT setting. The patch is under `lbt/patch/`.

### Setting up M3 codebase
1. Clone the [LMOps codebase](https://github.com/microsoft/LMOps/)
```bash
git clone https://github.com/microsoft/LMOps/
cd LMOps
git checkout 0bdb5d7
```
2. Apply the patch
```bash
git apply <PATH_TO_LBT>/lbt/patch/lmops-lbtm3.patch
cd prompt_optimization
```

The codebase is set up under the `LMOps/prompt_optimization` directory. From now on, we will refer to this directory as the root directory.

### Configure LLM endpoints
Create a `config.py` file following the template in `config.py.example`.
This file contains the endpoints of the LLMs to be used as teacher and student models.
LLM endpoints can be any OpenAI-compatible API endpoint, including the official OpenAI API and self-hosted models using [vLLM](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html).

### Download the data
Run `dl_data.py` inside the subdirectories of the `data` directory to download the data for the tasks.

### Run the code
```bash
python3 main.py --init_method train --task fallacy --out outputs/fallacy.out --student_models llama3-8b,mistral-7b --teacher_model llama3-70b
```
The above command will run the optimization for the fallacy detection task using the teacher model `llama3-70b` and student models `llama3-8b` and `mistral-7b`. The output will be saved in `outputs/fallacy.out`. The `--init_method` argument specifies the initialization method for the in-context examples. The available options are `train` and `gen`. The `train` option will randomly select examples from the training set, while the `gen` option will perform a zero-shot generation of examples using the teacher model.


```bash
python main.py --help
```

For usage instructions. Some of the arguments include:

* `--task`: Task name like 'fallacy', 'liar', etc.
* `--max_threads`: Maximum number of threads to be used.
* `--n_icl`: Number of in-context learning examples to be optimized.
* `--teacher_model`: Teacher model name, must be an endpoint in `config.py`.
* `--student_models`: Comma-separated list of student model names. Must be endpoints in `config.py`.
* `--gradient_mix`: Number of reflections to be combined to generate new examples. Default is 1 (no mixing).
* `--init_method`: Initialization method for in-context examples. Options: `gen`, `train`.
* `...`: Various other parameters related to optimization and evaluation.

## Code Acknowledgement
* We refer to [the Leetcode-Hard Gym codebase](https://github.com/GammaTauAI/leetcode-hard-gym) of [the Reflexion paper](https://arxiv.org/abs/2303.11366) to implement M1 for code synthesis,  and [the OpenCompass codebase](https://github.com/open-compass/opencompass) for the answer extraction for mathematical reasoning (NOTE: We change the implementation to improve the extraction accuracy). 
* We use the [codebase](https://github.com/eric-mitchell/direct-preference-optimization) of the paper [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) for M2 experiment.
* We base the M3 implementation on the [codebase](https://github.com/microsoft/LMOps/tree/main/prompt_optimization) of the paper [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://aclanthology.org/2023.emnlp-main.494/).
