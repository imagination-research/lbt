from abc import abstractmethod
import os
from openai import OpenAI, AzureOpenAI
import openai

import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


from lbt.base import Component


class BaseModel(Component):
    REGISTRY = "model"

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def conv_template_type(self):
        pass

    @abstractmethod
    def text_generator(
        self, iterator, batch_size, num_return_sequences, **generate_kwargs
    ):
        """
        Return a generator object that yields the generations to the prompts in `iterator` one by one
        """


class StubModel(BaseModel):
    NAME = "stub"

    @property
    def name(self):
        return "stub"

    @property
    def conv_template_type(self):
        return "raw"

    def text_generator(
        self, iterator, batch_size, num_return_sequences, **generate_kwargs
    ):
        for _ in iterator:
            yield [
                {"generated_text": f"random answer {index}"}
                for index in range(num_return_sequences)
            ]


class OpenAIModel(BaseModel):
    NAME = "openai"

    def __init__(
        self,
        model,
        name=None,
        api_key=None,
        fastchat=True,
    ):
        super().__init__()

        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = OpenAI(api_key=self._api_key)
        self._name = name or model
        self.fastchat = fastchat

    @property
    def conv_template_type(self):
        return "chatgpt"

    @property
    def name(self):
        return self._name

    def text_generator(self, iterator, return_full_text, **generate_kwargs):
        return self._request(
            iterator, return_full_text=return_full_text, **generate_kwargs
        )

    @retry(
        retry=retry_if_not_exception_type((openai.BadRequestError, TypeError)),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(15),
    )
    def _retry_wrapper(self, messages, **generate_kwargs):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **generate_kwargs,
        )
        return response

    def _request(self, conv_iterator, return_full_text, **generate_kwargs):
        for conv, partial_answer in conv_iterator:
            messages = []
            messages.append({"role": "system", "content": conv.system_message})
            if partial_answer is not None:
                conv.messages[-1][1] = partial_answer
            else:
                del conv.messages[-1]
            for message in conv.messages:
                messages.append({"role": message[0], "content": message[1]})
            response = self._retry_wrapper(messages, **generate_kwargs)
            answers = []
            for choice in response.choices:
                answer = choice.message.content
                if return_full_text:
                    answer = partial_answer + answer
                answers.append({"generated_text": answer})
            yield answers


class AzureOpenAIModel(OpenAIModel):
    NAME = "azure_openai"

    def __init__(
        self,
        model,
        name=None,
        api_key=None,
        api_endpoint=None,
        api_version="2024-02-15-preview",
        fastchat=True,
    ):
        BaseModel.__init__(self)

        self._model = model
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self._api_endpoint = api_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self._api_version = api_version
        self._client = AzureOpenAI(
            api_key=self._api_key,
            api_version=self._api_version,
            azure_endpoint=self._api_endpoint,
        )
        self._name = name or "azure_" + model
        self.fastchat = fastchat


class HFModel(BaseModel):
    NAME = "huggingface"

    def __init__(
        self,
        path,
        pt_path=None,
        max_new_tokens=1024,
        conv_template_type=None,
        name=None,
        fastchat=False,
    ):
        super().__init__()

        self.path = path
        self._conv_template_type = conv_template_type or path
        self._name = name or path
        self.fastchat = fastchat

        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).eval()

        if pt_path:
            state_dict = torch.load(pt_path)["state"]
            self.model.load_state_dict(state_dict)
            print(f"loading pre-trained weights from {pt_path}")

        self.generator = pipeline(
            task="text-generation",
            tokenizer=self.tokenizer,
            model=self.model,
            device_map="auto",
            trust_remote_code=True,
        )
        # set default max_new_tokens
        if self.generator.model.generation_config.max_new_tokens is None:
            self.generator.model.generation_config.max_new_tokens = max_new_tokens

        # set padding token
        if self.generator.tokenizer.pad_token_id is None:
            if self.generator.model.generation_config.pad_token_id is not None:
                self.generator.tokenizer.pad_token_id = (
                    self.generator.model.generation_config.pad_token_id
                )
            else:
                eos_token_id = self.generator.model.generation_config.eos_token_id
                if isinstance(eos_token_id, (list, tuple)):
                    eos_token_id = eos_token_id[0]
                self.generator.tokenizer.pad_token_id = eos_token_id

    @property
    def conv_template_type(self):
        return self._conv_template_type

    @property
    def name(self):
        return self._name

    def _transform_conv_iterator_to_prompt_iterator(self, conv_iterator):
        for conv, partial_answer in conv_iterator:
            if self.fastchat:
                prompt = conv.get_prompt()
            else:
                prompt = self.generator.tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True
                )

            if partial_answer is not None:
                prompt += partial_answer

            yield prompt

    def text_generator(
        self, iterator, batch_size, num_return_sequences, **generate_kwargs
    ):
        # TODO: parallel test
        if "llama-3" in self.path.lower():
            terminators = [
                self.generator.tokenizer.eos_token_id,
                self.generator.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]

            return self.generator(
                self._transform_conv_iterator_to_prompt_iterator(iterator),
                batch_size=batch_size,
                num_return_sequences=num_return_sequences,
                eos_token_id=terminators,
                **generate_kwargs,
            )

        return self.generator(
            self._transform_conv_iterator_to_prompt_iterator(iterator),
            batch_size=batch_size,
            num_return_sequences=num_return_sequences,
            **generate_kwargs,
        )

