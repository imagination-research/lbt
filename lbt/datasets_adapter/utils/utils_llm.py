import os
from typing import Dict, List, Sequence
import json
import yaml
from termcolor import colored

from fastchat.model import get_conversation_template
from fastchat.conversation import get_conv_template

from lbt.base import Component


class LanguageFunction:
    def __init__(self, config: Dict, **model_kwargs) -> None:
        self.chat_model = Component.init_from_cfg(config["gpt_model_cfgs"], "model")
        self.sample_cfg = config["gpt_model_cfgs"]["sample_cfg"]

        # read prompts
        function = dict(config["function"])
        self.stub_items = function.get("stub_items", [])
        self.exam_template = function["exam_template"]

    def __call__(self, callback=False, **kwargs) -> Dict:
        """
        Call the Agent Function with the given arguments.
        """
        # add fschat
        try:
            # exact match conversation name
            conv = get_conv_template(self.chat_model.conv_template_type)
        except KeyError:
            # get through model adapter
            conv = get_conversation_template(self.chat_model.conv_template_type)
            # For base model, use conv_template type "raw"

        for t_item in self.stub_items:
            if t_item["role"] == "user":
                conv.append_message(conv.roles[0], t_item["content"])
            elif t_item["role"] == "assistant":
                conv.append_message(conv.roles[1], t_item["content"])
            else:
                raise ValueError(f"Invalid role: {t_item['role']}")

        # exam samples
        exam_user = self.exam_template.format(**kwargs)
        exam_assistant = None

        conv.append_message(conv.roles[0], exam_user)
        conv.append_message(conv.roles[1], None)

        # call the chat model for responses
        response = self.chat_model.text_generator(
            (conv, exam_assistant), return_full_text=False, **self.sample_cfg
        )
        response_dict = {"response": response}
        return response_dict

    @classmethod
    def from_yaml(cls, filepath: str):
        """
        Load an agent from a YAML file.

        Args:
            filepath (str): The path to the YAML file.

        Returns:
            Agent: The agent.
        """
        with open(filepath, "r", encoding="utf-8") as file:
            yaml_obj = yaml.safe_load(file)
        return cls(yaml_obj)
