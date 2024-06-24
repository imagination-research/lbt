class QAItem(dict):
    def __init__(self, question, rationale=None, answer=None, prompt=None, task_id=None):
        super().__init__()
        self["question"] = question
        self["rationale"] = rationale
        self["answer"] = answer
        self["prompt"] = prompt
        self["task_id"] = task_id

    def __getattr__(self, attr_name):
        if attr_name in self:
            return self[attr_name]
        raise super().__getattribute__(attr_name)

    def __setattr__(self, attr_name, value):
        self[attr_name] = value
