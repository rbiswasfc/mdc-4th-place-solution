from torch.utils.data import Dataset
from transformers import AutoTokenizer

SYS_PROMPT = """You are an expert at analyzing research data usage in academic papers. You will be shown a snippet of text likely containing one or more Accession IDs and/or DOIs.

Analyze the context to determine how a given Accession ID and/or DOI is being used:

A) PRIMARY - The authors generated this data for their current study
B) SECONDARY - The authors are using existing data from other sources
C) NONE - This is not a research data citation"""

USER_TEMPLATE = """# Context:\n\n{context}

# Focus DOI/Accession ID: **{dataset_id}**

# Task: Classify {dataset_id} as:
A) PRIMARY - Dataset created by these authors for this paper
B) SECONDARY - Dataset from previous work being reused
C) NONE - Not a dataset citation

Respond with only one letter: A, B, or C."""


def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.backbone_path,
        use_fast=cfg.model.tokenizer.use_fast,
        add_eos_token=False,
        truncation_side=cfg.model.tokenizer.truncation_side,
    )

    tokenizer.padding_side = "left"  # use left padding

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eod_id is not None:
            tokenizer.pad_token = tokenizer.eod
            tokenizer.pad_token_id = tokenizer.eod_id
            tokenizer.bos_token = tokenizer.im_start
            tokenizer.bos_token_id = tokenizer.im_start_id
            tokenizer.eos_token = tokenizer.im_end
            tokenizer.eos_token_id = tokenizer.im_end_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


class MDCDataset(Dataset):
    """
    Dataset class for Make Data Count - Finding Data References (MDC) context classification task
    """

    def __init__(self, cfg, df):
        self.cfg = cfg
        self.tokenizer = get_tokenizer(cfg)
        self.df = df
        self.label_map = {"Primary": 0, "Secondary": 1, "NA": 2, "None": 2}

    def __len__(self):
        return len(self.df)

    def _tokenize_function(self, texts):
        tx = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            return_length=True,
            add_special_tokens=True,
        )
        return tx

    def _build_input(self, row):
        choices = [USER_TEMPLATE]

        texts = []

        for template in choices:
            user_message = template.format(context=row["context"], dataset_id=row["dataset_id"])
            conversation = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_message}]
            text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            texts.append(text)
        return texts

    def extract_metadata(self, row):
        meta_keys = ["citation_count", "first_citation", "last_citation", "cited_by", "detected_dois"]
        ret = ""
        for k in meta_keys:
            ret += f"{k}: {row[k]}\n"
        return ret

    def __getitem__(self, idx):
        input_texts = []
        labels = []

        data = self.df.iloc[idx].to_dict()
        formatted_texts = self._build_input(data)

        input_texts.extend(formatted_texts)
        if isinstance(data["type"], str):  # TODO: adapt for distillaton / dense cross-entropy
            label_idx = self.label_map[data["type"]]
            label = [0.0, 0.0, 0.0]
            label[label_idx] = 1.0
            labels.extend([label] * len(formatted_texts))
        else:
            assert len(data["type"]) == 3, f"type must be soft label of 3 elements: {data['type']}"
            labels.extend([data["type"]] * len(formatted_texts))

        tx = self._tokenize_function(input_texts)

        return dict(input_ids=tx["input_ids"], attention_mask=tx["attention_mask"], length=tx["length"], labels=labels)
