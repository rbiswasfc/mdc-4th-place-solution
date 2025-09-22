from torch.utils.data import Dataset
from transformers import AutoTokenizer

SYS_PROMPT = """You are an expert at identifying research data citations in scientific literature. You excel at identifying and cataloging data citations in academic papers.

Your task is to determine whether a given DOI or Accession ID represents a valid citation based on the provided text context. Focus on identifying genuine data citations, not just any mention of an identifier."""

USER_TEMPLATE_W_DOI = """Text snippet: {context}

Detected DOI in the whole article: {detected_dois}

Focus DOI/Accession ID: {dataset_id}

Is {dataset_id} a valid research data citation in this text snippet?

Respond with only Yes or No"""

USER_TEMPLATE_WO_DOI = """Text snippet: {context}

Focus DOI/Accession ID: {dataset_id}

Is {dataset_id} a valid research data citation in this text snippet?

Respond with only Yes or No"""


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

    def __init__(self, cfg, df, aug_df=None, seed=None):
        self.cfg = cfg
        self.df = df
        self.aug_df = aug_df
        self.tokenizer = get_tokenizer(cfg)
        self.label_map = {"Yes": 1, "No": 0}

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

    def _build_input(self, context, dataset_id, detected_dois):
        choices = [USER_TEMPLATE_WO_DOI, USER_TEMPLATE_W_DOI]
        texts = []

        if len(detected_dois.strip()) == 0:
            detected_dois = "N/A"

        for template in choices:
            user_message = template.format(context=context, dataset_id=dataset_id, detected_dois=detected_dois)
            conversation = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_message}]
            text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            texts.append(text)

        return texts

    def __getitem__(self, idx):
        input_texts = []
        labels = []

        data = self.df.iloc[idx].to_dict()
        dataset_id = data["dataset_id"]
        context = data["context"]
        detected_dois = data["detected_dois"]

        formatted_texts = self._build_input(context, dataset_id, detected_dois)
        input_texts.extend(formatted_texts)
        if isinstance(data["label"], str):
            labels.extend([self.label_map[data["label"]]] * len(formatted_texts))
        else:
            labels.extend([data["label"]] * len(formatted_texts))

        tx = self._tokenize_function(input_texts)

        return dict(input_ids=tx["input_ids"], attention_mask=tx["attention_mask"], length=tx["length"], labels=labels)
