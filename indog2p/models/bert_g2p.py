# indog2p/models/bert_g2p.py

import torch
import torch.nn as nn
from transformers import BertConfig, BertPreTrainedModel, BertModel, BertForMaskedLM

class BertG2PConfig(BertConfig):
    """
    Konfigurasi model BertG2P.
    Bisa inisialisasi dari dictionary config YAML project.
    """
    def __init__(
        self,
        vocab_size=32,
        hidden_size=128,
        num_attention_heads=8,
        num_hidden_layers=2,
        intermediate_size=128,
        max_position_embeddings=32,
        pad_token_id=0,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
            **kwargs
        )

class BertG2P(BertPreTrainedModel):
    """
    Model BERT untuk Grapheme-to-Phoneme Bahasa Indonesia (karakter-level).
    Kompatibel dengan loss Masked LM, cocok untuk masking karakter ambigu seperti 'e'.
    """
    config_class = BertG2PConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len], optional
            labels: [batch_size, seq_len], optional (untuk training/finetune)
        Returns:
            loss (jika labels diberikan)
            logits (selalu)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        logits = self.mlm_head(sequence_output)      # [batch, seq_len, vocab]
        
        loss = None
        if labels is not None:
            # CrossEntropyLoss, ignore_index=pad_token_id
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            # logits: [B, L, V], labels: [B, L] --> flatten
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    @classmethod
    def from_project_config(cls, config_dict, pad_token_id=0):
        """
        Instansiasi dari config dict (YAML project) + custom pad_token_id.
        """
        cfg = BertG2PConfig(
            vocab_size=config_dict["model"]["vocab_size"],
            hidden_size=config_dict["model"]["embedding_dim"],
            num_attention_heads=config_dict["model"]["num_heads"],
            num_hidden_layers=config_dict["model"]["num_layers"],
            intermediate_size=config_dict["model"]["feedforward_dim"],
            max_position_embeddings=config_dict["model"]["max_len"],
            pad_token_id=pad_token_id,
        )
        return cls(cfg)

    @classmethod
    def from_pretrained_hf(cls, pretrained_name, config_override=None):
        """
        Load model BERT dari HuggingFace, bisa di-finetune.
        """
        if config_override is not None:
            config = BertConfig.from_pretrained(pretrained_name, **config_override)
            return cls.from_pretrained(pretrained_name, config=config)
        return cls.from_pretrained(pretrained_name)

# ---- Untuk integrasi dengan script lain ----
def build_bert_g2p_from_yaml_config(yaml_path, vocab_path):
    """
    Helper untuk membangun model langsung dari file YAML dan vocab.
    """
    import yaml
    import json
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    with open(vocab_path, "r") as f:
        vocab_dict = json.load(f)
    pad_token_id = vocab_dict["token2id"]["[PAD]"]
    model = BertG2P.from_project_config(config_dict, pad_token_id=pad_token_id)
    return model

# ---- Logging detail model summary ----
def print_model_summary(model):
    """
    Print summary model, jumlah parameter, info layer.
    """
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[BertG2P] Total parameter: {n_params/1e6:.2f}M")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")

