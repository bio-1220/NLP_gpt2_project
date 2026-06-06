#!/usr/bin/env python3
"""Load a Phase 1 emotion classifier and predict the emotion label of text.

Used by Pipeline A (E1/K1): predict the emotion of a poem's first 3 lines so it
can be prepended as a [emotion: X] / [감정: X] prompt prefix during both training
and generation.
"""

from __future__ import annotations

import torch

from experiments.data_bridge import EMOTION_ID_TO_LABEL


class EmotionPredictor:
    def __init__(self, ckpt_path, device):
        from train_emotion import GPT2EmotionClassifier, KoGPT2EmotionClassifier, build_tokenizer

        saved = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        cfg = saved["model_config"]
        self.lang = cfg.lang
        self.max_length = getattr(cfg, "max_length", 64)
        dropout = getattr(cfg, "hidden_dropout_prob", 0.3)
        self.model = GPT2EmotionClassifier(dropout) if self.lang == "en" else KoGPT2EmotionClassifier(dropout)
        self.model.load_state_dict(saved["model"])
        self.model.to(device).eval()
        self.device = device
        self.tokenizer, _ = build_tokenizer(self.lang)

    @torch.no_grad()
    def predict(self, texts, batch_size=32):
        labels = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            ).to(self.device)
            logits = self.model(enc["input_ids"], enc["attention_mask"])
            labels.extend(EMOTION_ID_TO_LABEL[p] for p in logits.argmax(-1).cpu().tolist())
        return labels
