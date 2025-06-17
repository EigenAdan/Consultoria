# src/han_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnModule(nn.Module):
    def __init__(self, hidden_dim, attention_dim, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attention_layer = nn.Linear(hidden_dim, attention_dim)
            self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, hidden_outputs):
        if self.use_attention:
            u = torch.tanh(self.attention_layer(hidden_outputs))
            alpha = F.softmax(self.context_vector(u), dim=1)
            v = torch.sum(alpha * hidden_outputs, dim=1)
            return v, alpha.squeeze(-1)
        else:
            v = torch.mean(hidden_outputs, dim=1)
            return v, None


class HAN(nn.Module):
    def __init__(self, emb_mat, word_gru_hidden_dim=50, word_attention_dim=50,
                 sent_gru_hidden_dim=50, sent_attention_dim=50, num_classes=7,
                 attention=True):
        super().__init__()
        self.attention = attention
        # Embeddings
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(emb_mat, dtype=torch.float32), freeze=False
        )
        # GRU word-level
        self.word_gru = nn.GRU(
            emb_mat.shape[1], word_gru_hidden_dim,
            bidirectional=True, batch_first=True
        )
        self.word_attention = AttnModule(
            word_gru_hidden_dim * 2, word_attention_dim,
            use_attention=self.attention
        )
        # GRU sentence-level
        self.sent_gru = nn.GRU(
            word_gru_hidden_dim * 2, sent_gru_hidden_dim,
            bidirectional=True, batch_first=True
        )
        self.sent_attention = AttnModule(
            sent_gru_hidden_dim * 2, sent_attention_dim,
            use_attention=self.attention
        )
        # Classifier
        self.classifier = nn.Linear(sent_gru_hidden_dim * 2, num_classes)

    def forward(self, input_docs, return_hidden: bool = False):
        batch_size, num_sents, num_words = input_docs.size()
        docs_flat = input_docs.view(-1, num_words)
        word_embeds = self.embedding(docs_flat)
        word_gru_out, _ = self.word_gru(word_embeds)
        sent_vectors, word_attention_scores = self.word_attention(word_gru_out)
        sent_vectors = sent_vectors.view(batch_size, num_sents, -1)
        sent_gru_out, _ = self.sent_gru(sent_vectors)
        doc_vectors, sent_attention_scores = self.sent_attention(sent_gru_out)
        output = self.classifier(doc_vectors)
        if word_attention_scores is not None:
            word_attention_scores = word_attention_scores.view(batch_size, num_sents, -1)
        if return_hidden:
            hidden_nonneg = F.relu(doc_vectors)
            return output, hidden_nonneg, word_attention_scores, sent_attention_scores
        return output, word_attention_scores, sent_attention_scores
