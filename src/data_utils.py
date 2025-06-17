import os
import re
import numpy as np
import nltk
import emoji
import xml.etree.ElementTree as ET
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
from torch.utils.data import Dataset

# Carga del tokenizador de spaCy (sin parser ni NER)
nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

# Patrones para limpieza de texto
URL_PAT     = re.compile(r'https?://\S+|www\.\S+')
USER_PAT    = re.compile(r'@\w+')
HASH_PAT    = re.compile(r'#(\w+)')
CHAR_FILTER = re.compile(r'[^0-9a-záéíóúñü_ ]+', flags=re.IGNORECASE)


def preprocess_tweet(text: str) -> str:
    """
    Limpia y tokeniza un tuit:
    - Convierte a minúsculas
    - Elimina URLs, menciones y el símbolo # de hashtags
    - Reemplaza emojis por espacio
    - Elimina caracteres especiales
    - Elimina stopwords
    """
    text = text.lower()
    text = URL_PAT.sub("", text)
    text = USER_PAT.sub("", text)
    text = HASH_PAT.sub(lambda m: m.group(1), text)
    text = emoji.replace_emoji(text, replace=" ")
    text = CHAR_FILTER.sub(" ", text)
    tokens = [tok.text for tok in nlp(text) if tok.text and tok.text not in STOP_WORDS]
    cleaned = " ".join(tokens)
    return re.sub(r"\s+", " ", cleaned).strip()


class AuthorProfilingDataset(Dataset):
    """
    Dataset para perfilado de autor en Twitter:
    - split_dir: carpeta con XMLs y truth.txt
    - label_name: 'country' o 'gender'
    - embedding_path: ruta a archivo .txt con embeddings Word2Vec (100-d)
    """
    def __init__(self, split_dir: str, label_name: str = "country", embedding_path: str = "word2vec_col.txt"):
        super().__init__()
        self.split_dir  = split_dir
        self.label_name = label_name.lower()
        if self.label_name not in {"country", "gender"}:
            raise ValueError("label_name debe ser 'country' o 'gender'.")
        self.label_col_ix = 2 if self.label_name == "country" else 1

        self._load_data()
        self.vocab, self.emb_mat = self._load_vocab_embeddings(embedding_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentences, label = self.data[idx]
        tokenised = [nltk.word_tokenize(sent) for sent in sentences]
        sent_ids  = [[self.vocab.get(tok, self.vocab.get("[UNK]", 1)) for tok in sent]
                     for sent in tokenised]
        return sent_ids, label, sentences

    def _load_data(self):
        truth_path = os.path.join(self.split_dir, "truth.txt")
        with open(truth_path, encoding="utf-8") as f:
            label_vals = sorted({ln.strip().split(":::")[self.label_col_ix] for ln in f})
        self.dict_labels     = {lab: i for i, lab in enumerate(label_vals)}
        self.dict_labels_inv = {i: lab for lab, i in self.dict_labels.items()}

        labels = {}
        with open(truth_path, encoding="utf-8") as f:
            for ln in f:
                parts = ln.strip().split(":::")
                labels[parts[0]] = self.dict_labels[parts[self.label_col_ix]]

        self.data = []
        for fname in os.listdir(self.split_dir):
            if not fname.endswith(".xml"): continue
            file_id = fname[:-4]
            label   = labels[file_id]
            tree = ET.parse(os.path.join(self.split_dir, fname))
            root = tree.getroot()
            cleaned_sentences = []
            for doc in root.findall('.//document'):
                raw = doc.text or ""
                proc = preprocess_tweet(raw)
                if proc:
                    cleaned_sentences.append(proc)
            self.data.append((cleaned_sentences, label))

    def _load_vocab_embeddings(self, embedding_path):
        vocab, vectors = {}, []
        with open(embedding_path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i == 0 and re.match(r"^\s*\d+\s+\d+\s*$", line): continue
                parts = line.rstrip().split()
                if len(parts) < 2: continue
                token, vec = parts[0], parts[1:]
                try:
                    vec = np.asarray(vec, dtype="float32")
                except ValueError:
                    continue
                vocab[token] = len(vectors) + 2
                vectors.append(vec)
        emb_dim = vectors[0].shape[0] if vectors else 100
        mean_vec = np.mean(np.vstack(vectors), axis=0) if vectors else np.zeros(emb_dim)
        vectors.insert(0, mean_vec)  # UNK
        vectors.insert(0, np.zeros(emb_dim))  # PAD
        vocab["[PAD]"], vocab["[UNK]"] = 0, 1
        return vocab, np.vstack(vectors)



import random, torch

# ------------------------------------------------------------
#  collate_fn – agrupa documentos en un batch fijo
# ------------------------------------------------------------
def collate_fn(batch):
    """
    batch: lista de tuplas (sent_ids, label, sent_texts)
    Devuelve:
        sentence_ids      – tensor  (B, max_num_sents, fixed_num_words)
        labels            – tensor  (B,)
        processed_sentences – lista  (B) con las frases en texto
    """
    zipped_batch = list(zip(*batch))

    fixed_num_words = 16    # tokens por oración
    max_num_sents   = 100   # oraciones por documento

    padded_batch_sentences = []
    processed_sentences    = []

    # --- procesar cada documento ---
    for sent_ids, sent_words in zip(zipped_batch[0], zipped_batch[2]):
        padded_sentences = []
        proc_sentences   = []

        # emparejar ids ↔ texto y recortar a max_num_sents
        sent_pairs = list(zip(sent_ids, sent_words))
        if len(sent_pairs) > max_num_sents:
            # si deseas reproducibilidad, fija random.seed(seed)
            sent_pairs = random.sample(sent_pairs, max_num_sents)

        for ids, words in sent_pairs:
            # truncar o padear cada oración
            if len(ids) > fixed_num_words:
                proc_sentences.append(words[:fixed_num_words])
                padded = torch.tensor(ids[:fixed_num_words], dtype=torch.long)
            else:
                proc_sentences.append(words)
                pad_len = fixed_num_words - len(ids)
                padded  = torch.cat([torch.tensor(ids, dtype=torch.long),
                                     torch.zeros(pad_len, dtype=torch.long)])
            padded_sentences.append(padded)

        # pad de oraciones vacías si el doc tiene < max_num_sents
        while len(padded_sentences) < max_num_sents:
            padded_sentences.append(torch.zeros(fixed_num_words, dtype=torch.long))

        padded_batch_sentences.append(torch.stack(padded_sentences))
        processed_sentences.append(proc_sentences)

    # stack final
    sentence_ids = torch.stack(padded_batch_sentences, dim=0)  # (B, S, W)
    labels       = torch.tensor(zipped_batch[1], dtype=torch.long)

    return sentence_ids, labels, processed_sentences

