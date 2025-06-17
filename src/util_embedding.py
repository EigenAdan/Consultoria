# ------------------------------------------------------------
#  util_embeddings.py  – carga sencillísima de Word2Vec 100-d
# ------------------------------------------------------------
import numpy as np

def load_word2vec_txt(path):
    """
    Lee un archivo .txt en formato Word2Vec/GloVe y devuelve:
        vocab  : dict  token → fila
        emb_mat: ndarray shape (vocab_size, dim)
    Descarta la cabecera "N dim" si existe.
    """
    vocab, vectors = {}, []

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i == 1 and len(line.split()) == 2 and line.split()[0].isdigit():
                # línea-cabecera; la ignoramos
                continue
            parts = line.rstrip().split()
            if len(parts) <= 2:          # línea vacía o corrupta
                continue
            token, vec = parts[0], parts[1:]
            vectors.append(np.asarray(vec, dtype="float32"))
            vocab[token] = len(vectors) - 1

    emb_mat = np.vstack(vectors)
    return vocab, emb_mat            # (vocab_size, dim)
