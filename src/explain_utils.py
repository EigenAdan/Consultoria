import torch
import numpy as np
import nltk
from typing import List, Optional

class BatchTweetListWrapper:
    """
    Wrapper que recibe una lista de listas de tweets (crudos o ya preprocesados),
    y devuelve una matriz de probabilidades (o logits) por clase para cada autor.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 vocab: dict,
                 preprocess_fn: Optional[callable] = None,
                 device: str = "cpu",
                 fixed_num_words: int = 16,
                 max_num_sents: int = 100):
        self.model = model.eval().to(device)
        self.vocab = vocab
        self.pre   = preprocess_fn
        self.device = torch.device(device)
        self.W = fixed_num_words
        self.S = max_num_sents
        self.pad = vocab["[PAD]"]
        self.unk = vocab["[UNK]"]

    def __call__(self,
                 batch_tweets: List[List[str]],
                 return_probs: bool = True) -> np.ndarray:
        """
        batch_tweets: lista de listas de strings; 
                      cada sublista son los tweets de un autor/documento.
        return_probs: si True retorna probabilidades, si False logits.
        ---
        Devuelve un array shape (B, n_classes).
        """
        batch_ids = []
        for tweets in batch_tweets:
            # 1) Preprocesar cada tweet si procede
            proc_sents = [self.pre(t) if self.pre is not None else t for t in tweets]
            proc_sents = [s for s in proc_sents if s]

            # 2) Limitar a S oraciones (tweets)
            proc_sents = proc_sents[:self.S]

            # 3) Tokenizar + ids + trunc/pad a W tokens
            ids_list = []
            for sent in proc_sents:
                toks = nltk.word_tokenize(sent)
                ids = [self.vocab.get(w, self.unk) for w in toks][:self.W]
                ids += [self.pad] * (self.W - len(ids))
                ids_list.append(ids)

            # 4) Pad de oraciones vacías hasta S
            while len(ids_list) < self.S:
                ids_list.append([self.pad] * self.W)

            batch_ids.append(torch.tensor(ids_list, dtype=torch.long))

        # 5) Crear tensor (B, S, W)
        x = torch.stack(batch_ids, dim=0).to(self.device)

        # 6) Inferencia
        with torch.no_grad():
            logits, *_ = self.model(x)
            if return_probs:
                return torch.softmax(logits, dim=-1).cpu().numpy()
            else:
                return logits.cpu().numpy()




import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_distances
from typing import List, Callable

class BatchLimeExplainer:
    """
    LIME-like explainer for models that take a list of strings (tweets per author)
    and return a probability vector per input (via a wrapper).
    """
    def __init__(self, kernel_width: float = 25.0):
        self.kernel_width = kernel_width

    def explain_instance(self,
                         tweets: List[str],
                         wrapper_func: Callable[[List[List[str]]], np.ndarray],
                         num_features: int = 10,
                         num_samples: int = 5000):
        """
        tweets       : List[str] for one document (tweets of an author).
        wrapper_func : function that takes List[List[str]] and returns
                       ndarray shape (B, C) of probabilities.
                       For one instance, call wrapper_func([tweets])[0].
        num_features : number of top features (tokens) to return.
        num_samples  : perturbation samples to generate.
        
        Returns:
            explanation : List[(token, coef)] sorted by abs(coef) desc.
            r2_score    : weighted R^2 of the local linear model.
        """
        # 1) Tokenize across all tweets
        tokenized_tweets = [t.split() for t in tweets]
        # flatten and record boundaries
        boundaries = []
        flat_tokens = []
        for sent in tokenized_tweets:
            boundaries.append(len(sent))
            flat_tokens.extend(sent)
        n_tokens = len(flat_tokens)
        
        # 2) Generate binary mask samples
        np.random.seed(0)
        samples = np.random.randint(0, 2, size=(num_samples, n_tokens))
        samples[0, :] = 1  # keep original
        
        # 3) Reconstruct perturbed inputs (list of tweets)
        perturbed_inputs = []
        for mask in samples:
            docs = []
            idx = 0
            for bnd in boundaries:
                tokens_chunk = [flat_tokens[i] for i in range(idx, idx + bnd) if mask[i] == 1]
                docs.append(" ".join(tokens_chunk))
                idx += bnd
            perturbed_inputs.append(docs)
        
        # 4) Get probabilities for class of interest
        orig_probs = wrapper_func([tweets])[0]
        class_idx = np.argmax(orig_probs)
        y = wrapper_func(perturbed_inputs)[:, class_idx]
        
        # 5) Compute kernel weights (cosine distance)
        dists = cosine_distances(samples, np.ones((1, n_tokens))).ravel()
        kernel = np.exp(-(dists ** 2) / (self.kernel_width ** 2))
        
        # 6) Fit weighted linear model
        model = Ridge(alpha=1.0, fit_intercept=True)
        model.fit(samples, y, sample_weight=kernel)
        coefs = model.coef_
        
        # 7) Compute weighted R2
        y_pred = model.predict(samples)
        y_mean = np.average(y, weights=kernel)
        ss_res = np.sum(kernel * (y - y_pred) ** 2)
        ss_tot = np.sum(kernel * (y - y_mean) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        # 8) Select top features
        top_idx = np.argsort(np.abs(coefs))[-num_features:]
        explanation = [(flat_tokens[i], coefs[i]) for i in top_idx]
        explanation = sorted(explanation, key=lambda x: -abs(x[1]))
        
        return explanation, r2




