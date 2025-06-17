# src/evaluate.py

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

def eval_model(model, dataloader, criterion, device):
    """
    Evalúa el modelo en un DataLoader:
      - model: instancia de nn.Module
      - dataloader: DataLoader que devuelve (input_docs, labels, sentences)
      - criterion: función de pérdida (ej. CrossEntropyLoss)
      - device: 'cpu' o 'cuda'

    Retorna:
      mean_loss: pérdida promedio
      accuracy: exactitud global
      word_attn: lista de listas con scores de atención por palabra
      sent_attn: lista de listas con scores de atención por sentencia
      sentences: lista de listas de cadenas (los tuits)
      preds: lista de predicciones
      labels: lista de etiquetas verdaderas
    """
    model.eval()
    losses = []
    preds = torch.empty(0, dtype=torch.long)
    targets = torch.empty(0, dtype=torch.long)
    word_attn_list = []
    sent_attn_list = []
    all_sentences = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_docs, labels, sentences in tqdm(dataloader, desc="Evaluating"):
            # liberar caché GPU para evitar OOM (si usas cuda)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            input_docs = input_docs.to(device)
            labels = labels.to(device)

            output, word_attn, sent_attn = model(input_docs)
            loss = criterion(output, labels)
            losses.append(loss.item())

            # predicciones y ground truth
            probs = F.softmax(output, dim=1)
            batch_preds = probs.argmax(dim=1)
            preds = torch.cat([preds, batch_preds.cpu()])
            targets = torch.cat([targets, labels.cpu()])

            # guardar scores de atención y texto
            if word_attn is not None:
                word_attn_list.extend(word_attn.cpu().tolist())
                sent_attn_list.extend(sent_attn.cpu().tolist())
                all_sentences.extend(sentences)
                all_preds.extend(batch_preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

    preds = preds.numpy()
    targets = targets.numpy()
    accuracy = (preds == targets).mean()

    return (
        np.mean(losses),
        accuracy,
        word_attn_list,
        sent_attn_list,
        all_sentences,
        all_preds,
        all_labels
    )
