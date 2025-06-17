# src/train_han.py

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils import AuthorProfilingDataset, collate_fn
from han_model import HAN
from evaluate import eval_model


def train(args):
    # Fija semilla para reproducibilidad
    torch.manual_seed(args.seed)

    # Carga de datasets
    train_ds = AuthorProfilingDataset(
        split_dir=os.path.join(args.data_dir, f"es_train_{args.label}"),
        label_name=args.label,
        embedding_path=args.embedding_path
    )
    val_ds = AuthorProfilingDataset(
        split_dir=os.path.join(args.data_dir, f"es_val_{args.label}"),
        label_name=args.label,
        embedding_path=args.embedding_path
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Instanciar modelo HAN
    model = HAN(
        emb_mat=train_ds.emb_mat,
        num_classes=len(train_ds.dict_labels),
        attention=True
    ).to(args.device)

    # Configurar optimizador y función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for input_docs, labels, _ in train_loader:
            input_docs, labels = input_docs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            logits, _, _ = model(input_docs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            total_loss += loss.item()

        # Evaluación
        train_loss, train_acc, *_ = eval_model(model, train_loader, criterion, args.device)
        val_loss, val_acc, *_ = eval_model(model, val_loader, criterion, args.device)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Guardar mejor modelo según accuracy en validación
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output_path)
            print("» Nuevo mejor modelo guardado")

    print(f"Mejor accuracy en validación: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento de HAN para Author Profiling"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/processed",
        help="Directorio base de splits es_train_x y es_val_x"
    )
    parser.add_argument(
        "--embedding-path", type=str, default="data/word2vec_col.txt",
        help="Ruta al archivo de embeddings Word2Vec"
    )
    parser.add_argument(
        "--label", type=str, choices=["country", "gender"],
        required=True, help="Etiqueta a predecir"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Tamaño de batch para entrenamiento"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate para el optimizador"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Número de épocas de entrenamiento"
    )
    parser.add_argument(
        "--clip-norm", type=float, default=5.0,
        help="Valor de recorte de gradiente"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla aleatoria"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Dispositivo para entrenamiento (cpu o cuda)"
    )
    parser.add_argument(
        "--output-path", type=str, default="models/best_han.pt",
        help="Ruta donde se guardará el mejor modelo"
    )

    args = parser.parse_args()
    train(args)
