"""Train the Transformer recommender on real ratings.

Task: given the token sequence of 3 movies a user liked (rating >= 4.0),
each represented as "title + genre words", predict another movie the same
user liked. This mirrors exactly how model.get_recommendations() consumes
the model at inference time (app.py enriches TMDB titles with genres the
same way).

Usage:
    python python/train.py                     # data/ratings_data.csv (ml-latest-small)
    python python/train.py /path/to/ml-25m     # train on a full MovieLens release

Produces:
    models/transformer.pt      - checkpoint (weights + vocab + label classes)
    images/training_loss.png   - loss curve
"""
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from cleaning import simple_tokenizer
from data import build_genres, build_ratings, match_movies
from model import Transformer

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'transformer.pt')
LOSS_PLOT_PATH = os.path.join(BASE_DIR, 'images', 'training_loss.png')

SEED = 42
LIKE_THRESHOLD = 4.0
SAMPLES_PER_USER = 30
MAX_TRAIN_SAMPLES = 200_000
MAX_VAL_SAMPLES = 20_000
INPUT_MOVIES = 3
EPOCHS = 12
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 1024
LR = 1e-3
WEIGHT_DECAY = 0.01
HPARAMS = dict(model_dim=128, num_heads=4, num_layers=2, hidden_dim=512, dropout=0.2)


def load_data():
    """Return (movies_df, ratings_df, genres_by_id)."""
    movies_df = pd.read_csv(os.path.join(DATA_DIR, 'Top_1000_IMDb_movies_New_version.csv'))
    if len(sys.argv) > 1:
        ml_dir = sys.argv[1]
        print(f"Building training data from {ml_dir}")
        ml_to_local, ml_movies = match_movies(ml_dir)
        ratings_df = build_ratings(ml_dir, ml_to_local)
        genres_df = build_genres(ml_dir, ml_to_local, ml_movies)
    else:
        ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'ratings_data.csv'))
        genres_df = pd.read_csv(os.path.join(DATA_DIR, 'movie_genres.csv')).fillna('')
    genres_by_id = dict(zip(genres_df['movieId'], genres_df['genres']))
    return movies_df, ratings_df, genres_by_id


def build_vocab_and_labels(movies_df, genres_by_id):
    """Vocab over title + genre words; labels over movie names."""
    names = movies_df['Movie Name'].astype(str).tolist()
    texts = [
        f"{row['Movie Name']} {genres_by_id.get(row['Unnamed: 0'], '')}".strip()
        for _, row in movies_df.iterrows()
    ]
    vocab = set()
    for text in texts:
        vocab.update(simple_tokenizer(text))
    vocab = {word: idx + 1 for idx, word in enumerate(sorted(vocab))}  # 0 = padding
    label_encoder = LabelEncoder()
    label_encoder.fit(names)
    max_text_len = max(len(simple_tokenizer(t)) for t in texts)
    return vocab, label_encoder, max_text_len


def encode_texts(texts, vocab, max_seq_len):
    tokens = [vocab.get(w, 0) for t in texts for w in simple_tokenizer(t)]
    tokens = tokens[:max_seq_len]
    return tokens + [0] * (max_seq_len - len(tokens))


def build_samples(ratings_df, user_ids, id_to_text, id_to_class, vocab,
                  max_seq_len, max_samples, rng):
    """Sample (3 liked movies -> another liked movie) pairs for the given users."""
    liked_by_user = (
        ratings_df[(ratings_df['rating'] >= LIKE_THRESHOLD)
                   & (ratings_df['userId'].isin(user_ids))]
        .groupby('userId')['movieId'].agg(list)
    )
    inputs, targets = [], []
    user_order = list(liked_by_user.index)
    rng.shuffle(user_order)
    for user_id in user_order:
        liked = [mid for mid in liked_by_user[user_id] if mid in id_to_text]
        if len(liked) < INPUT_MOVIES + 1:
            continue
        for _ in range(min(SAMPLES_PER_USER, len(liked))):
            picks = rng.sample(liked, INPUT_MOVIES + 1)
            inputs.append(encode_texts([id_to_text[m] for m in picks[:INPUT_MOVIES]],
                                       vocab, max_seq_len))
            targets.append(id_to_class[picks[INPUT_MOVIES]])
        if len(inputs) >= max_samples:
            break
    return (torch.tensor(inputs[:max_samples], dtype=torch.long),
            torch.tensor(targets[:max_samples], dtype=torch.long))


def evaluate(model, inputs, targets, criterion, device):
    model.eval()
    total_loss, hits, n = 0.0, 0, len(inputs)
    with torch.no_grad():
        for start in range(0, n, EVAL_BATCH_SIZE):
            src = inputs[start:start + EVAL_BATCH_SIZE].to(device)
            tgt = targets[start:start + EVAL_BATCH_SIZE].to(device)
            logits = model(src, src)[:, -1, :]
            total_loss += criterion(logits, tgt).item() * len(src)
            top5 = torch.topk(logits, 5, dim=-1).indices
            hits += (top5 == tgt.unsqueeze(1)).any(dim=1).sum().item()
    return total_loss / n, hits / n


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    movies_df, ratings_df, genres_by_id = load_data()
    vocab, label_encoder, max_text_len = build_vocab_and_labels(movies_df, genres_by_id)
    max_seq_len = INPUT_MOVIES * max_text_len
    vocab_size = len(vocab) + 1  # +1 for padding index 0

    # Per-movie input text ("title + genre words") and target class
    class_index = {name: idx for idx, name in enumerate(label_encoder.classes_)}
    id_to_text, id_to_class = {}, {}
    for _, row in movies_df.iterrows():
        local_id = int(row['Unnamed: 0'])
        genres = genres_by_id.get(local_id, '')
        id_to_class[local_id] = class_index[str(row['Movie Name'])]
        id_to_text[local_id] = f"{row['Movie Name']} {genres}".strip()

    # Split by USER so validation users are never seen in training
    users = sorted(ratings_df['userId'].unique())
    rng.shuffle(users)
    split = int(len(users) * 0.9)
    train_users, val_users = set(users[:split]), set(users[split:])

    train_x, train_y = build_samples(ratings_df, train_users, id_to_text, id_to_class,
                                     vocab, max_seq_len, MAX_TRAIN_SAMPLES, rng)
    val_x, val_y = build_samples(ratings_df, val_users, id_to_text, id_to_class,
                                 vocab, max_seq_len, MAX_VAL_SAMPLES, rng)
    print(f"Vocab: {len(vocab)} words | classes: {len(label_encoder.classes_)} | "
          f"max_seq_len: {max_seq_len}")
    print(f"Samples: {len(train_x)} train ({len(train_users)} users) | "
          f"{len(val_x)} val ({len(val_users)} users)")

    # Popularity baseline: always recommend the 5 most-liked training targets
    top5_popular = torch.tensor(np.bincount(train_y.numpy()).argsort()[-5:].copy())
    pop_hit5 = (val_y.unsqueeze(1) == top5_popular).any(dim=1).float().mean().item()
    print(f"Popularity baseline hit@5: {pop_hit5:.1%}")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on {device}")

    model = Transformer(
        src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
        max_seq_len=max_seq_len, **HPARAMS,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    best_val_loss, best_state, best_epoch, best_hit5 = float('inf'), None, 0, 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        shuffled = torch.randperm(len(train_x))
        for start in range(0, len(shuffled), BATCH_SIZE):
            batch = shuffled[start:start + BATCH_SIZE]
            src = train_x[batch].to(device)
            tgt = train_y[batch].to(device)
            optimizer.zero_grad()
            logits = model(src, src)[:, -1, :]
            loss = criterion(logits, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        train_loss = epoch_loss / len(train_x)

        val_loss, hit5 = evaluate(model, val_x, val_y, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, best_hit5 = val_loss, epoch, hit5
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:2d}/{EPOCHS} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val hit@5 {hit5:.1%}")

    print(f"Keeping best checkpoint from epoch {best_epoch} "
          f"(val loss {best_val_loss:.4f}, hit@5 {best_hit5:.1%} "
          f"vs popularity baseline {pop_hit5:.1%})")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        'state_dict': best_state,
        'vocab': vocab,
        'label_classes': label_encoder.classes_.tolist(),
        'max_seq_len': max_seq_len,
        'vocab_size': vocab_size,
        'hparams': HPARAMS,
    }, MODEL_PATH)
    print(f"Checkpoint saved to {MODEL_PATH}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='train loss')
    plt.plot(range(1, EPOCHS + 1), val_losses, label='validation loss')
    plt.axvline(best_epoch, linestyle='--', alpha=0.5, label=f'best epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy loss')
    plt.title('Transformer recommender training')
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    print(f"Loss curve saved to {LOSS_PLOT_PATH}")


if __name__ == '__main__':
    main()
