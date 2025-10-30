"""
Supervised Baseline for Sentiment Classification
Train on labeled source domains, test on target domain
Uses TF-IDF features with MLP classifier
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report

from data_loader_acl_tfidf import (
    NUM_FEATURES,
    load_source_corpus,
    load_target_test_corpus,
    build_vectorizer,
    make_loader_from_vectors
)
from models import FeatureExtractor

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 10
HIDDEN_DIM = 100
NUM_CLASSES = 2
DROPOUT = 0.3
CLIP_GRAD_NORM = 5.0

# Paths
BASE_PATH = r"C:\ITS\SEMESTER 7\Pra-TA\Replicate\processed_acl"
TARGET_DOMAIN = "kitchen"
DOMAINS = ["books", "dvd", "electronics", "kitchen"]
SOURCE_DOMAINS = [d for d in DOMAINS if d != TARGET_DOMAIN]

# Output paths
VEC_PATH = "tfidf_vectorizer_baseline.pkl"
MODEL_PATH = "supervised_tfidf_best.pt"
METRICS_PATH = "metrics_supervised.json"
CONFUSION_MATRIX_PATH = "confusion_matrix_supervised.csv"


class SentimentClassifier(nn.Module):
    """
    Simple MLP classifier for sentiment classification.
    Takes features from extractor and predicts sentiment (positive/negative).
    """
    
    def __init__(self, input_dim: int = HIDDEN_DIM, hidden_dim: int = HIDDEN_DIM, 
                 output_dim: int = NUM_CLASSES, dropout: float = DROPOUT):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.network(x)


def train_epoch(model, classifier, dataloader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Args:
        model: Feature extractor
        classifier: Sentiment classifier
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device (CPU/CUDA)
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    classifier.train()
    total_loss = 0.0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        features = model(X_batch)
        logits = classifier(features)
        loss = criterion(logits, y_batch)
        
        # Backward pass
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(classifier.parameters()), 
            CLIP_GRAD_NORM
        )
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, classifier, dataloader, device):
    """
    Evaluate model on test set.
    
    Args:
        model: Feature extractor
        classifier: Sentiment classifier
        dataloader: Test data loader
        device: Device (CPU/CUDA)
        
    Returns:
        Tuple of (accuracy, true_labels, predictions)
    """
    model.eval()
    classifier.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            
            features = model(X_batch)
            logits = classifier(features)
            predictions = logits.argmax(dim=1).cpu().numpy()
            
            all_predictions.append(predictions)
            all_labels.append(y_batch.numpy())
    
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_predictions)
    accuracy = float((y_true == y_pred).mean() * 100.0)
    
    return accuracy, y_true, y_pred


def save_metrics(best_acc, y_true, y_pred, history):
    """
    Save evaluation metrics to JSON and confusion matrix to CSV.
    
    Args:
        best_acc: Best accuracy achieved
        y_true: True labels
        y_pred: Predicted labels
        history: Training history
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    np.savetxt(CONFUSION_MATRIX_PATH, cm, fmt="%d", delimiter=",")
    
    # Classification report
    report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True)
    
    # Compile metrics
    metrics = {
        "best_accuracy": best_acc,
        "confusion_matrix_path": CONFUSION_MATRIX_PATH,
        "classification_report": report,
        "training_history": history,
        "vectorizer_path": VEC_PATH,
        "model_path": MODEL_PATH,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT
        }
    }
    
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {METRICS_PATH}")
    print(f"✓ Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


def main():
    """Main training function."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("SUPERVISED BASELINE - SENTIMENT CLASSIFICATION")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Source domains: {', '.join(SOURCE_DOMAINS)}")
    print(f"Target domain: {TARGET_DOMAIN}")
    print(f"Hyperparameters: LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={NUM_EPOCHS}")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading data...")
    src_texts, src_labels = load_source_corpus(BASE_PATH, SOURCE_DOMAINS)
    tgt_texts, tgt_labels = load_target_test_corpus(BASE_PATH, TARGET_DOMAIN)
    
    # Build vectorizer (baseline mode: source only)
    print("\n[2/5] Building TF-IDF vectorizer...")
    vectorizer = build_vectorizer(
        mode="baseline",
        save_path=VEC_PATH,
        source_texts=src_texts,
        target_unlabeled_texts=None,
        max_features=NUM_FEATURES
    )
    
    # Vectorize data
    print("\n[3/5] Vectorizing data...")
    X_source = vectorizer.transform(src_texts)
    X_target = vectorizer.transform(tgt_texts)
    print(f"Source shape: {X_source.shape}")
    print(f"Target shape: {X_target.shape}")
    
    # Create data loaders
    source_loader = make_loader_from_vectors(
        X_source, src_labels, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True
    )
    test_loader = make_loader_from_vectors(
        X_target, tgt_labels, 
        batch_size=BATCH_SIZE * 2, 
        shuffle=False, 
        drop_last=False
    )
    
    # Initialize models
    print("\n[4/5] Initializing models...")
    extractor = FeatureExtractor(input_dim=NUM_FEATURES, output_dim=HIDDEN_DIM).to(device)
    classifier = SentimentClassifier(input_dim=HIDDEN_DIM, output_dim=NUM_CLASSES).to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(
        list(extractor.parameters()) + list(classifier.parameters()), 
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print("\n[5/5] Training...")
    print("-" * 80)
    
    best_acc = 0.0
    history = {"epoch": [], "train_loss": [], "test_accuracy": []}
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss = train_epoch(extractor, classifier, source_loader, optimizer, criterion, device)
        
        # Evaluate
        test_acc, y_true, y_pred = evaluate(extractor, classifier, test_loader, device)
        
        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_accuracy"].append(test_acc)
        
        # Print progress
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "extractor": extractor.state_dict(),
                "classifier": classifier.state_dict(),
                "accuracy": best_acc,
                "epoch": epoch
            }, MODEL_PATH)
            print(f"           → Best model saved! (Acc: {best_acc:.2f}%)")
    
    print("-" * 80)
    print(f"\n✓ Training completed!")
    print(f"✓ Best test accuracy: {best_acc:.2f}%")
    
    # Save metrics
    save_metrics(best_acc, y_true, y_pred, history)
    
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
