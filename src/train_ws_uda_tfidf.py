"""
WS-UDA (Weighting Scheme based Unsupervised Domain Adaptation) - Algorithm 1
Paper: Adversarial Training Based Multi-Source Unsupervised Domain Adaptation for Sentiment Analysis

Multi-source domain adaptation using:
- Shared feature extractor (E_s): Domain-invariant features
- Private feature extractors (E_p): Domain-specific features  
- Domain discriminator (D): Domain classification + weighting
- Sentiment classifier (C): Sentiment prediction
- Gradient Reversal Layer (GRL): Adversarial training
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import cycle
from sklearn.metrics import confusion_matrix, classification_report

from data_loader_acl_tfidf import (
    NUM_FEATURES,
    load_source_corpus,
    load_target_unlabeled_corpus,
    load_target_test_corpus,
    build_vectorizer,
    make_loader_from_vectors,
    make_unlabeled_loader_from_vectors
)
from models import (
    FeatureExtractor, 
    SentimentClassifier, 
    DomainDiscriminator, 
    GradientReversalLayer
)

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Hyperparameters (from paper)
LEARNING_RATE_MAIN = 1e-4  # For E_s, E_p, C
LEARNING_RATE_D = 5e-5     # For discriminator D
BATCH_SIZE = 32
NUM_EPOCHS = 10
N_CRITIC = 5               # Train discriminator N times per main step
CLIP_GRAD_NORM = 5.0

# Loss weights (from paper)
ALPHA = 1.0  # Sentiment classification loss
BETA = 1.0   # Adversarial loss for shared features
GAMMA = 0.5  # Domain classification loss for private features

# Architecture
HIDDEN_DIM = 100
NUM_CLASSES = 2

# Dataset configuration
BASE_PATH = r"C:\ITS\SEMESTER 7\Pra-TA\Replicate\processed_acl"
TARGET_DOMAIN = "kitchen"
DOMAINS = ["books", "dvd", "electronics", "kitchen"]
SOURCE_DOMAINS = [d for d in DOMAINS if d != TARGET_DOMAIN]
NUM_SOURCE_DOMAINS = len(SOURCE_DOMAINS)
NUM_ALL_DOMAINS = NUM_SOURCE_DOMAINS + 1  # Sources + target

# Output paths
VECTORIZER_PATH = "tfidf_vectorizer_wsuda.pkl"
MODEL_PATH = "wsuda_tfidf_best.pt"
METRICS_PATH = "metrics_wsuda.json"
CONFUSION_MATRIX_PATH = "confusion_matrix_wsuda.csv"
DOMAIN_WEIGHTS_PATH = "domain_weights_wsuda.csv"
DOMAIN_WEIGHTS_LABELS_PATH = "domain_weights_wsuda_labels.txt"


def compute_lambda_grl(epoch, step, num_epochs, steps_per_epoch):
    """
    Compute lambda for Gradient Reversal Layer using schedule from paper.
    Lambda increases from 0 to 1 following: λ_p = 2/(1+exp(-10p)) - 1
    where p is the training progress (0 to 1).
    
    Args:
        epoch: Current epoch (0-indexed)
        step: Current step within epoch
        num_epochs: Total number of epochs
        steps_per_epoch: Number of steps per epoch
        
    Returns:
        Lambda value for GRL
    """
    total_steps = num_epochs * steps_per_epoch
    current_step = epoch * steps_per_epoch + step
    p = current_step / total_steps
    
    lambda_val = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
    
    # Warm-up: reduce lambda in first 2 epochs
    if epoch < 2:
        lambda_val *= 0.1
    
    return lambda_val


def train_discriminator(
    E_s, E_p_list, D, 
    source_batch, target_batch, 
    optimizer_D, criterion_domain, 
    device
):
    """
    Train discriminator to classify domain from features.
    
    Args:
        E_s: Shared feature extractor
        E_p_list: List of private feature extractors
        D: Domain discriminator
        source_batch: (X, y) from source domains
        target_batch: (X, y) from target domain
        optimizer_D: Discriminator optimizer
        criterion_domain: Domain classification loss
        device: Device (CPU/CUDA)
        
    Returns:
        Discriminator loss value
    """
    X_source, _ = source_batch
    X_target, _ = target_batch
    
    X_source = X_source.to(device)
    X_target = X_target.to(device)
    
    batch_size = X_source.size(0)
    
    optimizer_D.zero_grad()
    
    # Get shared features (detached, so no gradients flow to E_s)
    with torch.no_grad():
        z_s_source = E_s(X_source)
        z_s_target = E_s(X_target)
    
    # Domain labels for source (split batch evenly across K source domains as proxy)
    domain_labels_source = torch.arange(0, NUM_SOURCE_DOMAINS, device=device)
    domain_labels_source = domain_labels_source.repeat(
        int(np.ceil(batch_size / NUM_SOURCE_DOMAINS))
    )[:batch_size]
    
    # Domain label for target
    domain_labels_target = torch.full(
        (X_target.size(0),), NUM_SOURCE_DOMAINS, dtype=torch.long, device=device
    )
    
    # Loss from shared features
    loss_D_shared = criterion_domain(D(z_s_source), domain_labels_source)
    loss_D_shared += criterion_domain(D(z_s_target), domain_labels_target)
    
    # Loss from private features (source only)
    loss_D_private = 0.0
    source_chunks = torch.chunk(X_source, NUM_SOURCE_DOMAINS, dim=0)
    
    for domain_idx in range(NUM_SOURCE_DOMAINS):
        if source_chunks[domain_idx].size(0) == 0:
            continue
        
        with torch.no_grad():
            z_p = E_p_list[domain_idx](source_chunks[domain_idx])
        
        domain_label = torch.full(
            (z_p.size(0),), domain_idx, dtype=torch.long, device=device
        )
        loss_D_private += criterion_domain(D(z_p), domain_label)
    
    # Total discriminator loss
    loss_D = loss_D_shared + loss_D_private
    loss_D.backward()
    optimizer_D.step()
    
    return loss_D.item()


def train_main_networks(
    E_s, E_p_list, C, D, grl,
    source_batch, target_batch,
    optimizer_main, 
    criterion_sentiment, criterion_domain,
    device
):
    """
    Train main networks (E_s, E_p, C) with three objectives:
    1. Sentiment classification (supervised, source only)
    2. Adversarial training for shared features (fool discriminator)
    3. Domain-specific training for private features (help discriminator)
    
    Args:
        E_s: Shared feature extractor
        E_p_list: List of private feature extractors
        C: Sentiment classifier
        D: Domain discriminator (frozen)
        grl: Gradient reversal layer
        source_batch: (X, y) from source domains
        target_batch: (X, y) from target domain
        optimizer_main: Main optimizer
        criterion_sentiment: Sentiment classification loss
        criterion_domain: Domain classification loss
        device: Device (CPU/CUDA)
        
    Returns:
        Main loss value
    """
    X_source, y_source = source_batch
    X_target, _ = target_batch
    
    X_source = X_source.to(device)
    y_source = y_source.to(device)
    X_target = X_target.to(device)
    
    batch_size = X_source.size(0)
    
    optimizer_main.zero_grad()
    
    # Split source batch across domains (proxy approach)
    source_chunks_X = torch.chunk(X_source, NUM_SOURCE_DOMAINS, dim=0)
    source_chunks_y = torch.chunk(y_source, NUM_SOURCE_DOMAINS, dim=0)
    
    # === Loss 1: Sentiment Classification ===
    loss_sentiment = 0.0
    num_active_domains = 0
    
    for domain_idx in range(NUM_SOURCE_DOMAINS):
        if source_chunks_X[domain_idx].size(0) == 0:
            continue
        
        X_domain = source_chunks_X[domain_idx]
        y_domain = source_chunks_y[domain_idx]
        
        # Extract features
        z_s = E_s(X_domain)
        z_p = E_p_list[domain_idx](X_domain)
        z_concat = torch.cat([z_s, z_p], dim=1)
        
        # Classify sentiment
        logits = C(z_concat)
        loss_sentiment += criterion_sentiment(logits, y_domain)
        num_active_domains += 1
    
    if num_active_domains > 0:
        loss_sentiment = loss_sentiment / num_active_domains
    
    # === Loss 2: Adversarial Training for Shared Features ===
    # Goal: E_s should produce domain-invariant features
    # Method: Apply GRL to reverse gradients from discriminator
    
    # Source domains
    z_s_source = E_s(X_source)
    z_s_source_grl = grl(z_s_source)
    
    domain_labels_source = torch.arange(0, NUM_SOURCE_DOMAINS, device=device)
    domain_labels_source = domain_labels_source.repeat(
        int(np.ceil(batch_size / NUM_SOURCE_DOMAINS))
    )[:batch_size]
    
    loss_adv_source = criterion_domain(D(z_s_source_grl), domain_labels_source)
    
    # Target domain
    z_s_target = E_s(X_target)
    z_s_target_grl = grl(z_s_target)
    
    domain_labels_target = torch.full(
        (X_target.size(0),), NUM_SOURCE_DOMAINS, dtype=torch.long, device=device
    )
    
    loss_adv_target = criterion_domain(D(z_s_target_grl), domain_labels_target)
    
    loss_adversarial = loss_adv_source + loss_adv_target
    
    # === Loss 3: Domain-Specific Training for Private Features ===
    # Goal: E_p should capture domain-specific features
    # Method: Train E_p to help discriminator classify correctly
    
    loss_private = 0.0
    num_active_private = 0
    
    for domain_idx in range(NUM_SOURCE_DOMAINS):
        if source_chunks_X[domain_idx].size(0) == 0:
            continue
        
        X_domain = source_chunks_X[domain_idx]
        z_p = E_p_list[domain_idx](X_domain)
        
        domain_label = torch.full(
            (z_p.size(0),), domain_idx, dtype=torch.long, device=device
        )
        
        loss_private += criterion_domain(D(z_p), domain_label)
        num_active_private += 1
    
    if num_active_private > 0:
        loss_private = loss_private / num_active_private
    
    # === Combine Losses ===
    loss_main = (ALPHA * loss_sentiment + 
                 BETA * loss_adversarial + 
                 GAMMA * loss_private)
    
    loss_main.backward()
    
    # Gradient clipping for stability
    nn.utils.clip_grad_norm_(
        list(E_s.parameters()) + list(E_p_list.parameters()) + list(C.parameters()),
        CLIP_GRAD_NORM
    )
    
    optimizer_main.step()
    
    return loss_main.item()


def evaluate(E_s, E_p_list, C, D, test_loader, device):
    """
    Evaluate model using weighted combination of source classifiers.
    Weighting scheme: Use discriminator to compute domain similarity weights.
    
    Args:
        E_s: Shared feature extractor
        E_p_list: List of private feature extractors
        C: Sentiment classifier
        D: Domain discriminator
        test_loader: Test data loader
        device: Device (CPU/CUDA)
        
    Returns:
        Tuple of (accuracy, true_labels, predictions, avg_domain_weights)
    """
    E_s.eval()
    for E_p in E_p_list:
        E_p.eval()
    C.eval()
    D.eval()
    
    all_predictions = []
    all_labels = []
    all_domain_weights = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            
            # Get shared features
            z_s = E_s(X_batch)
            
            # Compute domain weights from discriminator
            domain_logits = D(z_s)  # [batch, num_all_domains]
            source_logits = domain_logits[:, :NUM_SOURCE_DOMAINS]  # Only source domains
            domain_weights = F.softmax(source_logits, dim=1)  # [batch, num_sources]
            
            all_domain_weights.append(domain_weights.cpu().numpy())
            
            # Get predictions from each source classifier
            source_predictions = []
            for domain_idx in range(NUM_SOURCE_DOMAINS):
                z_p = E_p_list[domain_idx](X_batch)
                z_concat = torch.cat([z_s, z_p], dim=1)
                logits = C(z_concat)
                source_predictions.append(logits)
            
            # Stack predictions: [batch, num_sources, num_classes]
            source_predictions = torch.stack(source_predictions, dim=1)
            
            # Weighted combination
            domain_weights_expanded = domain_weights.unsqueeze(2)  # [batch, num_sources, 1]
            weighted_logits = (source_predictions * domain_weights_expanded).sum(dim=1)  # [batch, num_classes]
            
            # Final predictions
            predictions = weighted_logits.argmax(dim=1).cpu().numpy()
            
            all_predictions.append(predictions)
            all_labels.append(y_batch.numpy())
    
    # Compute metrics
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_predictions)
    accuracy = float((y_true == y_pred).mean() * 100.0)
    
    # Average domain weights
    avg_domain_weights = np.concatenate(all_domain_weights, axis=0).mean(axis=0)
    
    return accuracy, y_true, y_pred, avg_domain_weights


def save_results(best_acc, y_true, y_pred, history, avg_domain_weights):
    """
    Save training results, metrics, and domain weights.
    
    Args:
        best_acc: Best accuracy achieved
        y_true: True labels
        y_pred: Predicted labels
        history: Training history
        avg_domain_weights: Average domain weights
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    np.savetxt(CONFUSION_MATRIX_PATH, cm, fmt="%d", delimiter=",")
    
    # Classification report
    report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True)
    
    # Domain weights
    np.savetxt(DOMAIN_WEIGHTS_PATH, avg_domain_weights[None, :], delimiter=",")
    with open(DOMAIN_WEIGHTS_LABELS_PATH, "w", encoding="utf-8") as f:
        f.write(",".join(SOURCE_DOMAINS) + "\n")
    
    # Compile all metrics
    metrics = {
        "best_accuracy": best_acc,
        "confusion_matrix_path": CONFUSION_MATRIX_PATH,
        "classification_report": report,
        "training_history": history,
        "vectorizer_path": VECTORIZER_PATH,
        "model_path": MODEL_PATH,
        "domain_weights_path": DOMAIN_WEIGHTS_PATH,
        "source_domains": SOURCE_DOMAINS,
        "average_domain_weights": avg_domain_weights.tolist(),
        "hyperparameters": {
            "learning_rate_main": LEARNING_RATE_MAIN,
            "learning_rate_d": LEARNING_RATE_D,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "n_critic": N_CRITIC,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA
        }
    }
    
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {METRICS_PATH}")
    print(f"✓ Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    print(f"✓ Domain weights saved to: {DOMAIN_WEIGHTS_PATH}")


def main():
    """Main training function for WS-UDA."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("WS-UDA: Weighting Scheme based Unsupervised Domain Adaptation")
    print("Paper: Adversarial Training Based Multi-Source UDA")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Source domains: {', '.join(SOURCE_DOMAINS)}")
    print(f"Target domain: {TARGET_DOMAIN}")
    print(f"Hyperparameters:")
    print(f"  - Learning rates: Main={LEARNING_RATE_MAIN}, D={LEARNING_RATE_D}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - N_CRITIC: {N_CRITIC}")
    print(f"  - Loss weights: α={ALPHA}, β={BETA}, γ={GAMMA}")
    print("=" * 80)
    
    # Load data
    print("\n[1/6] Loading data...")
    src_texts, src_labels = load_source_corpus(BASE_PATH, SOURCE_DOMAINS)
    tgt_unlabeled_texts = load_target_unlabeled_corpus(BASE_PATH, TARGET_DOMAIN)
    tgt_test_texts, tgt_test_labels = load_target_test_corpus(BASE_PATH, TARGET_DOMAIN)
    
    # Build vectorizer (WS-UDA mode: include target unlabeled)
    print("\n[2/6] Building TF-IDF vectorizer...")
    vectorizer = build_vectorizer(
        mode="wsuda",
        save_path=VECTORIZER_PATH,
        source_texts=src_texts,
        target_unlabeled_texts=tgt_unlabeled_texts,
        max_features=NUM_FEATURES
    )
    
    # Vectorize data
    print("\n[3/6] Vectorizing data...")
    X_source = vectorizer.transform(src_texts)
    X_target_unlabeled = vectorizer.transform(tgt_unlabeled_texts)
    X_target_test = vectorizer.transform(tgt_test_texts)
    
    print(f"Source shape: {X_source.shape}")
    print(f"Target unlabeled shape: {X_target_unlabeled.shape}")
    print(f"Target test shape: {X_target_test.shape}")
    
    # Create data loaders
    source_loader = make_loader_from_vectors(
        X_source, src_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    target_unlabeled_loader = make_unlabeled_loader_from_vectors(
        X_target_unlabeled,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    test_loader = make_loader_from_vectors(
        X_target_test, tgt_test_labels,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        drop_last=False
    )
    
    # Initialize models
    print("\n[4/6] Initializing models...")
    E_s = FeatureExtractor(input_dim=NUM_FEATURES, output_dim=HIDDEN_DIM).to(device)
    E_p_list = nn.ModuleList([
        FeatureExtractor(input_dim=NUM_FEATURES, output_dim=HIDDEN_DIM).to(device)
        for _ in range(NUM_SOURCE_DOMAINS)
    ])
    C = SentimentClassifier(input_dim=200, output_dim=NUM_CLASSES).to(device)  # 100+100
    D = DomainDiscriminator(input_dim=HIDDEN_DIM, output_dim=NUM_ALL_DOMAINS).to(device)
    grl = GradientReversalLayer().to(device)
    
    print(f"✓ E_s (shared): {NUM_FEATURES} → {HIDDEN_DIM}")
    print(f"✓ E_p (private): {NUM_SOURCE_DOMAINS} extractors, each {NUM_FEATURES} → {HIDDEN_DIM}")
    print(f"✓ C (classifier): {HIDDEN_DIM*2} → {NUM_CLASSES}")
    print(f"✓ D (discriminator): {HIDDEN_DIM} → {NUM_ALL_DOMAINS}")
    
    # Optimizers
    optimizer_main = optim.Adam(
        list(E_s.parameters()) + list(E_p_list.parameters()) + list(C.parameters()),
        lr=LEARNING_RATE_MAIN
    )
    optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE_D, weight_decay=1e-4)
    
    # Loss functions
    criterion_sentiment = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    
    # Training
    print("\n[5/6] Training...")
    print("-" * 80)
    
    best_acc = 0.0
    history = {"epoch": [], "loss_main": [], "loss_D": [], "accuracy": [], "lambda": []}
    
    steps_per_epoch = max(len(source_loader), len(target_unlabeled_loader))
    source_iterator = iter(cycle(source_loader))
    target_iterator = iter(cycle(target_unlabeled_loader))
    
    for epoch in range(NUM_EPOCHS):
        # Set models to training mode
        E_s.train()
        for E_p in E_p_list:
            E_p.train()
        C.train()
        D.train()
        
        epoch_loss_main = 0.0
        epoch_loss_D = 0.0
        
        for step in range(steps_per_epoch):
            # Compute lambda for GRL
            lambda_grl = compute_lambda_grl(epoch, step, NUM_EPOCHS, steps_per_epoch)
            grl.set_lambda(lambda_grl)
            
            # Get batches
            source_batch = next(source_iterator)
            target_batch = next(target_iterator)
            
            # === Train Discriminator ===
            # Freeze main networks
            for param in E_s.parameters():
                param.requires_grad = False
            for param in E_p_list.parameters():
                param.requires_grad = False
            for param in C.parameters():
                param.requires_grad = False
            for param in D.parameters():
                param.requires_grad = True
            
            for _ in range(N_CRITIC):
                loss_D = train_discriminator(
                    E_s, E_p_list, D,
                    source_batch, target_batch,
                    optimizer_D, criterion_domain,
                    device
                )
                epoch_loss_D += loss_D
            
            # === Train Main Networks ===
            # Freeze discriminator, unfreeze main networks
            for param in E_s.parameters():
                param.requires_grad = True
            for param in E_p_list.parameters():
                param.requires_grad = True
            for param in C.parameters():
                param.requires_grad = True
            for param in D.parameters():
                param.requires_grad = False
            
            loss_main = train_main_networks(
                E_s, E_p_list, C, D, grl,
                source_batch, target_batch,
                optimizer_main,
                criterion_sentiment, criterion_domain,
                device
            )
            epoch_loss_main += loss_main
        
        # Compute average losses
        avg_loss_main = epoch_loss_main / steps_per_epoch
        avg_loss_D = epoch_loss_D / (steps_per_epoch * N_CRITIC)
        
        # Evaluate
        acc, y_true, y_pred, avg_domain_weights = evaluate(
            E_s, E_p_list, C, D, test_loader, device
        )
        
        # Record history
        history["epoch"].append(epoch + 1)
        history["loss_main"].append(avg_loss_main)
        history["loss_D"].append(avg_loss_D)
        history["accuracy"].append(acc)
        history["lambda"].append(lambda_grl)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | λ={lambda_grl:.3f} | "
              f"Loss_main={avg_loss_main:.4f} | Loss_D={avg_loss_D:.4f} | "
              f"Test Acc={acc:.2f}%")
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "E_s": E_s.state_dict(),
                "E_p_list": E_p_list.state_dict(),
                "C": C.state_dict(),
                "D": D.state_dict(),
                "accuracy": best_acc,
                "epoch": epoch + 1
            }, MODEL_PATH)
            print(f"           → Best model saved! (Acc: {best_acc:.2f}%)")
    
    print("-" * 80)
    
    # Final evaluation with best model
    print("\n[6/6] Final evaluation...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    E_s.load_state_dict(checkpoint["E_s"])
    E_p_list.load_state_dict(checkpoint["E_p_list"])
    C.load_state_dict(checkpoint["C"])
    D.load_state_dict(checkpoint["D"])
    
    acc, y_true, y_pred, avg_domain_weights = evaluate(
        E_s, E_p_list, C, D, test_loader, device
    )
    
    print(f"✓ Final test accuracy: {acc:.2f}%")
    print(f"✓ Domain weights: {avg_domain_weights}")
    
    # Save results
    save_results(best_acc, y_true, y_pred, history, avg_domain_weights)
    
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
