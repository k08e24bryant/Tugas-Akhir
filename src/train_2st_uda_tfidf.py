"""
2ST-UDA (Two-Stage Training based Unsupervised Domain Adaptation) - Algorithm 2
Paper: Adversarial Training Based Multi-Source Unsupervised Domain Adaptation for Sentiment Analysis

Two-stage approach:
1. Stage 1: Pre-train WS-UDA (shared/private extractors, classifier, discriminator)
2. Stage 2: Self-training with pseudo-labels to train target-specific extractor
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_loader_acl_tfidf import (
    load_source_corpus,
    load_target_unlabeled_corpus,
    load_target_test_corpus
)
from models import (
    FeatureExtractor,
    SentimentClassifier,
    DomainDiscriminator,
    GradientReversalFunction
)

# Set random seeds
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ====================================================================================
# HYPERPARAMETERS (from paper)
# ====================================================================================

# Architecture
NUM_FEATURES = 10000
HIDDEN_DIM = 100
NUM_CLASSES = 2

# Dataset
BASE_PATH = r"C:\ITS\SEMESTER 7\Pra-TA\Replicate\processed_acl"
SOURCE_DOMAINS = ['books', 'dvd', 'electronics']
TARGET_DOMAIN = 'kitchen'
NUM_SOURCE_DOMAINS = len(SOURCE_DOMAINS)
NUM_ALL_DOMAINS = NUM_SOURCE_DOMAINS + 1

# Stage 1: Pre-training (WS-UDA)
BATCH_SIZE = 32
NUM_EPOCHS_STAGE1 = 10
LEARNING_RATE_MAIN = 1e-4
LEARNING_RATE_D = 5e-5
N_CRITIC = 5
ALPHA = 1.0  # Sentiment loss
BETA = 1.0   # Adversarial loss
GAMMA = 0.5  # Private domain loss

# Stage 2: Self-training
CONFIDENCE_INIT = 0.98    # Initial confidence threshold
CONFIDENCE_DECAY = 0.02   # Threshold decay per iteration
STOP_THRESHOLD = 10       # Stop if pseudo-label increase < this
ITER_PER_CONFIDENCE = 5   # Training iterations per confidence level
NUM_EPOCHS_FINETUNE = 20  # Final fine-tuning epochs

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output paths
STAGE1_MODEL_PATH = "2st_uda_stage1_best.pt"
FINAL_MODEL_PATH = "2st_uda_best.pt"
METRICS_PATH = "metrics_2st_uda.json"
CONFUSION_MATRIX_PATH = "confusion_matrix_2st_uda.csv"


# ====================================================================================
# UTILITIES
# ====================================================================================

def create_data_loader(X, y=None, domain=None, batch_size=32, shuffle=True):
    """
    Create PyTorch DataLoader from numpy arrays.
    
    Args:
        X: Feature matrix
        y: Labels (optional, will create dummy if None)
        domain: Domain labels (optional, will create dummy if None)
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader
    """
    X_tensor = torch.FloatTensor(X)
    
    if y is not None:
        y_tensor = torch.LongTensor(y)
    else:
        y_tensor = torch.zeros(len(X), dtype=torch.long)
    
    if domain is not None:
        d_tensor = torch.LongTensor(domain)
    else:
        d_tensor = torch.zeros(len(X), dtype=torch.long)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, d_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_grl_lambda(epoch, num_epochs):
    """
    Compute lambda for Gradient Reversal Layer.
    Schedule: λ = 2/(1+exp(-10p)) - 1, where p is training progress.
    """
    p = epoch / num_epochs
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0


# ====================================================================================
# STAGE 1: PRE-TRAINING (WS-UDA)
# ====================================================================================

def stage1_pretrain_wsuda():
    """
    Stage 1: Pre-train WS-UDA model.
    Returns trained models and vectorizer for Stage 2.
    """
    print("\n" + "=" * 80)
    print("STAGE 1: PRE-TRAINING WS-UDA")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading data...")
    source_texts = []
    source_labels = []
    source_domain_labels = []
    
    for domain_idx, domain in enumerate(SOURCE_DOMAINS):
        texts, labels = load_source_corpus(BASE_PATH, [domain])
        source_texts.extend(texts)
        source_labels.extend(labels)
        source_domain_labels.extend([domain_idx] * len(texts))
    
    target_unlabeled_texts = load_target_unlabeled_corpus(BASE_PATH, TARGET_DOMAIN)
    target_test_texts, target_test_labels = load_target_test_corpus(BASE_PATH, TARGET_DOMAIN)
    
    print(f"Source: {len(source_texts)} samples from {SOURCE_DOMAINS}")
    print(f"Target unlabeled: {len(target_unlabeled_texts)} samples")
    print(f"Target test: {len(target_test_texts)} samples")
    
    # Build TF-IDF vectorizer
    print("\n[2/5] Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=NUM_FEATURES,
        lowercase=False,
        tokenizer=lambda s: s.strip().split(),
        token_pattern=None,
        dtype=float,
        norm="l2",
        sublinear_tf=True
    )
    
    # Fit on source + target unlabeled (WS-UDA approach)
    all_train_texts = source_texts + target_unlabeled_texts
    vectorizer.fit(all_train_texts)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Vectorize
    X_source = vectorizer.transform(source_texts).toarray()
    X_target_unlabeled = vectorizer.transform(target_unlabeled_texts).toarray()
    X_target_test = vectorizer.transform(target_test_texts).toarray()
    
    # Create data loaders
    source_loader = create_data_loader(
        X_source, source_labels, source_domain_labels, 
        batch_size=BATCH_SIZE, shuffle=True
    )
    target_unlabeled_loader = create_data_loader(
        X_target_unlabeled, 
        domain=[NUM_SOURCE_DOMAINS] * len(X_target_unlabeled),
        batch_size=BATCH_SIZE, shuffle=True
    )
    target_test_loader = create_data_loader(
        X_target_test, target_test_labels,
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    # Initialize models
    print("\n[3/5] Initializing models...")
    E_s = FeatureExtractor(input_dim=NUM_FEATURES, output_dim=HIDDEN_DIM).to(DEVICE)
    E_p_list = nn.ModuleList([
        FeatureExtractor(input_dim=NUM_FEATURES, output_dim=HIDDEN_DIM).to(DEVICE)
        for _ in range(NUM_SOURCE_DOMAINS)
    ])
    C = SentimentClassifier(input_dim=200, output_dim=NUM_CLASSES).to(DEVICE)
    D = DomainDiscriminator(input_dim=HIDDEN_DIM, output_dim=NUM_ALL_DOMAINS).to(DEVICE)
    
    # Optimizers
    optimizer_main = optim.Adam(
        list(E_s.parameters()) + list(E_p_list.parameters()) + list(C.parameters()),
        lr=LEARNING_RATE_MAIN
    )
    optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE_D)
    
    # Loss functions
    criterion_sentiment = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    
    # Training
    print("\n[4/5] Training WS-UDA...")
    print("-" * 80)
    
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS_STAGE1):
        E_s.train()
        for E_p in E_p_list:
            E_p.train()
        C.train()
        D.train()
        
        # Compute GRL lambda
        lambda_grl = compute_grl_lambda(epoch, NUM_EPOCHS_STAGE1)
        
        total_loss_main = 0.0
        total_loss_D = 0.0
        num_batches = 0
        
        source_iter = iter(source_loader)
        target_iter = iter(target_unlabeled_loader)
        
        max_batches = max(len(source_loader), len(target_unlabeled_loader))
        
        for batch_idx in range(max_batches):
            # Get batches
            try:
                x_src, y_src, d_src = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                x_src, y_src, d_src = next(source_iter)
            
            try:
                x_tgt, _, d_tgt = next(target_iter)
            except StopIteration:
                target_iter = iter(target_unlabeled_loader)
                x_tgt, _, d_tgt = next(target_iter)
            
            x_src = x_src.to(DEVICE)
            y_src = y_src.to(DEVICE)
            d_src = d_src.to(DEVICE)
            x_tgt = x_tgt.to(DEVICE)
            d_tgt = d_tgt.to(DEVICE)
            
            # === Train Discriminator ===
            for _ in range(N_CRITIC):
                optimizer_D.zero_grad()
                
                loss_D = 0.0
                
                # Discriminate shared features
                with torch.no_grad():
                    z_s_src = E_s(x_src)
                    z_s_tgt = E_s(x_tgt)
                
                loss_D += criterion_domain(D(z_s_src), d_src)
                loss_D += criterion_domain(D(z_s_tgt), d_tgt)
                
                # Discriminate private features (source only)
                for d_idx in range(NUM_SOURCE_DOMAINS):
                    mask = (d_src == d_idx)
                    if mask.sum() > 0:
                        with torch.no_grad():
                            z_p = E_p_list[d_idx](x_src[mask])
                        loss_D += criterion_domain(D(z_p), d_src[mask])
                
                loss_D.backward()
                optimizer_D.step()
                total_loss_D += loss_D.item()
            
            # === Train Main Networks ===
            optimizer_main.zero_grad()
            
            loss_main = 0.0
            
            # Sentiment classification loss
            for d_idx in range(NUM_SOURCE_DOMAINS):
                mask = (d_src == d_idx)
                if mask.sum() > 0:
                    z_s = E_s(x_src[mask])
                    z_p = E_p_list[d_idx](x_src[mask])
                    z_concat = torch.cat([z_s, z_p], dim=1)
                    logits = C(z_concat)
                    loss_main += ALPHA * criterion_sentiment(logits, y_src[mask])
            
            # Adversarial loss for shared features
            x_all = torch.cat([x_src, x_tgt], dim=0)
            d_all = torch.cat([d_src, d_tgt], dim=0)
            
            z_s_all = E_s(x_all)
            z_s_all_grl = GradientReversalFunction.apply(z_s_all, lambda_grl)
            loss_main += BETA * criterion_domain(D(z_s_all_grl), d_all)
            
            # Private feature domain loss
            for d_idx in range(NUM_SOURCE_DOMAINS):
                mask = (d_src == d_idx)
                if mask.sum() > 0:
                    z_p = E_p_list[d_idx](x_src[mask])
                    z_p_grl = GradientReversalFunction.apply(z_p, -lambda_grl)
                    loss_main += GAMMA * criterion_domain(D(z_p_grl), d_src[mask])
            
            loss_main.backward()
            optimizer_main.step()
            
            total_loss_main += loss_main.item()
            num_batches += 1
        
        # Evaluate
        acc = evaluate_wsuda(E_s, E_p_list, C, D, target_test_loader)
        
        avg_loss_main = total_loss_main / num_batches
        avg_loss_D = total_loss_D / (num_batches * N_CRITIC)
        
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS_STAGE1} | λ={lambda_grl:.3f} | "
              f"Loss_main={avg_loss_main:.4f} | Loss_D={avg_loss_D:.4f} | "
              f"Acc={acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'E_s': E_s.state_dict(),
                'E_p_list': [E_p.state_dict() for E_p in E_p_list],
                'C': C.state_dict(),
                'D': D.state_dict(),
                'accuracy': best_acc
            }, STAGE1_MODEL_PATH)
            print(f"           → Best saved! (Acc: {best_acc:.2f}%)")
    
    print("-" * 80)
    print(f"[5/5] Stage 1 completed. Best accuracy: {best_acc:.2f}%\n")
    
    return E_s, E_p_list, C, D, vectorizer, X_target_unlabeled, X_target_test, target_test_labels


def evaluate_wsuda(E_s, E_p_list, C, D, test_loader):
    """
    Evaluate WS-UDA using weighted combination of source classifiers.
    """
    E_s.eval()
    for E_p in E_p_list:
        E_p.eval()
    C.eval()
    D.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            
            x = x.to(DEVICE)
            
            # Get shared features
            z_s = E_s(x)
            
            # Get domain weights from discriminator
            d_logits = D(z_s)
            d_probs = F.softmax(d_logits[:, :NUM_SOURCE_DOMAINS], dim=1)
            
            # Get predictions from each source classifier
            source_preds = []
            for d_idx in range(NUM_SOURCE_DOMAINS):
                z_p = E_p_list[d_idx](x)
                z_concat = torch.cat([z_s, z_p], dim=1)
                logits = C(z_concat)
                source_preds.append(F.softmax(logits, dim=1))
            
            # Stack and weight
            source_preds = torch.stack(source_preds, dim=1)  # [B, K, 2]
            d_probs_expanded = d_probs.unsqueeze(2)  # [B, K, 1]
            weighted = (source_preds * d_probs_expanded).sum(dim=1)  # [B, 2]
            
            preds = weighted.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
    
    acc = accuracy_score(all_labels, all_preds) * 100
    return acc


# ====================================================================================
# STAGE 2: SELF-TRAINING
# ====================================================================================

def stage2_selftraining(E_s, E_p_list, C, D, vectorizer, X_target_unlabeled, X_target_test, target_test_labels):
    """
    Stage 2: Train target-specific extractor using pseudo-labels.
    """
    print("=" * 80)
    print("STAGE 2: SELF-TRAINING WITH PSEUDO-LABELS")
    print("=" * 80)
    
    # Initialize target-specific extractor
    print("\n[1/4] Initializing target extractor...")
    E_t = FeatureExtractor(input_dim=NUM_FEATURES, output_dim=HIDDEN_DIM).to(DEVICE)
    optimizer_t = optim.Adam(E_t.parameters(), lr=LEARNING_RATE_MAIN)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare data
    X_target_tensor = torch.FloatTensor(X_target_unlabeled).to(DEVICE)
    X_test_tensor = torch.FloatTensor(X_target_test).to(DEVICE)
    y_test_tensor = torch.LongTensor(target_test_labels).to(DEVICE)
    
    # Track pseudo-labels
    pseudo_labeled = []  # List of (index, label)
    unlabeled_indices = set(range(len(X_target_unlabeled)))
    
    # Iterative pseudo-labeling
    print("\n[2/4] Generating pseudo-labels...")
    print("-" * 80)
    
    confidence = CONFIDENCE_INIT
    prev_count = 0
    
    while confidence >= 0.5:
        print(f"\nConfidence threshold: {confidence:.2f}")
        
        E_s.eval()
        for E_p in E_p_list:
            E_p.eval()
        C.eval()
        D.eval()
        E_t.eval()
        
        new_pseudo = []
        
        # Generate pseudo-labels for unlabeled data
        with torch.no_grad():
            for idx in unlabeled_indices:
                x = X_target_tensor[idx:idx+1]
                
                # Source prediction (weighted combination)
                z_s = E_s(x)
                d_logits = D(z_s)
                d_probs = F.softmax(d_logits[:, :NUM_SOURCE_DOMAINS], dim=1)
                
                source_preds = []
                for d_idx in range(NUM_SOURCE_DOMAINS):
                    z_p = E_p_list[d_idx](x)
                    z_concat = torch.cat([z_s, z_p], dim=1)
                    logits = C(z_concat)
                    source_preds.append(F.softmax(logits, dim=1))
                
                source_preds = torch.stack(source_preds, dim=1)
                y_source = (source_preds * d_probs.unsqueeze(2)).sum(dim=1)
                
                # Target prediction
                z_t = E_t(x)
                z_concat_t = torch.cat([z_s, z_t], dim=1)
                y_target = F.softmax(C(z_concat_t), dim=1)
                
                # Check agreement and confidence
                conf_src, pred_src = y_source.max(dim=1)
                conf_tgt, pred_tgt = y_target.max(dim=1)
                
                if (pred_src == pred_tgt and 
                    conf_src.item() > confidence and 
                    conf_tgt.item() > confidence):
                    new_pseudo.append((idx, pred_src.item()))
        
        # Add new pseudo-labels
        if len(new_pseudo) > 0:
            pseudo_labeled.extend(new_pseudo)
            for idx, _ in new_pseudo:
                unlabeled_indices.discard(idx)
            
            print(f"  Added: {len(new_pseudo)} | Total: {len(pseudo_labeled)}")
            
            # Train E_t with current pseudo-labels
            E_t.train()
            for iter_idx in range(ITER_PER_CONFIDENCE):
                total_loss = 0.0
                num_batches = 0
                
                indices = [idx for idx, _ in pseudo_labeled]
                labels = [label for _, label in pseudo_labeled]
                
                for i in range(0, len(indices), BATCH_SIZE):
                    batch_indices = indices[i:i+BATCH_SIZE]
                    batch_labels = labels[i:i+BATCH_SIZE]
                    
                    x_batch = X_target_tensor[batch_indices]
                    y_batch = torch.LongTensor(batch_labels).to(DEVICE)
                    
                    optimizer_t.zero_grad()
                    
                    with torch.no_grad():
                        z_s = E_s(x_batch)
                    
                    z_t = E_t(x_batch)
                    z_concat = torch.cat([z_s, z_t], dim=1)
                    logits = C(z_concat)
                    
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer_t.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                if (iter_idx + 1) % 5 == 0:
                    acc = evaluate_2st_uda(E_s, E_t, C, X_test_tensor, y_test_tensor)
                    avg_loss = total_loss / num_batches if num_batches > 0 else 0
                    print(f"    Iter {iter_idx+1}/{ITER_PER_CONFIDENCE}: loss={avg_loss:.4f}, acc={acc:.2f}%")
        else:
            print(f"  No new pseudo-labels")
        
        # Check stopping condition
        current_count = len(pseudo_labeled)
        increment = current_count - prev_count
        
        if increment < STOP_THRESHOLD:
            print(f"\n  → Stopping: increment ({increment}) < threshold ({STOP_THRESHOLD})")
            break
        
        prev_count = current_count
        confidence -= CONFIDENCE_DECAY
    
    print("-" * 80)
    print(f"\nTotal pseudo-labels generated: {len(pseudo_labeled)} / {len(X_target_unlabeled)}")
    
    # Fine-tuning
    print("\n[3/4] Fine-tuning with all pseudo-labels...")
    print("-" * 80)
    
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS_FINETUNE):
        E_t.train()
        total_loss = 0.0
        num_batches = 0
        
        indices = [idx for idx, _ in pseudo_labeled]
        labels = [label for _, label in pseudo_labeled]
        
        # Shuffle
        perm = np.random.permutation(len(indices))
        indices = [indices[i] for i in perm]
        labels = [labels[i] for i in perm]
        
        for i in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[i:i+BATCH_SIZE]
            batch_labels = labels[i:i+BATCH_SIZE]
            
            x_batch = X_target_tensor[batch_indices]
            y_batch = torch.LongTensor(batch_labels).to(DEVICE)
            
            optimizer_t.zero_grad()
            
            with torch.no_grad():
                z_s = E_s(x_batch)
            
            z_t = E_t(x_batch)
            z_concat = torch.cat([z_s, z_t], dim=1)
            logits = C(z_concat)
            
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer_t.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        acc = evaluate_2st_uda(E_s, E_t, C, X_test_tensor, y_test_tensor)
        avg_loss = total_loss / num_batches
        
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS_FINETUNE}: loss={avg_loss:.4f}, acc={acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'E_s': E_s.state_dict(),
                'E_t': E_t.state_dict(),
                'C': C.state_dict(),
                'accuracy': best_acc
            }, FINAL_MODEL_PATH)
            print(f"           → Best saved! (Acc: {best_acc:.2f}%)")
    
    print("-" * 80)
    print(f"[4/4] Stage 2 completed. Best accuracy: {best_acc:.2f}%\n")
    
    return E_s, E_t, C, best_acc


def evaluate_2st_uda(E_s, E_t, C, X_test, y_test):
    """
    Evaluate 2ST-UDA model.
    """
    E_s.eval()
    E_t.eval()
    C.eval()
    
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            x = X_test[i:i+BATCH_SIZE]
            
            z_s = E_s(x)
            z_t = E_t(x)
            z_concat = torch.cat([z_s, z_t], dim=1)
            
            logits = C(z_concat)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
    
    acc = accuracy_score(y_test.cpu().numpy(), all_preds) * 100
    return acc


# ====================================================================================
# FINAL EVALUATION
# ====================================================================================

def final_evaluation(E_s, E_t, C, X_test, y_test):
    """
    Comprehensive evaluation and save results.
    """
    print("=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    E_s.eval()
    E_t.eval()
    C.eval()
    
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            x = X_test[i:i+BATCH_SIZE]
            
            z_s = E_s(x)
            z_t = E_t(x)
            z_concat = torch.cat([z_s, z_t], dim=1)
            
            logits = C(z_concat)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
    
    y_true = y_test.cpu().numpy()
    y_pred = np.array(all_preds)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred) * 100
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'], output_dict=True)
    
    print(f"\nTest Accuracy: {acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save results
    np.savetxt(CONFUSION_MATRIX_PATH, cm, delimiter=',', fmt='%d')
    
    results = {
        'accuracy': float(acc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'model_path': FINAL_MODEL_PATH,
        'hyperparameters': {
            'stage1_epochs': NUM_EPOCHS_STAGE1,
            'stage2_confidence_init': CONFIDENCE_INIT,
            'stage2_confidence_decay': CONFIDENCE_DECAY,
            'stage2_finetune_epochs': NUM_EPOCHS_FINETUNE
        }
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved:")
    print(f"  - Metrics: {METRICS_PATH}")
    print(f"  - Confusion matrix: {CONFUSION_MATRIX_PATH}")
    print(f"  - Model: {FINAL_MODEL_PATH}")


# ====================================================================================
# MAIN
# ====================================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("2ST-UDA: Two-Stage Training based Unsupervised Domain Adaptation")
    print("Paper: Adversarial Training Based Multi-Source UDA for Sentiment Analysis")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Source domains: {SOURCE_DOMAINS}")
    print(f"Target domain: {TARGET_DOMAIN}")
    print("=" * 80)
    
    # Stage 1: Pre-training
    E_s, E_p_list, C, D, vectorizer, X_target_unlabeled, X_target_test, target_test_labels = stage1_pretrain_wsuda()
    
    # Stage 2: Self-training
    E_s, E_t, C, best_acc = stage2_selftraining(
        E_s, E_p_list, C, D, vectorizer,
        X_target_unlabeled, X_target_test, target_test_labels
    )
    
    # Final evaluation
    X_test_tensor = torch.FloatTensor(X_target_test).to(DEVICE)
    y_test_tensor = torch.LongTensor(target_test_labels).to(DEVICE)
    final_evaluation(E_s, E_t, C, X_test_tensor, y_test_tensor)
    
    print("\n" + "=" * 80)
    print(f"✓ TRAINING COMPLETED!")
    print(f"✓ Best accuracy: {best_acc:.2f}%")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
