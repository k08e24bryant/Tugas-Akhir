"""
Data Loader for Amazon Review Dataset with TF-IDF Vectorization
Supports both baseline (source-only) and WS-UDA (source + target unlabeled) modes
"""

import os
import re
import json
import pickle
from typing import List, Tuple, Optional

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer

# Constants
NUM_FEATURES = 10000
DEFAULT_BATCH_SIZE = 32

# File names
FILE_POSITIVE = "positive.review"
FILE_NEGATIVE = "negative.review"
FILE_UNLABELED = "unlabeled.review"

# Regex for text cleaning
SPACE_PATTERN = re.compile(r"\s+")


def read_lines(file_path: str) -> List[str]:
    """
    Read lines from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of lines (stripped of newlines)
        
    Raises:
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    
    with open(file_path, "r", encoding="latin-1") as f:
        return [line.rstrip("\n") for line in f]


def load_source_corpus(base_path: str, domains: List[str]) -> Tuple[List[str], List[int]]:
    """
    Load labeled data from source domains.
    
    Args:
        base_path: Base directory containing domain folders
        domains: List of domain names (e.g., ['books', 'dvd', 'electronics'])
        
    Returns:
        Tuple of (texts, labels) where labels are 1 for positive, 0 for negative
    """
    all_texts = []
    all_labels = []
    
    for domain in domains:
        domain_path = os.path.join(base_path, domain)
        
        # Load positive reviews
        positive_lines = read_lines(os.path.join(domain_path, FILE_POSITIVE))
        all_texts.extend(positive_lines)
        all_labels.extend([1] * len(positive_lines))
        
        # Load negative reviews
        negative_lines = read_lines(os.path.join(domain_path, FILE_NEGATIVE))
        all_texts.extend(negative_lines)
        all_labels.extend([0] * len(negative_lines))
        
        print(f"[{domain}] Loaded: pos={len(positive_lines)}, neg={len(negative_lines)}, "
              f"total={len(positive_lines) + len(negative_lines)}")
    
    return all_texts, all_labels


def load_target_unlabeled_corpus(base_path: str, target_domain: str) -> List[str]:
    """
    Load unlabeled data from target domain.
    
    Args:
        base_path: Base directory containing domain folders
        target_domain: Target domain name (e.g., 'kitchen')
        
    Returns:
        List of unlabeled texts
    """
    target_path = os.path.join(base_path, target_domain)
    unlabeled_lines = read_lines(os.path.join(target_path, FILE_UNLABELED))
    
    print(f"[{target_domain}] Loaded unlabeled: {len(unlabeled_lines)}")
    return unlabeled_lines


def load_target_test_corpus(base_path: str, target_domain: str) -> Tuple[List[str], List[int]]:
    """
    Load test data from target domain.
    
    Args:
        base_path: Base directory containing domain folders
        target_domain: Target domain name (e.g., 'kitchen')
        
    Returns:
        Tuple of (texts, labels)
    """
    target_path = os.path.join(base_path, target_domain)
    
    # Load positive reviews
    positive_lines = read_lines(os.path.join(target_path, FILE_POSITIVE))
    # Load negative reviews
    negative_lines = read_lines(os.path.join(target_path, FILE_NEGATIVE))
    
    all_texts = positive_lines + negative_lines
    all_labels = [1] * len(positive_lines) + [0] * len(negative_lines)
    
    print(f"[{target_domain}] Loaded test: pos={len(positive_lines)}, neg={len(negative_lines)}, "
          f"total={len(all_texts)}")
    
    return all_texts, all_labels


def simple_tokenizer(text: str) -> List[str]:
    """
    Simple whitespace tokenizer.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    cleaned = SPACE_PATTERN.sub(" ", text.strip())
    return cleaned.split(" ")


def build_vectorizer(
    mode: str,
    save_path: str,
    source_texts: List[str],
    target_unlabeled_texts: Optional[List[str]] = None,
    max_features: int = NUM_FEATURES
) -> TfidfVectorizer:
    """
    Build and save TF-IDF vectorizer.
    
    Args:
        mode: Either 'baseline' (source only) or 'wsuda' (source + target unlabeled)
        save_path: Path to save the vectorizer
        source_texts: Source domain texts
        target_unlabeled_texts: Target unlabeled texts (required for 'wsuda' mode)
        max_features: Maximum number of features
        
    Returns:
        Fitted TfidfVectorizer
    """
    if mode not in ["baseline", "wsuda"]:
        raise ValueError(f"Mode must be 'baseline' or 'wsuda', got: {mode}")
    
    # Prepare training corpus
    if mode == "baseline":
        train_corpus = source_texts
    else:
        if target_unlabeled_texts is None:
            raise ValueError("target_unlabeled_texts required for 'wsuda' mode")
        train_corpus = source_texts + target_unlabeled_texts
    
    # Create vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        lowercase=False,
        tokenizer=simple_tokenizer,
        token_pattern=None,
        dtype=float,
        norm="l2",
        sublinear_tf=True
    )
    
    # Fit vectorizer
    vectorizer.fit(train_corpus)
    
    # Save vectorizer
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save metadata
    metadata = {
        "mode": mode,
        "vocab_size": len(vectorizer.vocabulary_),
        "max_features": max_features,
        "train_corpus_size": len(train_corpus)
    }
    with open(save_path + ".json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[vectorizer] Fitted in '{mode}' mode:")
    print(f"  - Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  - Training corpus: {len(train_corpus)} texts")
    print(f"  - Saved to: {save_path}")
    
    return vectorizer


def load_vectorizer(path: str) -> TfidfVectorizer:
    """
    Load a saved TF-IDF vectorizer.
    
    Args:
        path: Path to the saved vectorizer
        
    Returns:
        Loaded TfidfVectorizer
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def to_dense_tensor(sparse_matrix) -> torch.FloatTensor:
    """
    Convert sparse matrix to dense PyTorch tensor.
    
    Args:
        sparse_matrix: Sparse matrix (e.g., from TF-IDF transform)
        
    Returns:
        Dense FloatTensor
    """
    if hasattr(sparse_matrix, "toarray"):
        sparse_matrix = sparse_matrix.toarray()
    return torch.as_tensor(sparse_matrix, dtype=torch.float32)


def make_loader_from_vectors(
    X,
    y,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create DataLoader from feature vectors and labels.
    
    Args:
        X: Feature matrix (sparse or dense)
        y: Label array
        batch_size: Batch size
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        DataLoader
    """
    X_tensor = to_dense_tensor(X)
    y_tensor = torch.as_tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def make_unlabeled_loader_from_vectors(
    X,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create DataLoader for unlabeled data (with dummy labels).
    
    Args:
        X: Feature matrix (sparse or dense)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        DataLoader with dummy labels (-1)
    """
    X_tensor = to_dense_tensor(X)
    # Create dummy labels
    y_dummy = torch.full((X_tensor.size(0),), -1, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_dummy)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


if __name__ == "__main__":
    # Smoke test
    print("Testing data loader...")
    
    BASE_PATH = r"C:\ITS\SEMESTER 7\Pra-TA\Replicate\processed_acl"
    SOURCE_DOMAINS = ["books", "dvd", "electronics"]
    TARGET_DOMAIN = "kitchen"
    
    # Test loading
    src_texts, src_labels = load_source_corpus(BASE_PATH, SOURCE_DOMAINS)
    print(f"\nTotal source samples: {len(src_texts)}")
    
    tgt_unlabeled = load_target_unlabeled_corpus(BASE_PATH, TARGET_DOMAIN)
    print(f"Total target unlabeled: {len(tgt_unlabeled)}")
    
    tgt_test_texts, tgt_test_labels = load_target_test_corpus(BASE_PATH, TARGET_DOMAIN)
    print(f"Total target test: {len(tgt_test_texts)}")
    
    print("\n✓ Data loader test passed!")
