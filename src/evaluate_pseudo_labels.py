"""
Evaluate Pseudo-Label Quality for 2ST-UDA
Compares generated pseudo-labels against true test labels to measure quality.
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from data_loader_acl_tfidf import (
    load_target_unlabeled_corpus,
    load_target_test_corpus,
    build_vectorizer
)
from models import FeatureExtractor, SentimentClassifier

# Constants
BASE_PATH = Path("processed_acl")
TARGET_DOMAIN = "kitchen"
VOCAB_SIZE = 10000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model paths
WSUDA_MODEL_PATH = "wsuda_tfidf_best.pt"
STAGE1_MODEL_PATH = "2st_uda_stage1_best.pt"


def load_models_stage1():
    """Load Stage 1 (WS-UDA) models."""
    print("\n[1/4] Loading Stage 1 models...")
    
    checkpoint = torch.load(STAGE1_MODEL_PATH, map_location=DEVICE)
    
    # Initialize models
    E_s = FeatureExtractor(input_dim=VOCAB_SIZE).to(DEVICE)
    # WS-UDA classifier expects 200-dim input (100 from E_s + 100 from E_p)
    C = SentimentClassifier(input_dim=200).to(DEVICE)
    
    # Load weights
    E_s.load_state_dict(checkpoint['E_s'])
    C.load_state_dict(checkpoint['C'])
    
    # Also load E_p networks for weighted combination
    E_p_list = [FeatureExtractor(input_dim=VOCAB_SIZE).to(DEVICE) for _ in range(3)]
    for i, E_p in enumerate(E_p_list):
        E_p.load_state_dict(checkpoint['E_p_list'][i])
        E_p.eval()
    
    # Load discriminator for weights
    from models import DomainDiscriminator
    D = DomainDiscriminator(output_dim=4).to(DEVICE)  # 3 source domains + 1 target (for adversarial)
    D.load_state_dict(checkpoint['D'])
    D.eval()
    
    E_s.eval()
    C.eval()
    
    print(f"  ✓ Loaded from: {STAGE1_MODEL_PATH}")
    return E_s, E_p_list, C, D


def generate_pseudo_labels(E_s, E_p_list, C, D, unlabeled_loader):
    """
    Generate pseudo-labels using Stage 1 (WS-UDA) model with weighted combination.
    Implements the same logic as in train_2st_uda_tfidf.py
    """
    print("\n[2/4] Generating pseudo-labels...")
    
    E_s.eval()
    C.eval()
    for E_p in E_p_list:
        E_p.eval()
    D.eval()
    
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in unlabeled_loader:
            X_batch = batch[0].to(DEVICE)  # Unpack tuple from DataLoader
            
            # Extract features from E_s
            f_s = E_s(X_batch)
            
            # Extract features from each E_p
            f_p_list = [E_p(X_batch) for E_p in E_p_list]
            
            # Get domain weights from discriminator
            domain_logits = D(f_s)
            # Use only first 3 logits (source domains), ignore the 4th (target)
            domain_logits_source = domain_logits[:, :3]
            domain_weights = torch.softmax(domain_logits_source, dim=1)  # Shape: (batch, 3)
            
            # Weighted combination of private features
            f_p_weighted = torch.zeros_like(f_p_list[0])
            for i, f_p in enumerate(f_p_list):
                # Expand weights to match feature dimensions
                weights = domain_weights[:, i].unsqueeze(1)  # Shape: (batch, 1)
                f_p_weighted += weights * f_p
            
            # Concatenate shared and weighted private features
            combined_features = torch.cat([f_s, f_p_weighted], dim=1)
            
            # Forward pass through classifier
            logits = C(combined_features)
            probs = torch.softmax(logits, dim=1)
            
            # Get predictions and confidence
            confidence, predictions = torch.max(probs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_confidences = np.array(all_confidences)
    
    print(f"  ✓ Generated predictions for {len(all_predictions)} samples")
    print(f"  ✓ Mean confidence: {all_confidences.mean():.4f}")
    print(f"  ✓ Min confidence: {all_confidences.min():.4f}")
    print(f"  ✓ Max confidence: {all_confidences.max():.4f}")
    
    return all_predictions, all_confidences


def apply_confidence_thresholding(predictions, confidences, true_labels, thresholds):
    """
    Apply different confidence thresholds and compute accuracy for each.
    """
    print("\n[3/4] Evaluating pseudo-labels at different confidence thresholds...")
    print("="*80)
    print(f"{'Threshold':<12} {'Selected':<12} {'%Selected':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")
    print("="*80)
    
    results = []
    
    for theta in thresholds:
        # Select high-confidence predictions
        mask = confidences >= theta
        selected_predictions = predictions[mask]
        selected_true = true_labels[mask]
        
        if len(selected_predictions) == 0:
            continue
        
        # Calculate metrics
        acc = accuracy_score(selected_true, selected_predictions)
        prec, rec, f1, _ = precision_recall_fscore_support(
            selected_true, selected_predictions, average='binary', zero_division=0
        )
        
        # Store results
        results.append({
            'threshold': theta,
            'n_selected': len(selected_predictions),
            'percent_selected': len(selected_predictions) / len(predictions) * 100,
            'accuracy': acc * 100,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        })
        
        print(f"{theta:<12.2f} {len(selected_predictions):<12} "
              f"{len(selected_predictions)/len(predictions)*100:<12.1f} "
              f"{acc*100:<12.2f} {prec:<12.3f} {rec:<12.3f}")
    
    print("="*80)
    
    return results


def detailed_analysis(predictions, confidences, true_labels):
    """
    Perform detailed analysis of pseudo-label quality.
    """
    print("\n[4/4] Detailed Analysis...")
    print("="*80)
    
    # Overall accuracy (all predictions)
    overall_acc = accuracy_score(true_labels, predictions)
    print(f"\n1. OVERALL PSEUDO-LABEL QUALITY:")
    print(f"   - Total samples: {len(predictions)}")
    print(f"   - Overall accuracy: {overall_acc*100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print(f"\n2. CONFUSION MATRIX:")
    print(f"   True\\Pred  Negative  Positive")
    print(f"   Negative   {cm[0,0]:<9} {cm[0,1]:<9}")
    print(f"   Positive   {cm[1,0]:<9} {cm[1,1]:<9}")
    
    # Error analysis
    errors = predictions != true_labels
    n_errors = errors.sum()
    error_rate = n_errors / len(predictions) * 100
    
    print(f"\n3. ERROR ANALYSIS:")
    print(f"   - Total errors: {n_errors}/{len(predictions)} ({error_rate:.2f}%)")
    print(f"   - False Positives (Neg→Pos): {cm[0,1]}")
    print(f"   - False Negatives (Pos→Neg): {cm[1,0]}")
    
    # Confidence analysis for correct vs incorrect predictions
    correct_mask = predictions == true_labels
    incorrect_mask = ~correct_mask
    
    if correct_mask.sum() > 0:
        correct_conf = confidences[correct_mask].mean()
        print(f"\n4. CONFIDENCE ANALYSIS:")
        print(f"   - Avg confidence (correct predictions): {correct_conf:.4f}")
    
    if incorrect_mask.sum() > 0:
        incorrect_conf = confidences[incorrect_mask].mean()
        print(f"   - Avg confidence (incorrect predictions): {incorrect_conf:.4f}")
        print(f"   - Confidence gap: {correct_conf - incorrect_conf:.4f}")
    
    # Class distribution
    pred_neg = (predictions == 0).sum()
    pred_pos = (predictions == 1).sum()
    true_neg = (true_labels == 0).sum()
    true_pos = (true_labels == 1).sum()
    
    print(f"\n5. CLASS DISTRIBUTION:")
    print(f"   - True: Negative={true_neg}, Positive={true_pos}")
    print(f"   - Predicted: Negative={pred_neg}, Positive={pred_pos}")
    print(f"   - Distribution shift: {abs(pred_neg-true_neg)} samples")
    
    # Per-class accuracy
    neg_mask = true_labels == 0
    pos_mask = true_labels == 1
    
    if neg_mask.sum() > 0:
        neg_acc = accuracy_score(true_labels[neg_mask], predictions[neg_mask])
        print(f"\n6. PER-CLASS ACCURACY:")
        print(f"   - Negative class: {neg_acc*100:.2f}%")
    
    if pos_mask.sum() > 0:
        pos_acc = accuracy_score(true_labels[pos_mask], predictions[pos_mask])
        print(f"   - Positive class: {pos_acc*100:.2f}%")
    
    print("="*80)
    
    return {
        'overall_accuracy': overall_acc * 100,
        'confusion_matrix': cm,
        'n_errors': n_errors,
        'error_rate': error_rate,
        'correct_confidence': correct_conf if correct_mask.sum() > 0 else 0,
        'incorrect_confidence': incorrect_conf if incorrect_mask.sum() > 0 else 0
    }


def main():
    """Main evaluation function."""
    print("="*80)
    print("PSEUDO-LABEL QUALITY EVALUATION")
    print("="*80)
    print(f"Device: {DEVICE}")
    
    # Load vectorizer
    print("\nLoading TF-IDF vectorizer...")
    vectorizer_path = "tfidf_vectorizer_wsuda.pkl"
    from joblib import load as joblib_load
    vectorizer = joblib_load(vectorizer_path)
    print(f"  ✓ Loaded from: {vectorizer_path}")
    
    # Load test data (true labels)
    print("\nLoading test data (true labels)...")
    test_texts, test_labels = load_target_test_corpus(BASE_PATH, TARGET_DOMAIN)
    test_labels = np.array(test_labels)
    print(f"  ✓ Loaded {len(test_labels)} test labels")
    
    # The unlabeled.review file contains MORE data than test
    # We need to use ONLY the unlabeled samples for evaluation
    # Since we don't have the exact mapping, we'll evaluate on test set instead
    print("\n⚠ Note: Using test set for pseudo-label generation instead")
    print("  (unlabeled corpus has different samples than test set)")
    
    # Re-vectorize test data
    X_test = vectorizer.transform(test_texts).toarray()
    
    # Create DataLoader for test data
    from torch.utils.data import TensorDataset, DataLoader
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"  ✓ Prepared {len(test_texts)} test samples for evaluation")
    
    # Load models and generate pseudo-labels
    E_s, E_p_list, C, D = load_models_stage1()
    predictions, confidences = generate_pseudo_labels(E_s, E_p_list, C, D, test_loader)
    
    # Evaluate at different confidence thresholds
    thresholds = [0.98, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    threshold_results = apply_confidence_thresholding(
        predictions, confidences, test_labels, thresholds
    )
    
    # Detailed analysis
    analysis_results = detailed_analysis(predictions, confidences, test_labels)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Overall pseudo-label accuracy: {analysis_results['overall_accuracy']:.2f}%")
    print(f"Total errors: {analysis_results['n_errors']} ({analysis_results['error_rate']:.2f}%)")
    print(f"Confidence gap (correct vs incorrect): {analysis_results['correct_confidence'] - analysis_results['incorrect_confidence']:.4f}")
    
    # Find optimal threshold (highest accuracy)
    if threshold_results:
        best_result = max(threshold_results, key=lambda x: x['accuracy'])
        print(f"\nBest threshold: {best_result['threshold']:.2f}")
        print(f"  - Selects: {best_result['n_selected']}/{len(predictions)} ({best_result['percent_selected']:.1f}%)")
        print(f"  - Accuracy: {best_result['accuracy']:.2f}%")
        print(f"  - F1-Score: {best_result['f1_score']:.3f}")
    
    print("="*80)
    print("\n✓ Evaluation completed!")
    
    # Save results
    output_file = "pseudo_label_evaluation.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PSEUDO-LABEL QUALITY EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Accuracy: {analysis_results['overall_accuracy']:.2f}%\n")
        f.write(f"Total Errors: {analysis_results['n_errors']} ({analysis_results['error_rate']:.2f}%)\n\n")
        f.write("Threshold Analysis:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Threshold':<12} {'Selected':<12} {'%Selected':<12} {'Accuracy':<12} {'F1-Score':<12}\n")
        f.write("-"*80 + "\n")
        for result in threshold_results:
            f.write(f"{result['threshold']:<12.2f} {result['n_selected']:<12} "
                   f"{result['percent_selected']:<12.1f} {result['accuracy']:<12.2f} "
                   f"{result['f1_score']:<12.3f}\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
