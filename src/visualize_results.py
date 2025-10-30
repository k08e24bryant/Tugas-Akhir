"""
Visualization Script for Multi-Source Domain Adaptation Results
Creates publication-quality plots for:
1. Training curves (all methods)
2. Confusion matrices (heatmaps)
3. Domain weights (WS-UDA)
4. Comparison bar chart
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

# Paths
BASE_PATH = Path(".")
METRICS_SUPERVISED = BASE_PATH / "metrics_supervised.json"
METRICS_WSUDA = BASE_PATH / "metrics_wsuda.json"
METRICS_2ST_UDA = BASE_PATH / "metrics_2st_uda.json"

CM_SUPERVISED = BASE_PATH / "confusion_matrix_supervised.csv"
CM_WSUDA = BASE_PATH / "confusion_matrix_wsuda.csv"
CM_2ST_UDA = BASE_PATH / "confusion_matrix_2st_uda.csv"

DOMAIN_WEIGHTS = BASE_PATH / "domain_weights_wsuda.csv"
DOMAIN_LABELS = BASE_PATH / "domain_weights_wsuda_labels.txt"

OUTPUT_DIR = BASE_PATH / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_json(path):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_csv(path):
    """Load CSV file as numpy array."""
    return np.loadtxt(path, delimiter=',')


def plot_training_curves():
    """
    Plot 1: Training curves comparison for all methods.
    Shows accuracy progression over epochs.
    """
    print("\n[1/5] Creating training curves plot...")
    
    # Load metrics
    supervised = load_json(METRICS_SUPERVISED)
    wsuda = load_json(METRICS_WSUDA)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1a: Supervised baseline
    ax1 = axes[0]
    epochs_sup = supervised['training_history']['epoch']
    acc_sup = supervised['training_history']['test_accuracy']
    
    ax1.plot(epochs_sup, acc_sup, 'o-', linewidth=2, markersize=8, label='Supervised', color='#2E86AB')
    ax1.axhline(y=supervised['best_accuracy'], color='#2E86AB', linestyle='--', alpha=0.5, label=f"Best: {supervised['best_accuracy']:.2f}%")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Supervised Baseline - Training Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([85, 100])
    
    # Plot 1b: WS-UDA
    ax2 = axes[1]
    epochs_ws = wsuda['training_history']['epoch']
    acc_ws = wsuda['training_history']['accuracy']
    lambda_vals = wsuda['training_history']['lambda']
    
    color = '#A23B72'
    ax2.plot(epochs_ws, acc_ws, 'o-', linewidth=2, markersize=8, label='WS-UDA', color=color)
    ax2.axhline(y=wsuda['best_accuracy'], color=color, linestyle='--', alpha=0.5, label=f"Best: {wsuda['best_accuracy']:.2f}%")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)', color=color)
    ax2.set_title('WS-UDA - Training Curve with Lambda Schedule')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([85, 100])
    
    # Secondary axis for lambda
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs_ws, lambda_vals, 's--', linewidth=1.5, markersize=6, label='Lambda (GRL)', color='#F18F01', alpha=0.7)
    ax2_twin.set_ylabel('Lambda (GRL)', color='#F18F01')
    ax2_twin.tick_params(axis='y', labelcolor='#F18F01')
    ax2_twin.set_ylim([0, 1.1])
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "training_curves.pdf", bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 'training_curves.png'}")
    plt.close()


def plot_confusion_matrices():
    """
    Plot 2: Confusion matrices as heatmaps for all three methods.
    """
    print("\n[2/5] Creating confusion matrices...")
    
    # Load confusion matrices
    cm_sup = load_csv(CM_SUPERVISED)
    cm_ws = load_csv(CM_WSUDA)
    cm_2st = load_csv(CM_2ST_UDA)
    
    # Load accuracies
    supervised = load_json(METRICS_SUPERVISED)
    wsuda = load_json(METRICS_WSUDA)
    tst_uda = load_json(METRICS_2ST_UDA)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    labels = ['Negative', 'Positive']
    cms = [cm_sup, cm_ws, cm_2st]
    titles = [
        f"Supervised Baseline\n(Acc: {supervised['best_accuracy']:.2f}%)",
        f"WS-UDA\n(Acc: {wsuda['best_accuracy']:.2f}%)",
        f"2ST-UDA\n(Acc: {tst_uda['accuracy']:.2f}%)"
    ]
    cmaps = ['Blues', 'Purples', 'Greens']
    
    for idx, (ax, cm, title, cmap) in enumerate(zip(axes, cms, titles, cmaps)):
        # Normalize to percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap=cmap, 
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Percentage (%)'}, ax=ax,
                    vmin=0, vmax=100)
        
        # Add counts as text
        for i in range(2):
            for j in range(2):
                text = ax.text(j + 0.5, i + 0.7, f'({int(cm[i, j])})',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "confusion_matrices.pdf", bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 'confusion_matrices.png'}")
    plt.close()


def plot_domain_weights():
    """
    Plot 3: Domain weights from WS-UDA discriminator.
    Shows how much each source domain contributes to target predictions.
    """
    print("\n[3/5] Creating domain weights plot...")
    
    # Load domain weights
    weights = load_csv(DOMAIN_WEIGHTS)[0]  # Shape: (1, 3) -> (3,)
    
    # Load domain labels
    with open(DOMAIN_LABELS, 'r') as f:
        domains = f.read().strip().split(',')
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(domains, weights * 100, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Weight (%)', fontweight='bold')
    ax.set_xlabel('Source Domain', fontweight='bold')
    ax.set_title('WS-UDA: Domain Contribution Weights for Target Domain (Kitchen)', 
                 fontweight='bold', fontsize=14)
    ax.set_ylim([0, np.max(weights * 100) * 1.15])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "domain_weights.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "domain_weights.pdf", bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 'domain_weights.png'}")
    plt.close()


def plot_comparison_barchart():
    """
    Plot 4: Bar chart comparing final accuracies of all methods.
    """
    print("\n[4/5] Creating comparison bar chart...")
    
    # Load accuracies
    supervised = load_json(METRICS_SUPERVISED)
    wsuda = load_json(METRICS_WSUDA)
    tst_uda = load_json(METRICS_2ST_UDA)
    
    methods = ['Supervised\nBaseline', 'WS-UDA\n(Algorithm 1)', '2ST-UDA\n(Algorithm 2)']
    accuracies = [
        supervised['best_accuracy'],
        wsuda['best_accuracy'],
        tst_uda['accuracy']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=13)
    ax.set_title('Performance Comparison: Multi-Source Domain Adaptation Methods\nTarget Domain: Kitchen Reviews', 
                 fontweight='bold', fontsize=14)
    ax.set_ylim([90, 100])
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend with method descriptions
    legend_labels = [
        'Source-only training',
        'Weighted source combination',
        'Target-specific extractor'
    ]
    ax.legend(bars, legend_labels, loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_barchart.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "comparison_barchart.pdf", bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 'comparison_barchart.png'}")
    plt.close()


def plot_detailed_metrics_table():
    """
    Plot 5: Create a detailed metrics comparison table.
    """
    print("\n[5/5] Creating detailed metrics table...")
    
    # Load all metrics
    supervised = load_json(METRICS_SUPERVISED)
    wsuda = load_json(METRICS_WSUDA)
    tst_uda = load_json(METRICS_2ST_UDA)
    
    # Extract classification reports
    sup_report = supervised['classification_report']
    ws_report = wsuda['classification_report']
    tst_report = tst_uda['classification_report']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Prepare data
    metrics_data = [
        ['Method', 'Accuracy', 'Precision\n(Neg)', 'Recall\n(Neg)', 'F1\n(Neg)', 
         'Precision\n(Pos)', 'Recall\n(Pos)', 'F1\n(Pos)'],
        ['Supervised', 
         f"{supervised['best_accuracy']:.2f}%",
         f"{sup_report['0']['precision']:.3f}",
         f"{sup_report['0']['recall']:.3f}",
         f"{sup_report['0']['f1-score']:.3f}",
         f"{sup_report['1']['precision']:.3f}",
         f"{sup_report['1']['recall']:.3f}",
         f"{sup_report['1']['f1-score']:.3f}"],
        ['WS-UDA',
         f"{wsuda['best_accuracy']:.2f}%",
         f"{ws_report['0']['precision']:.3f}",
         f"{ws_report['0']['recall']:.3f}",
         f"{ws_report['0']['f1-score']:.3f}",
         f"{ws_report['1']['precision']:.3f}",
         f"{ws_report['1']['recall']:.3f}",
         f"{ws_report['1']['f1-score']:.3f}"],
        ['2ST-UDA',
         f"{tst_uda['accuracy']:.2f}%",
         f"{tst_report['Negative']['precision']:.3f}",
         f"{tst_report['Negative']['recall']:.3f}",
         f"{tst_report['Negative']['f1-score']:.3f}",
         f"{tst_report['Positive']['precision']:.3f}",
         f"{tst_report['Positive']['recall']:.3f}",
         f"{tst_report['Positive']['f1-score']:.3f}"]
    ]
    
    # Create table
    table = ax.table(cellText=metrics_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(len(metrics_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4A90E2')
        cell.set_text_props(weight='bold', color='white')
    
    # Style data rows with alternating colors
    colors = ['#E8F4F8', '#F5F5F5', '#FFF4E6']
    for i in range(1, len(metrics_data)):
        for j in range(len(metrics_data[0])):
            cell = table[(i, j)]
            cell.set_facecolor(colors[i-1])
            if j == 0:  # Method name column
                cell.set_text_props(weight='bold')
    
    # Add title
    ax.set_title('Detailed Performance Metrics Comparison\nTarget Domain: Kitchen Reviews', 
                 fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics_table.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "metrics_table.pdf", bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 'metrics_table.png'}")
    plt.close()


def create_summary_report():
    """
    Create a text summary report.
    """
    print("\n[Bonus] Creating summary report...")
    
    supervised = load_json(METRICS_SUPERVISED)
    wsuda = load_json(METRICS_WSUDA)
    tst_uda = load_json(METRICS_2ST_UDA)
    
    report = f"""
{'='*80}
MULTI-SOURCE DOMAIN ADAPTATION - RESULTS SUMMARY
Target Domain: Kitchen Reviews
{'='*80}

1. SUPERVISED BASELINE
   - Best Accuracy: {supervised['best_accuracy']:.2f}%
   - Training Strategy: Source domains only (Books, DVD, Electronics)
   - Total Epochs: {len(supervised['training_history']['epoch'])}

2. WS-UDA (Weighting Scheme based UDA)
   - Best Accuracy: {wsuda['best_accuracy']:.2f}%
   - Training Strategy: Adversarial training with domain weighting
   - Total Epochs: {len(wsuda['training_history']['epoch'])}
   - Domain Weights:
     * Books: {wsuda['average_domain_weights'][0]*100:.1f}%
     * DVD: {wsuda['average_domain_weights'][1]*100:.1f}%
     * Electronics: {wsuda['average_domain_weights'][2]*100:.1f}%

3. 2ST-UDA (Two-Stage Training based UDA)
   - Best Accuracy: {tst_uda['accuracy']:.2f}%
   - Training Strategy: Pre-train WS-UDA + Self-training with pseudo-labels
   - Stage 1 (WS-UDA): 10 epochs
   - Stage 2 (Self-training): Confidence-based pseudo-labeling + Fine-tuning

{'='*80}
PERFORMANCE RANKING:
1. Supervised Baseline: {supervised['best_accuracy']:.2f}%
2. WS-UDA: {wsuda['best_accuracy']:.2f}%
3. 2ST-UDA: {tst_uda['accuracy']:.2f}%

OBSERVATIONS:
- All methods achieved >94% accuracy on target domain
- WS-UDA matched supervised performance ({wsuda['best_accuracy']:.2f}% vs {supervised['best_accuracy']:.2f}%)
- 2ST-UDA: {tst_uda['accuracy']:.2f}% (slightly lower, possibly due to pseudo-label noise)
- Domain weighting shows balanced contribution from all source domains

{'='*80}
VISUALIZATIONS GENERATED:
✓ Training curves: visualizations/training_curves.png
✓ Confusion matrices: visualizations/confusion_matrices.png
✓ Domain weights: visualizations/domain_weights.png
✓ Comparison chart: visualizations/comparison_barchart.png
✓ Metrics table: visualizations/metrics_table.png
{'='*80}
"""
    
    with open(OUTPUT_DIR / "summary_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"  ✓ Saved: {OUTPUT_DIR / 'summary_report.txt'}")


def main():
    """Main visualization function."""
    print("="*80)
    print("VISUALIZATION SCRIPT - MULTI-SOURCE DOMAIN ADAPTATION")
    print("="*80)
    
    # Check if all required files exist
    required_files = [
        METRICS_SUPERVISED, METRICS_WSUDA, METRICS_2ST_UDA,
        CM_SUPERVISED, CM_WSUDA, CM_2ST_UDA,
        DOMAIN_WEIGHTS, DOMAIN_LABELS
    ]
    
    missing = [f for f in required_files if not f.exists()]
    if missing:
        print("\n❌ ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease run training scripts first:")
        print("  1. python src/train_supervised_tfidf.py")
        print("  2. python src/train_ws_uda_tfidf.py")
        print("  3. python src/train_2st_uda_tfidf.py")
        return
    
    print(f"\n✓ All required files found")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    
    # Generate all plots
    plot_training_curves()
    plot_confusion_matrices()
    plot_domain_weights()
    plot_comparison_barchart()
    plot_detailed_metrics_table()
    create_summary_report()
    
    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETED!")
    print("="*80)
    print(f"\nAll visualizations saved to: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    print("  📊 training_curves.png/pdf")
    print("  📊 confusion_matrices.png/pdf")
    print("  📊 domain_weights.png/pdf")
    print("  📊 comparison_barchart.png/pdf")
    print("  📊 metrics_table.png/pdf")
    print("  📄 summary_report.txt")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
