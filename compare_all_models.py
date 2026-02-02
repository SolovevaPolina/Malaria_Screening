import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
dark2 = [
    "#1b9e77",
    "#66a61e",
    "#7570b3",
    "#e6ab02",
    "#d95f02",
    "#e7298a",
    "#a6761d",
    "#666666"
]

# Read data files
csv_path = Path(__file__).parent / "shufflenet_gridsearch_results.csv"
lrrfsvm_path = Path(__file__).parent / "LRRF_SVM_model_comparison_results.csv"

df_shuffle = pd.read_csv(csv_path)
df_lrrfsvm = pd.read_csv(lrrfsvm_path)

# Ensure train_size column exists in shufflenet results (for 30% default, 50% for new entry)
if 'train_size' not in df_shuffle.columns:
    train_sizes = [0.30] * (len(df_shuffle) - 1) + [0.50]
    df_shuffle.insert(0, 'train_size', train_sizes)

output_dir = Path(__file__).parent / "gridsearch_plots"
output_dir.mkdir(exist_ok=True)

# ============ EXTRACT KEY MODELS ============
# Best ShuffleNet by accuracy
best_acc_shuffle = df_shuffle.loc[df_shuffle['accuracy'].idxmax()]
# Best ShuffleNet by training time (fastest)
best_time_shuffle = df_shuffle.loc[df_shuffle['train_time_s'].idxmin()]

print("="*70)
print("BEST SHUFFLENET MODELS")
print("="*70)
print(f"\nBest by Accuracy:")
print(f"  LR={best_acc_shuffle['lr']}, BS={best_acc_shuffle['batch_size']}, Epochs={best_acc_shuffle['num_epochs']}")
print(f"  Accuracy: {best_acc_shuffle['accuracy']:.4f}")
print(f"  Precision: {best_acc_shuffle['precision_parasitic']:.4f}")
print(f"  Recall: {best_acc_shuffle['recall_parasitic']:.4f}")
print(f"  F1: {best_acc_shuffle['f1_macro']:.4f}")
print(f"  Train Time: {best_acc_shuffle['train_time_s']:.2f}s")

print(f"\nBest by Training Time (Fastest):")
print(f"  LR={best_time_shuffle['lr']}, BS={best_time_shuffle['batch_size']}, Epochs={best_time_shuffle['num_epochs']}")
print(f"  Accuracy: {best_time_shuffle['accuracy']:.4f}")
print(f"  Precision: {best_time_shuffle['precision_parasitic']:.4f}")
print(f"  Recall: {best_time_shuffle['recall_parasitic']:.4f}")
print(f"  F1: {best_time_shuffle['f1_macro']:.4f}")
print(f"  Train Time: {best_time_shuffle['train_time_s']:.2f}s")

print("\n" + "="*70)
print("LOGISTIC REGRESSION & RANDOM FOREST RESULTS")
print("="*70)
print(df_lrrfsvm.to_string())

# ============ CREATE COMPARISON PLOTS ============

fig = plt.figure(figsize=(10, 10))

# # 1. Accuracy comparison
# ax1 = plt.subplot(2, 3, 1)
# models_to_plot = ['Logistic Regression', 'Random Forest']
lr_data = df_lrrfsvm[df_lrrfsvm['model'] == 'Logistic Regression'].sort_values('train_size')
rf_data = df_lrrfsvm[df_lrrfsvm['model'] == 'Random Forest'].sort_values('train_size')
svm_data = df_lrrfsvm[df_lrrfsvm['model'] == 'SVM'].sort_values('train_size')

# ax1.plot(lr_data['Train Size'] * 100, lr_data['Precision'], marker='o', label='LR Precision', linewidth=2)
# ax1.plot(rf_data['Train Size'] * 100, rf_data['Precision'], marker='s', label='RF Precision', linewidth=2)
# # ax1.scatter(y=best_acc_shuffle['precision_parasitic'], color='r', label=f"SN Best-Acc: {best_acc_shuffle['precision_parasitic']:.4f}")
# # ax1.scatter(y=best_time_shuffle['precision_parasitic'], color='orange', label=f"SN Best-Time: {best_time_shuffle['precision_parasitic']:.4f}")
# ax1.scatter([30], [best_acc_shuffle['precision_parasitic']], color='r', s=100, label=f"SN Best-Acc: {best_acc_shuffle['precision_parasitic']:.4f}", zorder=5)
# ax1.scatter([30], [best_time_shuffle['precision_parasitic']], color='orange', s=100, label=f"SN Best-Time: {best_time_shuffle['precision_parasitic']:.4f}", zorder=5)

# ax1.set_xlabel('Training Size (%)', fontsize=11)
# ax1.set_ylabel('Precision', fontsize=11)
# ax1.set_title('Precision: LR/RF vs ShuffleNet', fontweight='bold')
# ax1.legend(fontsize=9)
# ax1.grid(True, alpha=0.3)

# # 2. Recall comparison
# ax2 = plt.subplot(2, 3, 2)
# ax2.plot(lr_data['Train Size'] * 100, lr_data['Recall'], marker='o', label='LR Recall', linewidth=2)
# ax2.plot(rf_data['Train Size'] * 100, rf_data['Recall'], marker='s', label='RF Recall', linewidth=2)
# # ax2.scatter(y=best_acc_shuffle['recall_parasitic'], color='r', label=f"SN Best-Acc: {best_acc_shuffle['recall_parasitic']:.4f}")
# # ax2.scatter(y=best_time_shuffle['recall_parasitic'], color='orange', label=f"SN Best-Time: {best_time_shuffle['recall_parasitic']:.4f}")
# ax2.scatter([30], [best_acc_shuffle['recall_parasitic']], color='r', s=100, label=f"SN Best-Acc: {best_acc_shuffle['recall_parasitic']:.4f}", zorder=5)
# ax2.scatter([30], [best_time_shuffle['recall_parasitic']], color='orange', s=100, label=f"SN Best-Time: {best_time_shuffle['recall_parasitic']:.4f}", zorder=5)

# ax2.set_xlabel('Training Size (%)', fontsize=11)
# ax2.set_ylabel('Recall', fontsize=11)
# ax2.set_title('Recall: LR/RF vs ShuffleNet', fontweight='bold')
# ax2.legend(fontsize=9)
# ax2.grid(True, alpha=0.3)

# 3. F1 Score comparison
ax3 = plt.subplot(2, 2, 1)
ax3.plot(lr_data['train_size'] * 100, lr_data['f1_score'], marker='o', label='LR F1', linewidth=2, color = dark2[0])
ax3.plot(rf_data['train_size'] * 100, rf_data['f1_score'], marker='s', label='RF F1', linewidth=2, color = dark2[1])
ax3.plot(svm_data['train_size'] * 100, svm_data['f1_score'], marker='+', markersize=10, markeredgewidth=2, label='SVM F1', linewidth=2, color = dark2[2])
# Add ShuffleNet at 30% and 50% training size
sn_30 = df_shuffle[df_shuffle['train_size'] == 0.30].nlargest(1, 'f1_macro')
sn_50 = df_shuffle[df_shuffle['train_size'] == 0.50]
if not sn_30.empty:
    ax3.scatter([30], sn_30['f1_macro'].values, s=100, label=f"SN (30%): {sn_30['f1_macro'].values[0]:.4f}", zorder=5, marker='^', color=dark2[3])
if not sn_50.empty:
    ax3.scatter([50], sn_50['f1_macro'].values, s=100, label=f"SN (50%): {sn_50['f1_macro'].values[0]:.4f}", zorder=5, marker='^', color=dark2[4])

ax3.set_xlabel('Training Size (%)', fontsize=15)
ax3.set_ylabel('F1 Score', fontsize=15)
ax3.set_title('F1 Score: LR/RF vs ShuffleNet', fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Training Time comparison
ax4 = plt.subplot(2, 2, 2)
ax4.plot(lr_data['train_size'] * 100, np.log(lr_data['training_time_s'] + 1), marker='o', label='LR Time', linewidth=2, color = dark2[0])
ax4.plot(rf_data['train_size'] * 100, np.log(rf_data['training_time_s'] + 1), marker='s', label='RF Time', linewidth=2, color = dark2[1])
ax4.plot(svm_data['train_size'] * 100, np.log(svm_data['training_time_s'] + 1), marker='+', markersize=10, markeredgewidth=2, label='SVM Time', linewidth=2, color = dark2[2])
# Add ShuffleNet training times at 30% and 50%
if not sn_30.empty:
    ax4.scatter([30], np.log(sn_30['train_time_s'].values),  s=100, label=f"SN (30%): {sn_30['train_time_s'].values[0]:.0f}s", zorder=5, marker='^', color=dark2[3])
if not sn_50.empty:
    ax4.scatter([50], np.log(sn_50['train_time_s'].values),  s=100, label=f"SN (50%): {sn_50['train_time_s'].values[0]:.0f}s", zorder=5, marker='^', color=dark2[4])
ax4.set_xlabel('Training Size (%)', fontsize=15)
ax4.set_ylabel('Log (Time (seconds) + 1)', fontsize=15)
ax4.set_title('Training Time: LR/RF vs ShuffleNet', fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. Model comparison at specific train size (0.5 = 50%)
ax5 = plt.subplot(2, 2, 3)
comparison_data = {
    'LR (50%)': {
        'Precision': lr_data[lr_data['train_size'] == 0.5]['precision'].values[0] if len(lr_data[lr_data['train_size'] == 0.5]) > 0 else 0,
        'Recall': lr_data[lr_data['train_size'] == 0.5]['recall'].values[0] if len(lr_data[lr_data['train_size'] == 0.5]) > 0 else 0,
        'F1': lr_data[lr_data['train_size'] == 0.5]['f1_score'].values[0] if len(lr_data[lr_data['train_size'] == 0.5]) > 0 else 0,
    },
    'RF (50%)': {
        'Precision': rf_data[rf_data['train_size'] == 0.5]['precision'].values[0] if len(rf_data[rf_data['train_size'] == 0.5]) > 0 else 0,
        'Recall': rf_data[rf_data['train_size'] == 0.5]['recall'].values[0] if len(rf_data[rf_data['train_size'] == 0.5]) > 0 else 0,
        'F1': rf_data[rf_data['train_size'] == 0.5]['f1_score'].values[0] if len(rf_data[rf_data['train_size'] == 0.5]) > 0 else 0,
    },
    'SVM (50%)': {
        'Precision': svm_data[svm_data['train_size'] == 0.5]['precision'].values[0] if len(svm_data[svm_data['train_size'] == 0.5]) > 0 else 0,
        'Recall': svm_data[svm_data['train_size'] == 0.5]['recall'].values[0] if len(svm_data[svm_data['train_size'] == 0.5]) > 0 else 0,
        'F1': svm_data[svm_data['train_size'] == 0.5]['f1_score'].values[0] if len(svm_data[svm_data['train_size'] == 0.5]) > 0 else 0,
    },
    'SN Best-Acc': {
        'Precision': best_acc_shuffle['precision_parasitic'],
        'Recall': best_acc_shuffle['recall_parasitic'],
        'F1': best_acc_shuffle['f1_macro'],
    },
    'SN Best-Time': {
        'Precision': best_time_shuffle['precision_parasitic'],
        'Recall': best_time_shuffle['recall_parasitic'],
        'F1': best_time_shuffle['f1_macro'],
    },
}

x = np.arange(len(comparison_data))
width = 0.25

precision_vals = [comparison_data[key]['Precision'] for key in comparison_data.keys()]
recall_vals = [comparison_data[key]['Recall'] for key in comparison_data.keys()]
f1_vals = [comparison_data[key]['F1'] for key in comparison_data.keys()]

ax5.bar(x - width, precision_vals, width, label='Precision', alpha=0.8, color=dark2[7])
ax5.bar(x, recall_vals, width, label='Recall', alpha=0.8, color=dark2[6])
ax5.bar(x + width, f1_vals, width, label='F1', alpha=0.8, color=dark2[5])

ax5.set_ylabel('Score', fontsize=15)
ax5.set_title('Metric Comparison (LR/RF/SVM @ 50% vs ShuffleNet)', fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(comparison_data.keys(), rotation=45, ha='right', fontsize=9)
ax5.legend(fontsize=8, loc='lower left')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Summary table
ax6 = plt.subplot(2, 2, 4)
ax6.axis('off')

summary_text = f"""
BEST SHUFFLENET MODELS SUMMARY
{'='*50}

BY ACCURACY (0.01 LR, BS=32, Ep=5):
  • Accuracy: {best_acc_shuffle['accuracy']:.4f}
  • Precision: {best_acc_shuffle['precision_parasitic']:.4f}
  • Recall: {best_acc_shuffle['recall_parasitic']:.4f}
  • F1-Macro: {best_acc_shuffle['f1_macro']:.4f}
  • Train Time: {best_acc_shuffle['train_time_s']:.2f}s

BY TRAINING TIME (0.01 LR, BS=16, Ep=2):
  • Accuracy: {best_time_shuffle['accuracy']:.4f}
  • Precision: {best_time_shuffle['precision_parasitic']:.4f}
  • Recall: {best_time_shuffle['recall_parasitic']:.4f}
  • F1-Macro: {best_time_shuffle['f1_macro']:.4f}
  • Train Time: {best_time_shuffle['train_time_s']:.2f}s

LR/RF/SVM at 50% Training Size:
  LR Prec: {comparison_data['LR (50%)']['Precision']:.4f} | RF Prec: {comparison_data['RF (50%)']['Precision']:.4f} | SVM Prec: {comparison_data['SVM (50%)']['Precision']:.4f}
  LR Rec:  {comparison_data['LR (50%)']['Recall']:.4f} | RF Rec:  {comparison_data['RF (50%)']['Recall']:.4f} | SVM Rec: {comparison_data['SVM (50%)']['Recall']:.4f}
  LR F1:   {comparison_data['LR (50%)']['F1']:.4f} | RF F1:   {comparison_data['RF (50%)']['F1']:.4f} | SVM F1: {comparison_data['SVM (50%)']['F1']:.4f}
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontfamily='monospace',
         fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Add subplot labels (a) .. (f) for the 2x3 grid
# ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes,
#          fontsize=12, fontweight='bold', va='top', ha='left')
# ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes,
#          fontsize=12, fontweight='bold', va='top', ha='left')
ax3.text(0.02, 0.98, '(a)', transform=ax3.transAxes,
         fontsize=12, fontweight='bold', va='top', ha='left')
ax4.text(0.02, 0.98, '(b)', transform=ax4.transAxes,
         fontsize=12, fontweight='bold', va='top', ha='left')
ax5.text(0.02, 0.98, '(c)', transform=ax5.transAxes,
         fontsize=12, fontweight='bold', va='top', ha='left')
ax6.text(0.02, 0.98, '(d)', transform=ax6.transAxes,
         fontsize=12, fontweight='bold', va='top', ha='left')

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison_best_models.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / 'model_comparison_best_models.png'}")

# ============ ADDITIONAL PLOT: ALL SHUFFLENET VS LR/RF ============


# Recall vs Precision scatter
fig = plt.figure(figsize=(10, 10))
plt.scatter(lr_data['precision'], lr_data['recall'], label='LR', s=150, marker='o', alpha=0.7, color = dark2[0])
plt.scatter(rf_data['precision'], rf_data['recall'], label='RF', s=150, marker='s', alpha=0.7, color = dark2[1])
plt.scatter(svm_data['precision'], svm_data['recall'], label='SVM', s=150, marker='+', alpha=0.7, color = dark2[2])
plt.scatter(df_shuffle['precision_parasitic'], df_shuffle['recall_parasitic'], 
          label='ShuffleNet', s=80, marker='^', alpha=0.5, c=dark2[5])
plt.scatter(best_acc_shuffle['precision_parasitic'], best_acc_shuffle['recall_parasitic'], 
          s=300, marker='*', c=dark2[3], edgecolors='black', linewidth=2, label='Best Accuracy', zorder=5)
plt.scatter(best_time_shuffle['precision_parasitic'], best_time_shuffle['recall_parasitic'], 
          s=300, marker='*', c=dark2[4], edgecolors='black', linewidth=2, label='Best Time', zorder=5)
plt.xlabel('Precision', fontsize=20)
plt.ylabel('Recall', fontsize=20)
plt.title('Precision vs Recall: All Models', fontweight='bold', fontsize=22)
plt.legend(fontsize=11, loc='lower left')
plt.grid(True, alpha=0.3)

# # ShuffleNet results by learning rate
# ax = axes[1]
# for lr_val in sorted(df_shuffle['lr'].unique()):
#     df_lr = df_shuffle[df_shuffle['lr'] == lr_val]
#     ax.scatter(df_lr['train_time_s'], df_lr['accuracy'], label=f'LR={lr_val}', s=100, alpha=0.7)
# ax.axhline(y=best_acc_shuffle['accuracy'], color='r', linestyle='--', label='Best Accuracy', linewidth=2)
# ax.axhline(y=best_time_shuffle['accuracy'], color='orange', linestyle='--', label='Best Time', linewidth=2)
# ax.set_xlabel('Training Time (s)', fontsize=11)
# ax.set_ylabel('Accuracy', fontsize=11)
# ax.set_title('ShuffleNet: Accuracy vs Time (by LR)', fontweight='bold')
# ax.legend(fontsize=9)
# ax.grid(True, alpha=0.3)

# # Add subplot labels (a) and (b)
# axes[0].text(0.02, 0.98, '(a)', transform=axes[0].transAxes,
#              fontsize=14, fontweight='bold', va='top', ha='left')
# axes[1].text(0.02, 0.98, '(b)', transform=axes[1].transAxes,
#              fontsize=14, fontweight='bold', va='top', ha='left')





plt.tight_layout()
plt.savefig(output_dir / 'model_comparison_detailed.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'model_comparison_detailed.png'}")

print("\n" + "="*70)
print("ALL PLOTS SAVED TO gridsearch_plots/")
print("="*70)
