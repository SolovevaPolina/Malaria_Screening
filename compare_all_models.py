import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")

# Read data files
csv_path = Path(__file__).parent / "shufflenet_gridsearch_results.csv"
lrrf_path = Path(__file__).parent / "LRRF_model_comparison_results.csv"

df_shuffle = pd.read_csv(csv_path)
df_lrrf = pd.read_csv(lrrf_path)

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
print(df_lrrf.to_string())

# ============ CREATE COMPARISON PLOTS ============

fig = plt.figure(figsize=(18, 12))

# 1. Accuracy comparison
ax1 = plt.subplot(2, 3, 1)
models_to_plot = ['Logistic Regression', 'Random Forest']
lr_data = df_lrrf[df_lrrf['Model Name'] == 'Logistic Regression'].sort_values('Train Size')
rf_data = df_lrrf[df_lrrf['Model Name'] == 'Random Forest'].sort_values('Train Size')

ax1.plot(lr_data['Train Size'] * 100, lr_data['Precision'], marker='o', label='LR Precision', linewidth=2)
ax1.plot(rf_data['Train Size'] * 100, rf_data['Precision'], marker='s', label='RF Precision', linewidth=2)
# ax1.scatter(y=best_acc_shuffle['precision_parasitic'], color='r', label=f"SN Best-Acc: {best_acc_shuffle['precision_parasitic']:.4f}")
# ax1.scatter(y=best_time_shuffle['precision_parasitic'], color='orange', label=f"SN Best-Time: {best_time_shuffle['precision_parasitic']:.4f}")
ax1.scatter([30], [best_acc_shuffle['precision_parasitic']], color='r', s=100, label=f"SN Best-Acc: {best_acc_shuffle['precision_parasitic']:.4f}", zorder=5)
ax1.scatter([30], [best_time_shuffle['precision_parasitic']], color='orange', s=100, label=f"SN Best-Time: {best_time_shuffle['precision_parasitic']:.4f}", zorder=5)

ax1.set_xlabel('Training Size (%)', fontsize=11)
ax1.set_ylabel('Precision', fontsize=11)
ax1.set_title('Precision: LR/RF vs ShuffleNet', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Recall comparison
ax2 = plt.subplot(2, 3, 2)
ax2.plot(lr_data['Train Size'] * 100, lr_data['Recall'], marker='o', label='LR Recall', linewidth=2)
ax2.plot(rf_data['Train Size'] * 100, rf_data['Recall'], marker='s', label='RF Recall', linewidth=2)
# ax2.scatter(y=best_acc_shuffle['recall_parasitic'], color='r', label=f"SN Best-Acc: {best_acc_shuffle['recall_parasitic']:.4f}")
# ax2.scatter(y=best_time_shuffle['recall_parasitic'], color='orange', label=f"SN Best-Time: {best_time_shuffle['recall_parasitic']:.4f}")
ax2.scatter([30], [best_acc_shuffle['recall_parasitic']], color='r', s=100, label=f"SN Best-Acc: {best_acc_shuffle['recall_parasitic']:.4f}", zorder=5)
ax2.scatter([30], [best_time_shuffle['recall_parasitic']], color='orange', s=100, label=f"SN Best-Time: {best_time_shuffle['recall_parasitic']:.4f}", zorder=5)

ax2.set_xlabel('Training Size (%)', fontsize=11)
ax2.set_ylabel('Recall', fontsize=11)
ax2.set_title('Recall: LR/RF vs ShuffleNet', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. F1 Score comparison
ax3 = plt.subplot(2, 3, 3)
ax3.plot(lr_data['Train Size'] * 100, lr_data['F1 Score'], marker='o', label='LR F1', linewidth=2)
ax3.plot(rf_data['Train Size'] * 100, rf_data['F1 Score'], marker='s', label='RF F1', linewidth=2)
# ax3.scatter(y=best_acc_shuffle['f1_macro'], color='r', label=f"SN Best-Acc: {best_acc_shuffle['f1_macro']:.4f}")
# ax3.scatter(y=best_time_shuffle['f1_macro'], color='orange', label=f"SN Best-Time: {best_time_shuffle['f1_macro']:.4f}")
ax3.scatter([30], [best_acc_shuffle['f1_macro']], color='r', s=100, label=f"SN Best-Acc: {best_acc_shuffle['f1_macro']:.4f}", zorder=5)
ax3.scatter([30], [best_time_shuffle['f1_macro']], color='orange', s=100, label=f"SN Best-Time: {best_time_shuffle['f1_macro']:.4f}", zorder=5)

ax3.set_xlabel('Training Size (%)', fontsize=11)
ax3.set_ylabel('F1 Score', fontsize=11)
ax3.set_title('F1 Score: LR/RF vs ShuffleNet', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Training Time comparison
ax4 = plt.subplot(2, 3, 4)
ax4.plot(lr_data['Train Size'] * 100, lr_data['Time Taken (s)'], marker='o', label='LR Time', linewidth=2)
ax4.plot(rf_data['Train Size'] * 100, rf_data['Time Taken (s)'], marker='s', label='RF Time', linewidth=2)
# ax4.scatter(y=best_acc_shuffle['train_time_s'], color='r', label=f"SN Best-Acc: {best_acc_shuffle['train_time_s']:.2f}s")
# ax4.scatter(y=best_time_shuffle['train_time_s'], color='orange', label=f"SN Best-Time: {best_time_shuffle['train_time_s']:.2f}s")
ax4.scatter([30], [best_acc_shuffle['train_time_s']], color='r', s=100, label=f"SN Best-Acc: {best_acc_shuffle['train_time_s']:.2f}s", zorder=5)
ax4.scatter([30], [best_time_shuffle['train_time_s']], color='orange', s=100, label=f"SN Best-Time: {best_time_shuffle['train_time_s']:.2f}s", zorder=5)
ax4.set_xlabel('Training Size (%)', fontsize=11)
ax4.set_ylabel('Time (seconds)', fontsize=11)
ax4.set_title('Training Time: LR/RF vs ShuffleNet', fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. Model comparison at specific train size (0.5 = 50%)
ax5 = plt.subplot(2, 3, 5)
comparison_data = {
    'LR (50%)': {
        'Precision': lr_data[lr_data['Train Size'] == 0.5]['Precision'].values[0] if len(lr_data[lr_data['Train Size'] == 0.5]) > 0 else 0,
        'Recall': lr_data[lr_data['Train Size'] == 0.5]['Recall'].values[0] if len(lr_data[lr_data['Train Size'] == 0.5]) > 0 else 0,
        'F1': lr_data[lr_data['Train Size'] == 0.5]['F1 Score'].values[0] if len(lr_data[lr_data['Train Size'] == 0.5]) > 0 else 0,
    },
    'RF (50%)': {
        'Precision': rf_data[rf_data['Train Size'] == 0.5]['Precision'].values[0] if len(rf_data[rf_data['Train Size'] == 0.5]) > 0 else 0,
        'Recall': rf_data[rf_data['Train Size'] == 0.5]['Recall'].values[0] if len(rf_data[rf_data['Train Size'] == 0.5]) > 0 else 0,
        'F1': rf_data[rf_data['Train Size'] == 0.5]['F1 Score'].values[0] if len(rf_data[rf_data['Train Size'] == 0.5]) > 0 else 0,
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

ax5.bar(x - width, precision_vals, width, label='Precision', alpha=0.8)
ax5.bar(x, recall_vals, width, label='Recall', alpha=0.8)
ax5.bar(x + width, f1_vals, width, label='F1', alpha=0.8)

ax5.set_ylabel('Score', fontsize=11)
ax5.set_title('Metric Comparison (LR/RF @ 50% vs ShuffleNet)', fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(comparison_data.keys(), rotation=45, ha='right', fontsize=9)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# 6. Summary table
ax6 = plt.subplot(2, 3, 6)
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

LR/RF at 50% Training Size:
  LR Prec: {comparison_data['LR (50%)']['Precision']:.4f} | RF Prec: {comparison_data['RF (50%)']['Precision']:.4f}
  LR Rec:  {comparison_data['LR (50%)']['Recall']:.4f} | RF Rec:  {comparison_data['RF (50%)']['Recall']:.4f}
  LR F1:   {comparison_data['LR (50%)']['F1']:.4f} | RF F1:   {comparison_data['RF (50%)']['F1']:.4f}
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontfamily='monospace',
         fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Add subplot labels (a) .. (f) for the 2x3 grid
ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes,
         fontsize=12, fontweight='bold', va='top', ha='left')
ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes,
         fontsize=12, fontweight='bold', va='top', ha='left')
ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes,
         fontsize=12, fontweight='bold', va='top', ha='left')
ax4.text(0.02, 0.98, '(d)', transform=ax4.transAxes,
         fontsize=12, fontweight='bold', va='top', ha='left')
ax5.text(0.02, 0.98, '(e)', transform=ax5.transAxes,
         fontsize=12, fontweight='bold', va='top', ha='left')
ax6.text(0.02, 0.98, '(f)', transform=ax6.transAxes,
         fontsize=12, fontweight='bold', va='top', ha='left')

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison_best_models.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / 'model_comparison_best_models.png'}")

# ============ ADDITIONAL PLOT: ALL SHUFFLENET VS LR/RF ============
fig2, axes = plt.subplots(2, 1, figsize=(16, 10))


# Recall vs Precision scatter
ax = axes[0]
ax.scatter(lr_data['Precision'], lr_data['Recall'], label='LR', s=150, marker='o', alpha=0.7)
ax.scatter(rf_data['Precision'], rf_data['Recall'], label='RF', s=150, marker='s', alpha=0.7)
ax.scatter(df_shuffle['precision_parasitic'], df_shuffle['recall_parasitic'], 
          label='ShuffleNet', s=80, marker='^', alpha=0.5, c='green')
ax.scatter(best_acc_shuffle['precision_parasitic'], best_acc_shuffle['recall_parasitic'], 
          s=300, marker='*', c='red', edgecolors='black', linewidth=2, label='Best Accuracy', zorder=5)
ax.scatter(best_time_shuffle['precision_parasitic'], best_time_shuffle['recall_parasitic'], 
          s=300, marker='*', c='orange', edgecolors='black', linewidth=2, label='Best Time', zorder=5)
ax.set_xlabel('Precision', fontsize=11)
ax.set_ylabel('Recall', fontsize=11)
ax.set_title('Precision vs Recall: All Models', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ShuffleNet results by learning rate
ax = axes[1]
for lr_val in sorted(df_shuffle['lr'].unique()):
    df_lr = df_shuffle[df_shuffle['lr'] == lr_val]
    ax.scatter(df_lr['train_time_s'], df_lr['accuracy'], label=f'LR={lr_val}', s=100, alpha=0.7)
ax.axhline(y=best_acc_shuffle['accuracy'], color='r', linestyle='--', label='Best Accuracy', linewidth=2)
ax.axhline(y=best_time_shuffle['accuracy'], color='orange', linestyle='--', label='Best Time', linewidth=2)
ax.set_xlabel('Training Time (s)', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('ShuffleNet: Accuracy vs Time (by LR)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Add subplot labels (a) and (b)
axes[0].text(0.02, 0.98, '(a)', transform=axes[0].transAxes,
             fontsize=14, fontweight='bold', va='top', ha='left')
axes[1].text(0.02, 0.98, '(b)', transform=axes[1].transAxes,
             fontsize=14, fontweight='bold', va='top', ha='left')





plt.tight_layout()
plt.savefig(output_dir / 'model_comparison_detailed.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'model_comparison_detailed.png'}")

print("\n" + "="*70)
print("ALL PLOTS SAVED TO gridsearch_plots/")
print("="*70)
