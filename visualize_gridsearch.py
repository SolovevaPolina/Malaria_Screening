import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 20)

# Read the gridsearch results
csv_path = Path(__file__).parent / "shufflenet_gridsearch_results.csv"
df = pd.read_csv(csv_path)

# Create output directory for plots
output_dir = Path(__file__).parent / "gridsearch_plots"
output_dir.mkdir(exist_ok=True)

# Create a large figure with subplots
fig = plt.figure(figsize=(20, 24))

# ============ TIME COMPARISONS ============
# 1. Batch Size vs Train Time
ax1 = plt.subplot(4, 3, 1)
sns.scatterplot(data=df, x='batch_size', y='train_time_s', hue='lr', size='num_epochs', ax=ax1, palette='viridis', sizes=(100, 300))
ax1.set_title('Batch Size vs Train Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Train Time (seconds)')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 2. Learning Rate vs Train Time
ax2 = plt.subplot(4, 3, 2)
sns.scatterplot(data=df, x='lr', y='train_time_s', hue='batch_size', size='num_epochs', ax=ax2, palette='coolwarm', sizes=(100, 300))
ax2.set_title('Learning Rate vs Train Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Learning Rate')
ax2.set_ylabel('Train Time (seconds)')
ax2.set_xscale('log')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 3. Epochs vs Train Time
ax3 = plt.subplot(4, 3, 3)
sns.scatterplot(data=df, x='num_epochs', y='train_time_s', hue='lr', size='batch_size', ax=ax3, palette='viridis', sizes=(100, 300))
ax3.set_title('Epochs vs Train Time', fontsize=14, fontweight='bold')
ax3.set_xlabel('Number of Epochs')
ax3.set_ylabel('Train Time (seconds)')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# ============ BATCH SIZE vs ACCURACY METRICS ============
# 4. Batch Size vs Accuracy
ax4 = plt.subplot(4, 3, 4)
sns.scatterplot(data=df, x='batch_size', y='accuracy', hue='lr', size='num_epochs', ax=ax4, palette='viridis', sizes=(100, 300))
ax4.set_title('Batch Size vs Accuracy', fontsize=14, fontweight='bold')
ax4.set_xlabel('Batch Size')
ax4.set_ylabel('Accuracy')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 5. Batch Size vs Recall
ax5 = plt.subplot(4, 3, 5)
sns.scatterplot(data=df, x='batch_size', y='recall_parasitic', hue='lr', size='num_epochs', ax=ax5, palette='viridis', sizes=(100, 300))
ax5.set_title('Batch Size vs Recall', fontsize=14, fontweight='bold')
ax5.set_xlabel('Batch Size')
ax5.set_ylabel('Recall')
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 6. Batch Size vs F1
ax6 = plt.subplot(4, 3, 6)
sns.scatterplot(data=df, x='batch_size', y='f1_macro', hue='lr', size='num_epochs', ax=ax6, palette='viridis', sizes=(100, 300))
ax6.set_title('Batch Size vs F1 (Macro)', fontsize=14, fontweight='bold')
ax6.set_xlabel('Batch Size')
ax6.set_ylabel('F1 Score')
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# ============ LEARNING RATE vs ACCURACY METRICS ============
# 7. Learning Rate vs Accuracy
ax7 = plt.subplot(4, 3, 7)
sns.scatterplot(data=df, x='lr', y='accuracy', hue='batch_size', size='num_epochs', ax=ax7, palette='coolwarm', sizes=(100, 300))
ax7.set_title('Learning Rate vs Accuracy', fontsize=14, fontweight='bold')
ax7.set_xlabel('Learning Rate')
ax7.set_ylabel('Accuracy')
ax7.set_xscale('log')
ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 8. Learning Rate vs Recall
ax8 = plt.subplot(4, 3, 8)
sns.scatterplot(data=df, x='lr', y='recall_parasitic', hue='batch_size', size='num_epochs', ax=ax8, palette='coolwarm', sizes=(100, 300))
ax8.set_title('Learning Rate vs Recall', fontsize=14, fontweight='bold')
ax8.set_xlabel('Learning Rate')
ax8.set_ylabel('Recall')
ax8.set_xscale('log')
ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 9. Learning Rate vs Precision
ax9 = plt.subplot(4, 3, 9)
sns.scatterplot(data=df, x='lr', y='precision_parasitic', hue='batch_size', size='num_epochs', ax=ax9, palette='coolwarm', sizes=(100, 300))
ax9.set_title('Learning Rate vs Precision', fontsize=14, fontweight='bold')
ax9.set_xlabel('Learning Rate')
ax9.set_ylabel('Precision')
ax9.set_xscale('log')
ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# ============ EPOCHS vs ACCURACY METRICS ============
# 10. Epochs vs Accuracy
ax10 = plt.subplot(4, 3, 10)
sns.scatterplot(data=df, x='num_epochs', y='accuracy', hue='lr', size='batch_size', ax=ax10, palette='viridis', sizes=(100, 300))
ax10.set_title('Epochs vs Accuracy', fontsize=14, fontweight='bold')
ax10.set_xlabel('Number of Epochs')
ax10.set_ylabel('Accuracy')
ax10.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 11. Epochs vs Recall
ax11 = plt.subplot(4, 3, 11)
sns.scatterplot(data=df, x='num_epochs', y='recall_parasitic', hue='lr', size='batch_size', ax=ax11, palette='viridis', sizes=(100, 300))
ax11.set_title('Epochs vs Recall', fontsize=14, fontweight='bold')
ax11.set_xlabel('Number of Epochs')
ax11.set_ylabel('Recall')
ax11.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 12. Epochs vs F1
ax12 = plt.subplot(4, 3, 12)
sns.scatterplot(data=df, x='num_epochs', y='f1_macro', hue='lr', size='batch_size', ax=ax12, palette='viridis', sizes=(100, 300))
ax12.set_title('Epochs vs F1 (Macro)', fontsize=14, fontweight='bold')
ax12.set_xlabel('Number of Epochs')
ax12.set_ylabel('F1 Score')
ax12.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'gridsearch_comprehensive.png', dpi=300, bbox_inches='tight')
print(f"Saved comprehensive plot to {output_dir / 'gridsearch_comprehensive.png'}")

# ============ ADDITIONAL DETAILED PLOTS ============

# Heatmap: Average Accuracy by Batch Size and Learning Rate
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))

# For each epoch count, create a heatmap
for idx, epochs in enumerate(sorted(df['num_epochs'].unique())):
    df_epoch = df[df['num_epochs'] == epochs]
    pivot_accuracy = df_epoch.pivot_table(values='accuracy', index='batch_size', columns='lr', aggfunc='mean')
    
    sns.heatmap(pivot_accuracy, annot=True, fmt='.4f', cmap='RdYlGn', ax=axes[idx], cbar_kws={'label': 'Accuracy'})
    axes[idx].set_title(f'Accuracy: Batch Size vs LR (Epochs={epochs})', fontweight='bold')
    axes[idx].set_xlabel('Learning Rate')
    axes[idx].set_ylabel('Batch Size')

plt.tight_layout()
plt.savefig(output_dir / 'accuracy_heatmaps_by_epochs.png', dpi=300, bbox_inches='tight')
print(f"Saved accuracy heatmaps to {output_dir / 'accuracy_heatmaps_by_epochs.png'}")

# Heatmap: F1 scores by Batch Size and Learning Rate
fig3, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, epochs in enumerate(sorted(df['num_epochs'].unique())):
    df_epoch = df[df['num_epochs'] == epochs]
    pivot_f1 = df_epoch.pivot_table(values='f1_macro', index='batch_size', columns='lr', aggfunc='mean')
    
    sns.heatmap(pivot_f1, annot=True, fmt='.4f', cmap='RdYlGn', ax=axes[idx], cbar_kws={'label': 'F1 Score'})
    axes[idx].set_title(f'F1 (Macro): Batch Size vs LR (Epochs={epochs})', fontweight='bold')
    axes[idx].set_xlabel('Learning Rate')
    axes[idx].set_ylabel('Batch Size')

plt.tight_layout()
plt.savefig(output_dir / 'f1_heatmaps_by_epochs.png', dpi=300, bbox_inches='tight')
print(f"Saved F1 heatmaps to {output_dir / 'f1_heatmaps_by_epochs.png'}")

# Train Time vs Accuracy (colored by hyperparameters)
fig4, axes = plt.subplots(2, 2, figsize=(15, 12))

# By Learning Rate
ax = axes[0, 0]
for lr_val in sorted(df['lr'].unique()):
    df_lr = df[df['lr'] == lr_val]
    ax.scatter(df_lr['train_time_s'], df_lr['accuracy'], label=f'LR={lr_val}', s=100, alpha=0.7)
ax.set_xlabel('Train Time (seconds)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Train Time vs Accuracy (colored by Learning Rate)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# By Batch Size
ax = axes[0, 1]
for bs_val in sorted(df['batch_size'].unique()):
    df_bs = df[df['batch_size'] == bs_val]
    ax.scatter(df_bs['train_time_s'], df_bs['accuracy'], label=f'BS={bs_val}', s=100, alpha=0.7)
ax.set_xlabel('Train Time (seconds)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Train Time vs Accuracy (colored by Batch Size)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# By Epochs
ax = axes[1, 0]
for ep_val in sorted(df['num_epochs'].unique()):
    df_ep = df[df['num_epochs'] == ep_val]
    ax.scatter(df_ep['train_time_s'], df_ep['accuracy'], label=f'Epochs={ep_val}', s=100, alpha=0.7)
ax.set_xlabel('Train Time (seconds)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Train Time vs Accuracy (colored by Epochs)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# All metrics comparison
ax = axes[1, 1]
ax.scatter(df['train_time_s'], df['accuracy'], label='Accuracy', s=100, alpha=0.7)
ax.scatter(df['train_time_s'], df['recall_parasitic'], label='Recall', s=100, alpha=0.7)
ax.scatter(df['train_time_s'], df['precision_parasitic'], label='Precision', s=100, alpha=0.7)
ax.scatter(df['train_time_s'], df['f1_macro'], label='F1 (Macro)', s=100, alpha=0.7)
ax.set_xlabel('Train Time (seconds)', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Train Time vs All Metrics', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'time_vs_performance.png', dpi=300, bbox_inches='tight')
print(f"Saved time vs performance plot to {output_dir / 'time_vs_performance.png'}")

# Summary statistics
print("\n" + "="*60)
print("GRIDSEARCH SUMMARY STATISTICS")
print("="*60)

print("\nBest by Accuracy:")
best_accuracy = df.loc[df['accuracy'].idxmax()]
print(f"  Accuracy: {best_accuracy['accuracy']:.4f}")
print(f"  LR: {best_accuracy['lr']}, Batch Size: {best_accuracy['batch_size']}, Epochs: {best_accuracy['num_epochs']}")
print(f"  Train Time: {best_accuracy['train_time_s']:.2f}s")
print(f"  Precision: {best_accuracy['precision_parasitic']:.4f}, Recall: {best_accuracy['recall_parasitic']:.4f}, F1: {best_accuracy['f1_macro']:.4f}")

print("\nBest by F1 (Macro):")
best_f1 = df.loc[df['f1_macro'].idxmax()]
print(f"  F1 (Macro): {best_f1['f1_macro']:.4f}")
print(f"  LR: {best_f1['lr']}, Batch Size: {best_f1['batch_size']}, Epochs: {best_f1['num_epochs']}")
print(f"  Train Time: {best_f1['train_time_s']:.2f}s")
print(f"  Accuracy: {best_f1['accuracy']:.4f}, Precision: {best_f1['precision_parasitic']:.4f}, Recall: {best_f1['recall_parasitic']:.4f}")

print("\nFastest Training:")
fastest = df.loc[df['train_time_s'].idxmin()]
print(f"  Train Time: {fastest['train_time_s']:.2f}s")
print(f"  LR: {fastest['lr']}, Batch Size: {fastest['batch_size']}, Epochs: {fastest['num_epochs']}")
print(f"  Accuracy: {fastest['accuracy']:.4f}, F1: {fastest['f1_macro']:.4f}")

print("\nAverage metrics by Learning Rate:")
for lr_val in sorted(df['lr'].unique()):
    df_lr = df[df['lr'] == lr_val]
    print(f"  LR={lr_val}: Accuracy={df_lr['accuracy'].mean():.4f}, F1={df_lr['f1_macro'].mean():.4f}, Time={df_lr['train_time_s'].mean():.2f}s")

print("\nAverage metrics by Batch Size:")
for bs_val in sorted(df['batch_size'].unique()):
    df_bs = df[df['batch_size'] == bs_val]
    print(f"  BS={bs_val}: Accuracy={df_bs['accuracy'].mean():.4f}, F1={df_bs['f1_macro'].mean():.4f}, Time={df_bs['train_time_s'].mean():.2f}s")

print("\nAverage metrics by Epochs:")
for ep_val in sorted(df['num_epochs'].unique()):
    df_ep = df[df['num_epochs'] == ep_val]
    print(f"  Epochs={ep_val}: Accuracy={df_ep['accuracy'].mean():.4f}, F1={df_ep['f1_macro'].mean():.4f}, Time={df_ep['train_time_s'].mean():.2f}s")

print("\n" + "="*60)
print(f"All plots saved to: {output_dir}")
print("="*60)
