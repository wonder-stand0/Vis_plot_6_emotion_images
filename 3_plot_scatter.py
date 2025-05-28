import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

# 設定中文字體以解決標題亂碼問題
# 請確保您的系統中已安裝 'Microsoft JhengHei' 字體，或者替換為其他支援繁體中文的字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# 讀取2D降維資料（例如 t-SNE 或 UMAP 處理後的結果）
csv_path = 'embeddings_2d.csv'
df = pd.read_csv(csv_path)

# 設定顏色和風格
plt.style.use('seaborn-v0_8-whitegrid') # 使用 seaborn 風格以獲得更佳的視覺效果
# 使用 'tab10' 調色盤，它為分類數據提供了高對比度的顏色
palette = sns.color_palette('tab10', n_colors=df['label'].nunique()) 

plt.figure(figsize=(12, 10))

# 繪製散佈圖
scatter_plot = sns.scatterplot(
    data=df, 
    x='x', 
    y='y', 
    hue='label', 
    palette=palette, 
    s=50,  # 增加點的大小以提高可見度
    alpha=0.8, # 設定點的透明度，使其稍微不透明
    edgecolor='w', # 設定點的邊緣顏色為白色，以增強分離感
    linewidth=0.5
)

# 添加密度輪廓或橢圓 (可選，會增加圖表複雜度，但能更好地顯示分佈)
for i, label in enumerate(df['label'].unique()):
    label_df = df[df['label'] == label]
    x_coords = label_df['x']
    y_coords = label_df['y']
    
    # 繪製信賴橢圓 (confidence ellipse)
    # 需要計算共變異數矩陣和平均值
    if len(x_coords) > 2 and len(y_coords) > 2: # 需要足夠的點來計算共變異數
        cov = np.cov(x_coords, y_coords)
        mean_x, mean_y = np.mean(x_coords), np.mean(y_coords)
        
        # 特徵分解以確定橢圓的方向和大小
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:,order]
        
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        
        # 橢圓的寬度和高度 (例如，代表2個標準差)
        # 卡方分佈 (Chi-squared distribution)，自由度為2，95%信賴區間 -> 臨界值約為5.991
        # 或者簡單地使用標準差的倍數，例如 2*sqrt(val)
        width, height = 2 * np.sqrt(5.991 * vals) # 代表95%信賴區間
        # width, height = 2 * 2 * np.sqrt(vals) # 較簡單的方式：2倍標準差

        ell = Ellipse(xy=(mean_x, mean_y),
                      width=width, height=height,
                      angle=theta, color=palette[i], alpha=0.05)
        scatter_plot.add_artist(ell)

plt.title('2D t-SNE embedding feature scatter plot (with confidence ellipse)', fontsize=16)
plt.xlabel('t-SNE Component 1', fontsize=14)
plt.ylabel('t-SNE Component 2', fontsize=14)
plt.legend(title='Emotional Types', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # 調整佈局以容納圖例

# 儲存圖表
output_plot_path = 'embeddings_scatter_plot_enhanced.png'
plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
print(f"Scatter plot saved to {output_plot_path}")

plt.show()
