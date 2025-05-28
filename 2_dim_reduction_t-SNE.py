import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

# 載入embedding特徵與標籤
embeddings = np.load('embeddings.npy')
labels = np.load('labels.npy')
filenames = np.load('filenames.npy') # 載入檔名

# t-SNE降維
# 您可以調整以下參數：perplexity（困惑度）、max_iter（最大迭代次數）、learning_rate（學習率）
# 常見的 perplexity 範圍是 5-50
tsne = TSNE(n_components=2, 
            perplexity=40, # 或者您選擇的困惑度
            learning_rate=200, 
            max_iter=2000,  # 確保迭代次數足夠
            init='pca',     # 嘗試 PCA 初始化
            random_state=42, 
            n_jobs=-1)
embeddings_2d = tsne.fit_transform(embeddings)

# 儲存降維後的2D座標
np.save('embeddings_2d.npy', embeddings_2d)
print('已儲存 embeddings_2d.npy (2D t-SNE降維特徵)')

# 轉存為csv檔案，方便d3.js等工具視覺化
csv_path = 'embeddings_2d.csv'
df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
df['label'] = labels
df['filename'] = filenames # 新增檔名欄位
df.to_csv(csv_path, index=False)
print(f'已儲存 {csv_path}')
