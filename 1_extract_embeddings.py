import os
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity # New import

warnings.filterwarnings('ignore') # 忽略警告訊息

SIMILARITY_THRESHOLD = 0.9

def extract_enhanced_features(img_path):
    """
    提取強化的多模態特徵。
    DeepFace本身會處理圖片的預處理。
    """

    features = []
    
    try:
        # 1. 情緒分析特徵 (主要特徵)
        emotion_obj_list = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
        emotion_obj = emotion_obj_list[0]
        emotion_scores = list(emotion_obj['emotion'].values())
        
        # 強化處理：計算情緒特徵的統計數據
        emotion_scores_np = np.array(emotion_scores)
        features.extend(emotion_scores_np)  # 原始Softmax分數
        features.append(np.max(emotion_scores_np))  # 最大情緒強度
        features.append(np.std(emotion_scores_np))  # 情緒離散度
        # 主要與次要情緒之間的差異
        sorted_emotions = np.sort(emotion_scores_np)
        if len(sorted_emotions) >= 2:
            features.append(sorted_emotions[-1] - sorted_emotions[-2])
        else:
            features.append(0) # 如果只有一個情緒分數，則使用0或其他佔位符
        
        # 2. 年齡與性別特徵 (輔助特徵)
        # 使用原始圖片路徑
        demography_obj_list = DeepFace.analyze(img_path=img_path, actions=['age', 'gender'], enforce_detection=False)
        demography_obj = demography_obj_list[0]
        age = demography_obj['age']
        # 性別可以用多種方式表示，此處取為「女性」的機率
        gender_prob_woman = demography_obj['gender']['Woman'] 
        features.extend([age / 100.0, gender_prob_woman])  # 標準化年齡
        
        # 3. 臉部嵌入特徵 (深度特徵)
        # 使用原始圖片路徑
        embedding_obj_list = DeepFace.represent(img_path=img_path, model_name='Facenet512', enforce_detection=False)
        embedding_obj = embedding_obj_list[0]
        face_embedding = embedding_obj['embedding']
        # 截斷至前128維以避免維度過高
        features.extend(face_embedding[:128])
        
    except Exception as e:
        print(f"特徵提取失敗：{img_path}，錯誤訊息：{e}")
        return None
    
    return np.array(features)


# 設定資料夾路徑
root_dir = os.path.join(os.path.dirname(__file__), 'dataset')
class_names = ['anger', 'disgust', 'fear', 'happy', 'pain', 'sad']

embeddings = []
labels = []
filenames = [] # 新增：用於儲存檔案名稱的列表
log_report_data = [] # New: For storing log information

print("開始提取強化特徵並進行資料清理...") # Updated message
print(f"使用相似度閾值 (SIMILARITY_THRESHOLD): {SIMILARITY_THRESHOLD}")
for class_name in class_names:
    class_dir = os.path.join(root_dir, class_name)
    image_files_all = os.listdir(class_dir) # Get all files first
    
    print(f"\\n處理類別: {class_name} (找到 {len(image_files_all)} 個檔案)")

    accepted_images_for_class_embeddings = [] # Store embeddings of accepted images for similarity check within this class
    
    processed_image_count = 0
    skipped_file_type_count = 0
    skipped_face_issue_count = 0
    skipped_similarity_count = 0
    skipped_extraction_error_count = 0

    for fname in image_files_all:
        img_path = os.path.join(class_dir, fname)
        log_entry = {'filename': fname, 'class': class_name, 'status': '', 'reason': ''}

        # 1. 排除非jpg/jpeg與png格式的圖片
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            skipped_file_type_count += 1
            log_entry['status'] = 'failed'
            log_entry['reason'] = '檔案格式不符'
            log_report_data.append(log_entry)
            continue

        # 2. 排除有兩個以上人臉的圖片 或 無法偵測人臉
        try:
            analysis_results = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False, silent=True)
            
            if not isinstance(analysis_results, list) or len(analysis_results) == 0:
                print(f"無法偵測到人臉或分析失敗，跳過：{img_path}")
                skipped_face_issue_count +=1
                log_entry['status'] = 'failed'
                log_entry['reason'] = '無法偵測到人臉或分析失敗'
                log_report_data.append(log_entry)
                continue
            if len(analysis_results) > 1:
                print(f"偵測到多於一張人臉 ({len(analysis_results)} 張)，跳過：{img_path}")
                skipped_face_issue_count += 1
                log_entry['status'] = 'failed'
                log_entry['reason'] = f'偵測到多於一張人臉 ({len(analysis_results)} 張)'
                log_report_data.append(log_entry)
                continue
            
        except Exception as e:
            print(f"人臉檢查時發生錯誤，跳過：{img_path}。錯誤：{e}")
            skipped_face_issue_count += 1
            log_entry['status'] = 'failed'
            log_entry['reason'] = f'人臉檢查時發生錯誤: {e}'
            log_report_data.append(log_entry)
            continue

        # 3. 排除相似度過高的圖片 (within the same class)
        try:
            current_embedding_obj_list = DeepFace.represent(img_path=img_path, model_name='Facenet512', enforce_detection=False)
            
            if not current_embedding_obj_list or not isinstance(current_embedding_obj_list, list) or len(current_embedding_obj_list) == 0:
                print(f"無法為相似度檢查生成嵌入，跳過：{img_path}")
                skipped_similarity_count +=1 
                log_entry['status'] = 'failed'
                log_entry['reason'] = '無法為相似度檢查生成嵌入'
                log_report_data.append(log_entry)
                continue

            current_embedding_obj = current_embedding_obj_list[0]
            current_face_embedding_list = current_embedding_obj['embedding']
            current_face_embedding_np = np.array(current_face_embedding_list).reshape(1, -1)

            is_too_similar = False
            if accepted_images_for_class_embeddings:
                previous_embeddings_np = np.array(accepted_images_for_class_embeddings)
                similarities = cosine_similarity(current_face_embedding_np, previous_embeddings_np)
                if np.any(similarities > SIMILARITY_THRESHOLD):
                    is_too_similar = True
                    skipped_similarity_count += 1
                    log_entry['status'] = 'failed'
                    log_entry['reason'] = f'圖片相似度過高 (與同類別中已接受圖片的最大相似度: {np.max(similarities):.2f})'
                    log_report_data.append(log_entry)
            
            if is_too_similar:
                continue
            
            accepted_images_for_class_embeddings.append(current_face_embedding_list)

        except Exception as e:
            print(f"相似度檢查或嵌入提取時發生錯誤，跳過：{img_path}。錯誤：{e}")
            skipped_similarity_count += 1
            log_entry['status'] = 'failed'
            log_entry['reason'] = f'相似度檢查或嵌入提取時發生錯誤: {e}'
            log_report_data.append(log_entry)
            continue
            
        enhanced_features = extract_enhanced_features(img_path)
        
        if enhanced_features is not None:
            embeddings.append(enhanced_features)
            labels.append(class_name)
            filenames.append(fname)
            processed_image_count += 1
            log_entry['status'] = 'success'
            log_report_data.append(log_entry)
        else:
            print(f"因強化特徵提取錯誤而跳過檔案（在清理後）：{img_path}")
            skipped_extraction_error_count +=1
            log_entry['status'] = 'failed'
            log_entry['reason'] = '強化特徵提取錯誤 (extract_enhanced_features 回傳 None)'
            log_report_data.append(log_entry)

    print(f"類別 {class_name} 處理完成。")
    print(f"  - 總檔案數: {len(image_files_all)}")
    print(f"  - 因檔案類型跳過: {skipped_file_type_count}")
    print(f"  - 因人臉問題跳過: {skipped_face_issue_count}")
    print(f"  - 因相似度過高跳過: {skipped_similarity_count}")
    print(f"  - 因後續提取錯誤跳過: {skipped_extraction_error_count}")
    print(f"  - 成功處理並加入特徵的圖片數量: {processed_image_count}")
    print("-" * 30)

if not embeddings:
    print("未提取到任何嵌入特徵。請檢查資料集和特徵提取過程。")
    # 退出或進行適當的錯誤處理
    exit()

embeddings = np.array(embeddings)
labels = np.array(labels)
filenames = np.array(filenames) # 新增：轉換為 NumPy 陣列

print(f"\\n原始特徵維度: {embeddings.shape}")
print(f"標籤數量: {len(labels)}")
    
# 標準化特徵
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# 特徵選擇 - 選擇最具區別性的特徵
# 確保 k 值不大於特徵數量
k_features = min(50, embeddings_scaled.shape[1]) 
if k_features > 0 :
    selector = SelectKBest(score_func=f_classif, k=k_features)
    embeddings_selected = selector.fit_transform(embeddings_scaled, labels)
    print(f"\\n特徵選擇後維度: {embeddings_selected.shape}")
else:
    print("\\n因 k_features 為 0 或負數，跳過特徵選擇。")
    embeddings_selected = embeddings_scaled


# 儲存embedding特徵與標籤
np.save('embeddings.npy', embeddings_selected)
np.save('labels.npy', labels)
np.save('filenames.npy', filenames) # 新增：儲存檔案名稱陣列
print('已儲存 embeddings.npy、labels.npy 與 filenames.npy')

# 新增：將所有資訊合併並儲存為 CSV 檔案
print("\\n開始將整合資訊儲存為 CSV 檔案...")
try:
    # 建立特徵欄位名稱
    feature_columns = [f'feature_{i}' for i in range(embeddings_selected.shape[1])]
    
    # 建立 DataFrame
    df_features = pd.DataFrame(embeddings_selected, columns=feature_columns)
    df_features['label'] = labels
    df_features['filename'] = filenames
    
    # 儲存為 CSV
    csv_output_path = 'enhanced_features_details.csv'
    df_features.to_csv(csv_output_path, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 以確保中文字元正確顯示
    print(f"已將整合資訊儲存至 {csv_output_path}")
except Exception as e:
    print(f"儲存 CSV 檔案時發生錯誤: {e}")

# New: Save the log report
print("\\n開始儲存特徵提取日誌報告...")
try:
    df_log = pd.DataFrame(log_report_data)
    log_csv_output_path = 'feature_extraction_log.csv'
    df_log.to_csv(log_csv_output_path, index=False, encoding='utf-8-sig')
    print(f"已將特徵提取日誌儲存至 {log_csv_output_path}")
except Exception as e:
    print(f"儲存特徵提取日誌 CSV 檔案時發生錯誤: {e}")