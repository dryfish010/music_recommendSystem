import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import streamlit as st
import time
import requests

def search_artist_online(song_name):
    url = f"https://api.spotify.com/v1/search?q={song_name}&type=track"
    headers = {"Authorization": "Bearer YOUR_ACCESS_TOKEN"} 
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if data["tracks"]["items"]:
            return data["tracks"]["items"][0]["artists"][0]["name"]  # 提取歌手名稱
    return None
# ------------------------------
# 資料預處理
# ------------------------------
df = pd.read_csv("music.csv", encoding="ISO-8859-1")
df.columns = [c.strip() for c in df.columns]
df['title'] = df['title'].apply(lambda x: re.sub(r"\s*\(.*?\)", "", x))

features = ['bpm', 'nrgy', 'dnce', 'dB', 'val', 'acous', 'spch','pop','dur']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 動態選擇最適群集數（Elbow Method）
# ------------------------------
def find_optimal_clusters(X_scaled):
    distortions = []
    K_range = range(5, 20)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)  # 記錄誤差

    optimal_k = K_range[np.argmin(np.gradient(distortions))]  # 選擇誤差變化最小的
    return optimal_k

# 訓練 KMeans
optimal_clusters = find_optimal_clusters(X_scaled)
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 訓練決策樹 & 隨機森林
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['cluster'], test_size=0.2, random_state=42)

tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# 推薦函數處理
def get_cluster(song_name, model):
    if song_name not in df['title'].values:
        return None
    idx = df[df['title'] == song_name].index[0]
    return model.predict(X_scaled[idx].reshape(1, -1))[0]

def recommend_from_cluster(cluster_id, song_name, top_n=5):
    recs = df[(df['cluster'] == cluster_id) & (df['title'] != song_name)]
    return recs[['title', 'artist']].head(top_n)

#相似度推薦
def get_similar_songs(song_name, top_n=5):
    if song_name not in df['title'].values:
        return None
    idx = df[df['title'] == song_name].index[0]
    song_vector = X_scaled[idx].reshape(1, -1)

    similarities = cosine_similarity(song_vector, X_scaled)[0]
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]  # 排序相似度，排除自身

    return df.iloc[similar_indices][['title', 'artist']]
#融合推薦(三個不同演算法)
def fusion_recommend(song_name, top_n=5):
    song_name = song_name.split(" (")[0]

    if song_name not in df['title'].values:
        artist_name = df[df['title'].str.contains(song_name, case=False, na=False)]['artist'].values
        if len(artist_name) > 0:
            artist_name = artist_name[0]
        else:
            artist_name = search_artist_online(song_name)  # 用 API 查找歌手名稱
        
        if artist_name:
            return df[df['artist'] == artist_name][['title', 'artist']].head(top_n)
        else:
            return None  # 如果找不到歌手，就回傳 None

    # 如果歌曲在資料庫中，則執行原本的推薦邏輯：
    idx = df[df['title'] == song_name].index[0]
    kmeans_cluster = df.loc[idx, 'cluster']
    tree_cluster = get_cluster(song_name, tree_model)
    rf_cluster = get_cluster(song_name, rf_model)
    #排序
    recs = []
    recs += recommend_from_cluster(kmeans_cluster, song_name, 10)['title'].tolist()
    recs += recommend_from_cluster(tree_cluster, song_name, 10)['title'].tolist()
    recs += recommend_from_cluster(rf_cluster, song_name, 10)['title'].tolist()
    recs += get_similar_songs(song_name, 10)['title'].tolist()  # 加入相似度推薦

    counter = Counter(recs)
    common_titles = [title for title, _ in counter.most_common() if title != song_name]
    
    rec_df = df[df['title'].isin(common_titles)][['title', 'artist']].drop_duplicates()

    return rec_df.head(top_n)


# YouTube 搜尋
def get_youtube_link(song_name, artist_name):
    search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={song_name} {artist_name}&type=video&key=YOUR_API_KEY"
    response = requests.get(search_url)
    
    if response.status_code == 200:
        data = response.json()
        video_id = data["items"][0]["id"]["videoId"]
        return f"https://www.youtube.com/watch?v={video_id}"
    return None

# Streamlit 使用者介面
st.title("🎵 音樂推薦系統")
song_name = st.text_input("請輸入歌曲名稱：").strip()
selected_song = None  # 儲存使用者選擇的歌曲
if song_name:
    result = fusion_recommend(song_name, top_n=5)

    if result is None:
        st.write(f"❌ 找不到歌曲：{song_name}")
    elif 'title' in result.columns:  # ✅ 有推薦歌曲，顯示推薦歌曲
        st.subheader("推薦歌曲")
        for i, row in result.iterrows():
            if st.button(f"🎵 {row['title']} - {row['artist']}", key=row['title']):
                selected_song = row

    elif 'artist' in result.columns:  # ✅ 沒找到歌曲，但提供歌手的其他作品
        st.subheader(f"這首歌可能是由 {result['artist'].values[0]} 演唱")
        st.write("推薦該歌手的其他作品：")
        for i, row in result.iterrows():
            st.write(f"🎵 {row['title']} - {row['artist']}")

# PCA 視覺化
def visualize_recommendation(song_name, recommended_titles):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))  # 創建 `fig, ax` 來替代 `plt.figure()`
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='cluster', palette='tab10', legend=False, alpha=0.5, ax=ax)

    song_point = df[df['title'] == song_name]
    rec_points = df[df['title'].isin(recommended_titles)]

    ax.scatter(song_point['PCA1'], song_point['PCA2'], c='red', label='Input Song', s=100, edgecolor='black')
    ax.scatter(rec_points['PCA1'], rec_points['PCA2'], c='green', label='Recommendations', s=80)

    ax.set_title(f"{song_name} Recommendation")
    ax.legend()
    
    st.pyplot(fig)  