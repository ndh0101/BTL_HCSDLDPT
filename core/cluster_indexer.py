import os
import pickle
import time
import numpy as np
from sklearn.cluster import KMeans
from db_connection import get_connection

# CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CORE_DIR)
DB_DIR = os.path.join(BASE_DIR, 'database')
KMEANS_MODEL_FILE = os.path.join(DB_DIR, 'kmeans_model.pkl')
IVF_INDEX_FILE = os.path.join(DB_DIR, 'ivf_index.pkl')

def build_clusters(k=20):
    """
    Đọc vector từ CSDL, chạy thuật toán K-Means gom cụm và tạo Chỉ mục ngược (IVF).
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT Image_ID, Feature_Vector FROM Bird_Metadata")
    features_dict = {}
    for row in cursor.fetchall():
        image_id = row[0]
        vector = pickle.loads(row[1]) # Giải nén nhị phân thành numpy array
        features_dict[image_id] = vector
    cursor.close()
    conn.close()
    if not features_dict:
        print("Dữ liệu vector trong CSDL trống.")
        return
    # Chuẩn bị ma trận dữ liệu cho thuật toán K-Means
    list_ids = list(features_dict.keys())
    list_vectors = list(features_dict.values())
    # Chuyển đổi thành ma trận Numpy (N x D), trong đó N là số ảnh, D là số chiều vector
    X = np.array(list_vectors)
    print(f"Kích thước ma trận dữ liệu: {X.shape[0]} ảnh x {X.shape[1]} chiều.")
    # Khởi tạo và Huấn luyện mô hình K-Means
    start_time = time.time()
    print(f"Huấn luyện thuật toán K-Means gom thành {k} cụm...")
    # Sử dụng n_init='auto' và random_state=42 để kết quả phân cụm luôn cố định qua các lần chạy
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    # Xây dựng Chỉ mục ngược (Inverted Index)
    ivf_index = {}
    for i in range(k):
        ivf_index[i] = [] # Khởi tạo danh sách rỗng cho từng cụm
    # Duyệt qua các nhãn cụm (kmeans.labels_) và đưa ID ảnh vào đúng cụm của nó
    labels = kmeans.labels_
    for i, cluster_id in enumerate(labels):
        image_id = list_ids[i]
        ivf_index[cluster_id].append(image_id)
    # In thống kê phân bố ảnh trong các cụm
    print("Thống kê phân bố ảnh trong các cụm:")
    for cluster_id, images_in_cluster in ivf_index.items():
        print(f"   - Cụm {cluster_id:<2}: {len(images_in_cluster):<3} ảnh")
    # Lưu Model và IVF
    with open(KMEANS_MODEL_FILE, 'wb') as f:
        pickle.dump(kmeans, f)
    with open(IVF_INDEX_FILE, 'wb') as f:
        pickle.dump(ivf_index, f)
    # Tổng kết
    end_time = time.time()
    print(f"Thời gian huấn luyện: {round(end_time - start_time, 2)} giây")
    print(f"Mô hình lưu tại: {KMEANS_MODEL_FILE}")
    print(f"Chỉ mục lưu tại: {IVF_INDEX_FILE}")

if __name__ == "__main__":
    build_clusters(k=30)