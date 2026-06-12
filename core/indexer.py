import os
import pickle
import time
import numpy as np
from db_connection import get_connection, init_db
from feature_extractor import get_image_features

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CORE_DIR)
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
DB_DIR = os.path.join(BASE_DIR, 'database')
KMEANS_MODEL_FILE = os.path.join(DB_DIR, 'kmeans_model.pkl')
IVF_INDEX_FILE = os.path.join(DB_DIR, 'ivf_index.pkl')

def build_index():
    """
    Quét thư mục train, trích xuất đặc trưng và lưu vào CSDL & file Pickle.
    """
    # Khởi tạo thư mục và CSDL
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    init_db()
    conn = get_connection()
    if not conn:
        print("Không thể kết nối CSDL.")
        return
    cursor = conn.cursor()
    # Xóa dữ liệu cũ để tránh lỗi trùng lặp PK nếu chạy lại script
    print("Đang làm sạch dữ liệu cũ trong bảng Bird_Metadata...")
    cursor.execute("DELETE FROM Bird_Metadata")
    conn.commit()
    # Quét thư mục và xử lý ảnh
    total_processed = 0
    start_time = time.time()
    # Duyệt qua từng thư mục loài chim trong thư mục train
    for species_folder in os.listdir(TRAIN_DIR):
        species_path = os.path.join(TRAIN_DIR, species_folder)
        if not os.path.isdir(species_path):
            continue
        species_label = species_folder # Tên thư mục chính là nhãn loài chim
        # Duyệt qua từng file ảnh trong thư mục
        for img_file in os.listdir(species_path):
            if not img_file.lower().endswith('.png'):
                continue
            file_path = os.path.join(species_path, img_file)
            # Lấy tên file làm Image_ID (Bỏ đuôi .png) - VD: "AFRICAN OYSTER CATCHER (1)"
            image_id = os.path.splitext(img_file)[0] 
            # Gọi hàm trích xuất với trọng số đặc trưng
            vector = get_image_features(file_path)
            if vector is not None:
                # Lưu siêu dữ liệu vào SQL Server
                vector_bytes = pickle.dumps(vector) # Chuyển vector thành nhị phân
                cursor.execute("""
                    INSERT INTO Bird_Metadata (Image_ID, Species_Label, File_Path, Feature_Vector) 
                    VALUES (?, ?, ?, ?)
                """, (image_id, species_label, file_path, vector_bytes))
                total_processed += 1
    # Lưu thay đổi vào CSDL
    conn.commit()
    cursor.close()
    conn.close()
    # Tổng kết
    end_time = time.time()
    print(f"Tổng số ảnh đã xử lý: {total_processed}")
    print(f"Thời gian thực hiện: {round(end_time - start_time, 2)} giây")

def add_image_to_db(file_path, species_label, distance_threshold=0.8):
    """
    Thêm một ảnh mới vào CSDL, cập nhật file features.pkl và phân cụm IVF.
    Nếu khoảng cách đến cụm gần nhất lớn hơn distance_threshold, tạo cụm mới.
    """
    # Kiểm tra file và tạo ID
    if not os.path.exists(file_path):
        print("Lỗi: Không tìm thấy file ảnh.")
        return
    img_file = os.path.basename(file_path)
    image_id = os.path.splitext(img_file)[0]
    # Trích xuất đặc trưng
    vector = get_image_features(file_path)
    if vector is None:
        print("Lỗi: Không thể trích xuất đặc trưng ảnh.")
        return
    # Lưu vào SQL Server
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        try:
            vector_bytes = pickle.dumps(vector)
            cursor.execute("""
                INSERT INTO Bird_Metadata (Image_ID, Species_Label, File_Path, Feature_Vector) 
                VALUES (?, ?, ?, ?)
            """, (image_id, species_label, file_path, vector_bytes))
            conn.commit()
        except Exception as e:
            print(f"Cảnh báo SQL: {e}")
        finally:
            cursor.close()
            conn.close()
    else:
        features_dict = {}
    features_dict[image_id] = vector
    # Phân cụm và cập nhật IVF Index
    if os.path.exists(KMEANS_MODEL_FILE) and os.path.exists(IVF_INDEX_FILE):
        with open(KMEANS_MODEL_FILE, 'rb') as f:
            kmeans_model = pickle.load(f)
        with open(IVF_INDEX_FILE, 'rb') as f:
            ivf_index = pickle.load(f)
        # Tính khoảng cách đến các tâm cụm hiện tại
        query_vec_2d = vector.reshape(1, -1)
        distances = kmeans_model.transform(query_vec_2d)[0]
        min_dist = np.min(distances)
        nearest_cluster = np.argmin(distances)
        # Đánh giá tạo cụm mới hay thêm vào cụm cũ
        if min_dist > distance_threshold:
            # Tạo cụm mới bằng cách thêm tâm cụm vào mảng cluster_centers_
            new_cluster_id = len(kmeans_model.cluster_centers_)
            kmeans_model.cluster_centers_ = np.vstack([kmeans_model.cluster_centers_, vector])
            ivf_index[new_cluster_id] = [image_id]
            print(f"Tạo cụm (ID: {new_cluster_id}) do khoảng cách ({min_dist:.4f}) vượt ngưỡng {distance_threshold}.")
        else:
            # Thêm vào cụm gần nhất
            if image_id not in ivf_index[nearest_cluster]:
                ivf_index[nearest_cluster].append(image_id)
            print(f"Thêm ảnh vào Cụm {nearest_cluster} (Khoảng cách: {min_dist:.4f}).")
        # Lưu lại mô hình và index
        with open(KMEANS_MODEL_FILE, 'wb') as f:
            pickle.dump(kmeans_model, f)
        with open(IVF_INDEX_FILE, 'wb') as f:
            pickle.dump(ivf_index, f)
    else:
        print("Không tìm thấy file K-Means/IVF.")

if __name__ == "__main__":
    # Chạy lần đầu
    build_index()
    # Thêm ảnh thủ công
"""
    test_new_image_path = os.path.join(TRAIN_DIR, "NEW_BIRD_SPECIES", "bird_001.png")
    add_image_to_db(test_new_image_path, species_label="NEW_BIRD_SPECIES", distance_threshold=0.8)
"""