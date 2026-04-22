import os
import pickle
import time
from db_connection import get_connection, init_db
from feature_extractor import get_image_features

# CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CORE_DIR)
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
DB_DIR = os.path.join(BASE_DIR, 'database')
FEATURE_FILE = os.path.join(DB_DIR, 'features.pkl')

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
    features_dict = {} # Dictionary lưu trữ: {Image_ID: Feature_Vector}
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
            vector = get_image_features(file_path, alpha=0.5, beta=0.4, gamma=0.1)
            if vector is not None:
                # Lưu vector vào dictionary
                features_dict[image_id] = vector
                # Lưu siêu dữ liệu vào SQL Server
                cursor.execute("""
                    INSERT INTO Bird_Metadata (Image_ID, Species_Label, File_Path)
                    VALUES (?, ?, ?)
                """, (image_id, species_label, file_path))
                total_processed += 1
    # Lưu thay đổi vào CSDL
    conn.commit()
    cursor.close()
    conn.close()
    # Lưu mảng vector ra file Pickle
    with open(FEATURE_FILE, 'wb') as f:
        pickle.dump(features_dict, f)
    # Tổng kết
    end_time = time.time()
    print(f"Tổng số ảnh đã xử lý: {total_processed}")
    print(f"Thời gian thực hiện: {round(end_time - start_time, 2)} giây")
    print(f"File vector lưu tại: {FEATURE_FILE}")

if __name__ == "__main__":
    build_index()