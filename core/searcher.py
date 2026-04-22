import os
import pickle
import numpy as np
from db_connection import get_connection

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CORE_DIR)
DB_DIR = os.path.join(BASE_DIR, 'database')
FEATURE_FILE = os.path.join(DB_DIR, 'features.pkl')

class BirdSearcher:
    def __init__(self, feature_path=FEATURE_FILE):
        """Khởi tạo Searcher và tải file vector lên bộ nhớ RAM để tìm kiếm nhanh."""
        self.feature_path = feature_path
        self.features_db = self._load_features()

    def _load_features(self):
        """Đọc file features.pkl chứa dictionary {Image_ID: Vector}."""
        if not os.path.exists(self.feature_path):
            print(f"Không tìm thấy file dữ liệu tại {self.feature_path}")
            return {}
        print("Đang tải cơ sở dữ liệu vector lên bộ nhớ...")
        with open(self.feature_path, 'rb') as f:
            features = pickle.load(f)
        return features

    def calculate_distance(self, query_vec, db_vec, metric='cosine'):
        """Tính toán khoảng cách giữa vector truy vấn và vector trong CSDL."""
        if metric == 'euclidean':
            # Khoảng cách Euclid (Càng nhỏ càng giống)
            return np.linalg.norm(query_vec - db_vec)
        elif metric == 'cosine':
            # Khoảng cách Cosine = 1 - Độ tương đồng Cosine (Càng nhỏ càng giống)
            similarity = np.dot(query_vec, db_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(db_vec) + 1e-7)
            return 1.0 - similarity
        else:
            raise ValueError("Chỉ hỗ trợ độ đo 'euclidean' hoặc 'cosine'.")

    def search(self, query_vector, k=5, metric=''):
        """
        Thuật toán K-NN: Tìm K ảnh giống nhất với ảnh đầu vào.
        - k: Số lượng kết quả trả về (k=5)
        - metric: 'euclidean' hoặc 'cosine'
        """
        if not self.features_db:
            return []
        # 1. Quét CSDL và tính khoảng cách
        distances = []
        for image_id, db_vector in self.features_db.items():
            dist = self.calculate_distance(query_vector, db_vector, metric=metric)
            distances.append((image_id, dist))
        # 2. Xếp hạng (Sắp xếp theo chiều tăng dần của khoảng cách)
        distances.sort(key=lambda x: x[1])
        # Lấy Top K kết quả tốt nhất
        top_k_results = distances[:k]
        # 3. Truy vấn SQL Server để lấy thông tin chi tiết
        return self._fetch_metadata_from_sql(top_k_results)

    def _fetch_metadata_from_sql(self, top_k_results):
        """Truy vấn SQL Server để lấy đường dẫn và tên loài chim cho Top K ID."""
        conn = get_connection()
        if not conn:
            print("Không thể kết nối SQL Server để lấy Metadata.")
            return []
        # Lấy danh sách ID từ Top K
        top_k_ids = [item[0] for item in top_k_results]
        # Tạo chuỗi tham số động cho lệnh IN (?, ?, ?, ?, ?)
        placeholders = ','.join(['?'] * len(top_k_ids))
        query = f"SELECT Image_ID, Species_Label, File_Path FROM Bird_Metadata WHERE Image_ID IN ({placeholders})"
        cursor = conn.cursor()
        cursor.execute(query, top_k_ids)
        rows = cursor.fetchall()
        # Lưu dữ liệu từ SQL vào dictionary để dễ dàng lấy ra
        db_data = {row[0]: {'label': row[1], 'path': row[2]} for row in rows}
        # 4. Gắn kết quả (Đảm bảo giữ đúng thứ tự đã xếp hạng của thuật toán K-NN)
        final_results = []
        for image_id, dist in top_k_results:
            if image_id in db_data:
                final_results.append({
                    'image_id': image_id,
                    'distance': round(dist, 4), # Làm tròn 4 chữ số thập phân
                    'species': db_data[image_id]['label'],
                    'file_path': db_data[image_id]['path']
                })
        cursor.close()
        conn.close()
        return final_results

# CODE CHẠY THỬ (TEST)
if __name__ == "__main__":
    from feature_extractor import get_image_features
    # Khởi tạo bộ tìm kiếm
    searcher = BirdSearcher()
    # Ví dụ: Copy 1 ảnh vào thư mục core và đổi tên thành test_query.png)
    test_image_path = "core/test_query.png" 
    if os.path.exists(test_image_path):
        print(f"\nTrích xuất đặc trưng cho ảnh truy vấn: {test_image_path}")
        query_vec = get_image_features(test_image_path)
        if query_vec is not None:
            print("\nTìm kiếm bằng thuật toán K-NN (K=5)...")
            results = searcher.search(query_vec, k=5, metric='cosine')
            print("\nKẾT QUẢ TOP 5:")
            for i, res in enumerate(results, 1):
                print(f"Top {i}: {res['species']} (ID: {res['image_id']}) - Khoảng cách: {res['distance']}")
                print(f"    => Đường dẫn: {res['file_path']}")
    else:
        print("\nHãy copy 1 file ảnh chim bất kỳ vào thư mục core, đổi tên thành 'test_query.png' để chạy thử tìm kiếm.")