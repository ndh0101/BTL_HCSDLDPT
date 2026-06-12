import os
import pickle
import numpy as np
import time
from db_connection import get_connection

# CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CORE_DIR)
DB_DIR = os.path.join(BASE_DIR, 'database')
KMEANS_MODEL_FILE = os.path.join(DB_DIR, 'kmeans_model.pkl')
IVF_INDEX_FILE = os.path.join(DB_DIR, 'ivf_index.pkl')

class BirdSearcher:
    def __init__(self):
        """Khởi tạo bộ tìm kiếm: Tải Vector, Mô hình K-Means và Chỉ mục IVF vào RAM."""
        try:
            # Tải bộ vector đặc trưng từ SQL Server
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT Image_ID, Feature_Vector FROM Bird_Metadata")
            self.features_db = {row[0]: pickle.loads(row[1]) for row in cursor.fetchall()}
            cursor.close()
            conn.close()
            # Tải Mô hình K-Means (Chứa thông tin các Tâm cụm)
            with open(KMEANS_MODEL_FILE, 'rb') as f:
                self.kmeans_model = pickle.load(f)
            # Tải Chỉ mục ngược IVF (Inverted File Index)
            with open(IVF_INDEX_FILE, 'rb') as f:
                self.ivf_index = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Lỗi: Không tìm thấy file dữ liệu. Chi tiết: {e}")
            self.features_db = None
            self.kmeans_model = None
            self.ivf_index = None

    def search(self, query_vector, k=5, n_probe=1):
        """
        Tìm kiếm Top K ảnh tương đồng sử dụng K-Means và IVF.
        :param query_vector: Vector đặc trưng của ảnh đầu vào (1D Array).
        :param k: Số lượng ảnh trả về.
        :param n_probe: Số lượng cụm lân cận sẽ được mở ra để tìm kiếm.
        """
        if not self.kmeans_model or not self.ivf_index or not self.features_db:
            print("Hệ thống chưa được khởi tạo đúng cách.")
            return []
        # XÁC ĐỊNH CỤM GẦN NHẤT (ROUTING)
        # Đưa vector 1D thành ma trận 2D (1 x D) để tương thích với thư viện scikit-learn
        query_vec_2d = query_vector.reshape(1, -1)
        # Hàm transform() tính khoảng cách từ vector truy vấn đến TẤT CẢ các tâm cụm
        distances_to_clusters = self.kmeans_model.transform(query_vec_2d)[0]
        # Lấy ra ID của `n_probe` cụm có khoảng cách nhỏ nhất (Ví dụ: 2 cụm gần nhất)
        nearest_clusters = np.argsort(distances_to_clusters)[:n_probe]

        # THU HẸP KHÔNG GIAN TÌM KIẾM (CANDIDATE SELECTION)
        candidate_ids = []
        for cluster_id in nearest_clusters:
            # Tra cứu từ điển IVF để lấy ID các ảnh nằm trong cụm đó
            candidate_ids.extend(self.ivf_index[cluster_id])
        # Loại bỏ các ID trùng lặp (nếu có)
        candidate_ids = list(set(candidate_ids))
        # In log ra terminal để quan sát sự tối ưu hóa không gian tìm kiếm
        print(f"[IVF Search] Mở {n_probe} cụm lân cận gần nhất: {nearest_clusters}")

        # TÍNH KHOẢNG CÁCH TRONG CỤM
        results = []
        # Vì vector đã chuẩn hóa L2, nên np.linalg.norm xấp xỉ 1.
        # Chuẩn hóa lại ở mẫu số để đảm bảo triệt tiêu hoàn toàn sai số dấu phẩy động
        query_norm = np.linalg.norm(query_vector)
        for img_id in candidate_ids:
            if img_id not in self.features_db: continue
            db_vector = self.features_db[img_id]
            # Tính độ tương đồng Cosine
            db_norm = np.linalg.norm(db_vector)
            cosine_similarity = np.dot(query_vector, db_vector) / ((query_norm * db_norm) + 1e-7)
            # Chuyển thành khoảng cách
            distance = 1.0 - cosine_similarity
            distance = max(0.0, distance)
            results.append((img_id, distance))
        # Sắp xếp tăng dần theo khoảng cách (Khoảng cách càng nhỏ càng giống)
        results.sort(key=lambda x: x[1])
        top_k_results = results[:k]
        # LẤY METADATA TỪ SQL SERVER
        return self._fetch_metadata_from_sql(top_k_results)

    def _fetch_metadata_from_sql(self, top_k_results):
        """Truy vấn SQL Server để lấy đường dẫn và tên loài chim cho Top K ID."""
        conn = get_connection()
        if not conn:
            print("Không thể kết nối SQL Server để lấy Metadata.")
            return []
        top_k_ids = [item[0] for item in top_k_results]
        if not top_k_ids:
            return []
        placeholders = ','.join(['?'] * len(top_k_ids))
        query = f"SELECT Image_ID, Species_Label, File_Path FROM Bird_Metadata WHERE Image_ID IN ({placeholders})"
        cursor = conn.cursor()
        cursor.execute(query, top_k_ids)
        rows = cursor.fetchall()
        # Gắn metadata vào dictionary
        db_data = {row[0]: {'label': row[1], 'path': row[2]} for row in rows}
        # Tạo danh sách kết quả cuối cùng (Giữ đúng thứ tự đã xếp hạng của thuật toán)
        final_results = []
        for image_id, dist in top_k_results:
            if image_id in db_data:
                final_results.append({
                    'image_id': image_id,
                    'distance': round(dist, 4),
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
    # Ví dụ: Copy 1 ảnh vào thư mục core và đổi tên thành test_query.png
    test_image_path = "core/test_query.png" 
    if os.path.exists(test_image_path):
        print(f"Trích xuất đặc trưng cho ảnh truy vấn: {test_image_path}")
        # Các trọng số sẽ được dùng mặc định từ hàm get_image_features
        query_vec = get_image_features(test_image_path)
        if query_vec is not None:
            print("Bắt đầu đối sánh bằng cơ chế K-Means + IVF...")
            # Sử dụng tham số k=5 và tìm kiếm mở rộng n_probe=2
            results = searcher.search(query_vec, k=5, n_probe=2)
            for i, res in enumerate(results, 1):
                print(f"Top {i}: {res['species']} (ID: {res['image_id']})")
                print(f"  + Khoảng cách Cosine: {res['distance']}")
                print(f"  + Đường dẫn: {res['file_path']}\n")
    else:
        print("Hãy copy 1 file ảnh chim bất kỳ vào thư mục core, đổi tên thành 'test_query.png' để chạy thử luồng tìm kiếm mới này.")