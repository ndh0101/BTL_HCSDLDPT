import os
import sys
import time
# Thêm thư mục 'core' vào đường dẫn hệ thống để import các module
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
from feature_extractor import get_image_features
from searcher import BirdSearcher

# CẤU HÌNH ĐƯỜNG DẪN
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, 'test')

def evaluate_system(k=5, n_probe=1):
    """
    Đánh giá độ chính xác của hệ thống (K-Means + IVF) trên tập Test.
    """
    # Khởi tạo bộ tìm kiếm (Sẽ tự động nạp K-Means Model, IVF Index và Vector DB)
    searcher = BirdSearcher()
    if getattr(searcher, 'features_db', None) is None:
        print("Không thể tải CSDL vector. Dừng đánh giá.")
        return
    if not os.path.exists(TEST_DIR):
        print(f"Không tìm thấy thư mục test tại: {TEST_DIR}")
        return
    total_queries = 0
    total_precision = 0.0
    total_time = 0.0
    # Dictionary lưu trữ độ chính xác theo từng loài
    species_metrics = {}
    # Duyệt qua từng thư mục loài chim trong tập Test
    for species_folder in os.listdir(TEST_DIR):
        species_path = os.path.join(TEST_DIR, species_folder)
        if not os.path.isdir(species_path):
            continue
        true_species = species_folder
        species_precision_sum = 0.0
        species_query_count = 0
        # Duyệt qua từng ảnh test của loài đó
        for img_file in os.listdir(species_path):
            # Hỗ trợ cả định dạng png và jpg
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(species_path, img_file)
            # 1. Trích xuất đặc trưng (Kết hợp HOG, HSV, LBP)
            query_vec = get_image_features(img_path, alpha=0.5, beta=0.4, gamma=0.1)
            if query_vec is None:
                continue
            total_queries += 1
            species_query_count += 1
            # Bắt đầu bấm giờ truy vấn
            start_time = time.time()
            # 2. Tìm kiếm Top K với kỹ thuật phân cụm IVF
            results = searcher.search(query_vec, k=k, n_probe=n_probe)
            # Dừng bấm giờ
            query_time = time.time() - start_time
            total_time += query_time
            # 3. Tính Precision@K cho ảnh hiện tại
            # Đếm số lượng ảnh trong Top K có nhãn giống với true_species
            correct_hits = sum(1 for res in results if res['species'] == true_species)
            precision_at_k = correct_hits / k
            total_precision += precision_at_k
            species_precision_sum += precision_at_k
        # Lưu thống kê tổng hợp cho loài hiện tại
        if species_query_count > 0:
            species_metrics[true_species] = {
                'avg_precision': species_precision_sum / species_query_count,
                'count': species_query_count
            }
    # IN BÁO CÁO TỔNG KẾT
    if total_queries == 0:
        print("Không có ảnh nào trong tập test để đánh giá.")
        return
    # mAP (Mean Average Precision) cho tập k=5
    mAP = (total_precision / total_queries) * 100
    avg_query_time = (total_time / total_queries) * 1000  # Đổi sang mili-giây
    print(f"{'Tên loài chim':<30} | {'Số ảnh Test':<12} | {'Precision@5':<10}")
    for species, metrics in species_metrics.items():
        acc_str = f"{metrics['avg_precision']*100:.2f}%"
        print(f"{species:<30} | {metrics['count']:<12} | {acc_str:<10}")
    print(f"Tổng số ảnh truy vấn (Test) : {total_queries} ảnh")
    print(f"Độ chính xác trung bình (mAP): {mAP:.2f}%")
    print(f"Thời gian truy vấn trung bình: {avg_query_time:.2f} ms / ảnh")

if __name__ == "__main__":
    # Khởi chạy đánh giá với Top 5 và mở rộng tìm kiếm trên cụm lân cận
    evaluate_system(k=5, n_probe=1)