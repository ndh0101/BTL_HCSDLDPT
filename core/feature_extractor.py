import cv2
import math
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

# Hàm tiền xử lý ảnh
def read_image(file_path):
    """Đọc ảnh từ đường dẫn và chuyển đổi từ (OpenCV) sang RGB."""
    img_array = np.fromfile(file_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh tại đường dẫn: {file_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img_rgb.shape[:2] != (224, 224):
        img_rgb = cv2.resize(img_rgb, (224, 224))
    return img_rgb

def rgb_to_grayscale(img_rgb):
    """Chuyển đổi ảnh RGB sang ảnh Xám (Grayscale)."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

def rgb_to_hsv(img_rgb):
    """Chuyển đổi ảnh RGB sang không gian màu HSV."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

def create_background_mask(img_gray, threshold=240):
    """
    Tạo mặt nạ (Mask) để loại bỏ nền trắng.
    Pixel nào có cường độ >= 240 (gần trắng) sẽ bị loại bỏ (Mask = 0).
    Pixel thuộc về chim sẽ được giữ lại (Mask = 255).
    """
    _, mask = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask

# Hàm trích xuất đặc trưng
def extract_hog_features(img_gray):
    
    """Trích xuất vector đặc trưng hình dạng HOG."""
    hog_vector = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys', 
        visualize=False,
        feature_vector=True
    )
    hog_vector = hog_vector / (np.linalg.norm(hog_vector) + 1e-7)
    return hog_vector

def extract_hsv_histogram(img_hsv, mask, bins=(16, 4, 3)):
    """Trích xuất Color Histogram trên hệ màu HSV."""
    hist = cv2.calcHist(
        [img_hsv], 
        [0, 1, 2], 
        mask, 
        [bins[0], bins[1], bins[2]], 
        [0, 180, 0, 256, 0, 256]
    )
    hist_vector = hist.flatten()
    cv2.normalize(hist_vector, hist_vector, alpha=1, norm_type=cv2.NORM_L1)
    return hist_vector

def extract_lbp_features(img_gray, mask, radius=1):
    """
    Trích xuất đặc trưng kết cấu (Texture) bằng LBP, sử dụng Mask khử nền.
    Đầu ra: Vector kích thước 256 chiều.
    """
    n_points = 8 * radius
    lbp_image = local_binary_pattern(img_gray, n_points, radius, method='default')
    # Lọc qua mặt nạ
    lbp_bird_pixels = lbp_image[mask == 255]
    # Tính Histogram với 256 bins
    hist, _ = np.histogram(lbp_bird_pixels, bins=256, range=(0, 256))
    # Chuẩn hóa L1 cho LBP
    hist_vector = hist.astype("float")
    if hist_vector.sum() > 0:
        hist_vector /= (hist_vector.sum() + 1e-7)
    return hist_vector

def extract_hu_moments(mask):
    """
    Trích xuất đặc trưng Hình dạng toàn cục (Global Shape) bằng Hu Moments.
    Đầu ra: Vector kích thước 7 chiều.
    """
    # Tính toán các moment không gian (Spatial Moments)
    moments = cv2.moments(mask)
    # Tính 7 giá trị bất biến Hu Moments
    hu_moments = cv2.HuMoments(moments)
    # Trải phẳng mảng 2D thành vector 1D
    hu_vector = hu_moments.flatten()
    # Chuyển đổi Logarit (Log Scale Transform) để cân bằng biên độ
    for i in range(len(hu_vector)):
        # Áp dụng log10 để giảm kích thước con số, giữ nguyên dấu âm/dương
        if hu_vector[i] != 0:
            hu_vector[i] = -1 * math.copysign(1.0, hu_vector[i]) * math.log10(abs(hu_vector[i]))
    # Chuẩn hóa L2 để đồng bộ thang đo với HOG
    hu_vector = hu_vector / (np.linalg.norm(hu_vector) + 1e-7)
    return hu_vector

def extract_glcm_features(img_gray, mask):
    """
    Trích xuất đặc trưng Kết cấu thống kê bằng Ma trận đồng mức xám (GLCM).
    Đầu ra: Vector kích thước 16 chiều.
    """
    # Áp dụng Mask, đưa phông nền trắng về màu đen (giá trị 0)
    masked_gray = cv2.bitwise_and(img_gray, img_gray, mask=mask)
    # Khởi tạo GLCM với khoảng cách d=1 và 4 hướng (0, 45, 90, 135 độ)
    # Normed = False để tự chuẩn hóa sau khi cắt nền
    glcm = graycomatrix(masked_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=False)
    # Cắt bỏ nhiễu từ phông nền (Loại bỏ hàng 0 và cột 0)
    glcm_bird_only = glcm[1:, 1:, :, :]
    # Chuẩn hóa lại ma trận thành xác suất
    glcm_bird_only = glcm_bird_only.astype('float')
    for d in range(glcm_bird_only.shape[2]):
        for a in range(glcm_bird_only.shape[3]):
            sum_val = glcm_bird_only[:, :, d, a].sum()
            if sum_val > 0:
                glcm_bird_only[:, :, d, a] /= sum_val
    # Trích xuất 4 thuộc tính thống kê (mỗi thuộc tính sinh ra 4 giá trị do có 4 góc)
    contrast = graycoprops(glcm_bird_only, 'contrast').flatten()
    dissimilarity = graycoprops(glcm_bird_only, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm_bird_only, 'homogeneity').flatten()
    energy = graycoprops(glcm_bird_only, 'energy').flatten()
    # Ghép nối (Concatenate) thành vector 16 chiều và Chuẩn hóa L2
    glcm_vector = np.concatenate([contrast, dissimilarity, homogeneity, energy])
    glcm_vector = glcm_vector / (np.linalg.norm(glcm_vector) + 1e-7)
    return glcm_vector

# Hàm kết hợp đặc trưng (Feature Fusion)
def feature_fusion(hog_vec, hsv_vec, lbp_vec, glcm_vec, hu_vec, alpha=0.2, beta=0.2,
                    gamma=0.2, delta=0.2, epsilon=0.2):
    """
    Kết hợp các vector với trọng số tương ứng.
    """
    # Đưa HSV và LBP về chuẩn L2 để đồng bộ thang đo với HOG
    hsv_vec_l2 = hsv_vec / (np.linalg.norm(hsv_vec) + 1e-7)
    lbp_vec_l2 = lbp_vec / (np.linalg.norm(lbp_vec) + 1e-7)
    # Gán trọng số
    weighted_hog = alpha * hog_vec
    weighted_hsv = beta * hsv_vec_l2
    weighted_lbp = gamma * lbp_vec_l2
    weighted_glcm = delta * glcm_vec
    weighted_hu = epsilon * hu_vec
    # Nối vector (Concatenation)
    combined_vec = np.concatenate([weighted_hog, weighted_hsv, weighted_lbp, weighted_glcm, weighted_hu])
    # Chuẩn hóa L2 lần cuối cho vector tổng hợp
    final_vector = combined_vec / (np.linalg.norm(combined_vec) + 1e-7)
    return final_vector

# Hàm chính để trích xuất đặc trưng từ ảnh
def get_image_features(file_path, alpha=0.2, beta=0.2, gamma=0.2, delta=0.2, epsilon=0.2):
    """
    Hàm đóng gói toàn bộ quy trình: Đọc ảnh -> Tiền xử lý -> Trích xuất -> Kết hợp.
    """
    try:
        # Đọc và chuyển đổi ảnh
        img_rgb = read_image(file_path)
        img_gray = rgb_to_grayscale(img_rgb)
        img_hsv = rgb_to_hsv(img_rgb)
        # Tạo mặt nạ khử nền
        mask = create_background_mask(img_gray, threshold=240)
        # Trích xuất đặc trưng
        hog_features = extract_hog_features(img_gray)
        hsv_features = extract_hsv_histogram(img_hsv, mask)
        lbp_features = extract_lbp_features(img_gray, mask)
        glcm_features = extract_glcm_features(img_gray, mask)
        hu_features = extract_hu_moments(mask)
        # Kết hợp vector
        final_feature_vector = feature_fusion(hog_features, hsv_features, lbp_features, glcm_features,
                                            hu_features, alpha, beta, gamma, delta, epsilon)
        return final_feature_vector
    except Exception as e:
        print(f"Lỗi khi trích xuất đặc trưng cho ảnh {file_path}: {e}")
        return None

# Code chạy thử
if __name__ == "__main__":
    # Thay đường dẫn này bằng 1 ảnh thật trong máy để test
    test_image_path = "train/AFRICAN OYSTER CATCHER/AFRICAN OYSTER CATCHER (2).png" 
    import os
    if os.path.exists(test_image_path):
        vector = get_image_features(test_image_path)
        print(f"Trích xuất thành công! Kích thước vector tổng hợp: {vector.shape}")
    else:
        print("Vui lòng cung cấp một file ảnh mẫu hợp lệ để test.")