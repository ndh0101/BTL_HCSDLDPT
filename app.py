import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import exposure
from skimage.feature import hog, local_binary_pattern
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
from core.searcher import BirdSearcher
from core.feature_extractor import get_image_features, read_image, rgb_to_grayscale, rgb_to_hsv, create_background_mask

class CBIRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Nhận dạng và Tìm kiếm Ảnh Chim - CBIR (K-Means + IVF)")
        self.root.geometry("1350x900")
        self.root.configure(bg="#f0f0f0")
        self.query_image_path = None
        self.img_refs = []  # Lưu reference của ảnh để tránh bị Garbage Collector xóa mất
        # Khởi tạo bộ tìm kiếm
        try:
            self.searcher = BirdSearcher()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải CSDL vector: {e}")
            self.searcher = None
        self.setup_ui()

    def setup_ui(self):
        """Thiết kế bố cục giao diện"""
        # FRAME TRÊN (Điều khiển, Lược đồ)
        top_frame = tk.Frame(self.root, bg="#f0f0f0")
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        # Ảnh đầu vào & Nút bấm & Thông tin Vector
        left_col = tk.LabelFrame(top_frame, text="1. Đầu vào & Thông số", bg="#f0f0f0", font=("Arial", 10, "bold"))
        left_col.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        btn_select = tk.Button(left_col, text="Chọn ảnh", command=self.load_image, width=20, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        btn_select.pack(pady=5)
        self.lbl_query_img = tk.Label(left_col, bg="white", width=224, height=224)
        self.lbl_query_img.pack(padx=10, pady=5)
        btn_search = tk.Button(left_col, text="Tìm ảnh tương đồng", command=self.search_image, width=20, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        btn_search.pack(pady=5)
        # Khung hiển thị thông tin Vector đặc trưng
        self.lbl_vector_info = tk.Label(left_col, text="Thông tin Vector:\n(Chưa trích xuất)", bg="#e8f5e9", fg="#2e7d32", font=("Courier New", 9), justify=tk.LEFT, width=32, height=8, relief=tk.RIDGE)
        self.lbl_vector_info.pack(padx=10, pady=5)
        # Các bước trung gian (Xám, HSV, Mask, HOG, LBP)
        mid_col = tk.LabelFrame(top_frame, text="2. Các bước trích xuất (Tiền xử lý)", bg="#f0f0f0", font=("Arial", 10, "bold"))
        mid_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        # Tạo lưới 2x3
        self.lbl_gray = self.create_intermediate_label(mid_col, "Ảnh Xám", 0, 0)
        self.lbl_hsv = self.create_intermediate_label(mid_col, "Không gian HSV", 0, 1)
        self.lbl_mask = self.create_intermediate_label(mid_col, "Mặt nạ (Mask)", 0, 2)
        self.lbl_hog = self.create_intermediate_label(mid_col, "Đặc trưng HOG", 1, 0)
        self.lbl_lbp = self.create_intermediate_label(mid_col, "Kết cấu LBP (Masked)", 1, 1)
        # Biểu đồ Histogram (Màu sắc & Kết cấu)
        right_col = tk.LabelFrame(top_frame, text="3. Đặc trưng (Màu sắc & Kết cấu)", bg="#f0f0f0", font=("Arial", 10, "bold"))
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.hist_frame = tk.Frame(right_col, bg="white", width=400, height=450)
        self.hist_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # FRAME DƯỚI (Kết quả Top 5)
        bottom_frame = tk.LabelFrame(self.root, text="4. Kết quả truy xuất: Top 5 ảnh tương đồng nhất", bg="#f0f0f0", font=("Arial", 10, "bold"))
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.result_frames = []
        for i in range(5):
            frame = tk.Frame(bottom_frame, bg="white", bd=2, relief=tk.GROOVE)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            lbl_img = tk.Label(frame, bg="white")
            lbl_img.pack(pady=5)
            lbl_info = tk.Label(frame, text=f"Top {i+1}", bg="white", justify=tk.CENTER, font=("Arial", 9))
            lbl_info.pack(pady=5)
            self.result_frames.append((lbl_img, lbl_info))

    def create_intermediate_label(self, parent, text, row, col):
        """Tạo khung chứa ảnh trung gian"""
        frame = tk.Frame(parent, bg="#f0f0f0")
        frame.grid(row=row, column=col, padx=10, pady=5)
        tk.Label(frame, text=text, bg="#f0f0f0", font=("Arial", 9)).pack()
        lbl_img = tk.Label(frame, bg="white", width=120, height=120)
        lbl_img.pack()
        return lbl_img

    def load_image(self):
        """Xử lý sự kiện Chọn Ảnh"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh chim",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            self.query_image_path = file_path
            img_rgb = read_image(file_path)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.lbl_query_img.config(image=img_tk)
            self.lbl_query_img.image = img_tk
            # Khôi phục trạng thái UI ban đầu
            self.lbl_vector_info.config(text="Thông tin Vector:\n(Đang chờ...)")
            self.clear_results()

    def clear_results(self):
        empty_img = ImageTk.PhotoImage(Image.new("RGB", (120, 120), "white"))
        for lbl in [self.lbl_gray, self.lbl_hsv, self.lbl_mask, self.lbl_hog, self.lbl_lbp]:
            lbl.config(image=empty_img)
            lbl.image = empty_img
        for lbl_img, lbl_info in self.result_frames:
            lbl_img.config(image='')
            lbl_info.config(text="")
        for widget in self.hist_frame.winfo_children():
            widget.destroy()

    def search_image(self):
        if not self.query_image_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một bức ảnh trước!")
            return
        if not self.searcher:
            messagebox.showerror("Lỗi", "Hệ thống CSDL chưa sẵn sàng.")
            return
        # Tạo các ảnh trung gian
        img_rgb = read_image(self.query_image_path)
        img_gray = rgb_to_grayscale(img_rgb)
        img_hsv = rgb_to_hsv(img_rgb)
        mask = create_background_mask(img_gray, threshold=240)
        # Tạo HOG trực quan
        _, hog_img = hog(img_gray, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualize=True)
        hog_img_uint8 = (exposure.rescale_intensity(hog_img, in_range=(0, 10)) * 255).astype(np.uint8)
        # Tạo LBP trực quan và Mask
        lbp_image = local_binary_pattern(img_gray, 8, 1, method='default')
        lbp_image_uint8 = lbp_image.astype(np.uint8)
        lbp_masked_display = cv2.bitwise_and(lbp_image_uint8, lbp_image_uint8, mask=mask)
        # Hiển thị ảnh trung gian
        self.display_intermediate(self.lbl_gray, img_gray, is_gray=True)
        self.display_intermediate(self.lbl_hsv, img_hsv)
        self.display_intermediate(self.lbl_mask, mask, is_gray=True)
        self.display_intermediate(self.lbl_hog, hog_img_uint8, is_gray=True)
        self.display_intermediate(self.lbl_lbp, lbp_masked_display, is_gray=True)
        # Vẽ 2 Lược đồ Histogram
        self.draw_histograms(img_hsv, lbp_image, mask)
        # Trích xuất vector thực tế (Để tìm kiếm và hiển thị thông số)
        query_vec = get_image_features(self.query_image_path, alpha=0.5, beta=0.4, gamma=0.1)
        if query_vec is None:
            messagebox.showerror("Lỗi", "Không thể trích xuất đặc trưng từ ảnh này.")
            return
        # Hiển thị thông số Vector ra UI
        vec_size = query_vec.shape[0]
        # Lấy vài giá trị đầu (HOG) và cuối (LBP) để demo
        preview = f"Tổng số chiều: {vec_size}\n\n"
        preview += f"Đầu (HOG): [{query_vec[0]:.4f}, {query_vec[1]:.4f}...]\n"
        preview += f"Giữa (HSV): [...{query_vec[6084]:.4f}, {query_vec[6085]:.4f}...]\n"
        preview += f"Cuối (LBP): [...{query_vec[-2]:.4f}, {query_vec[-1]:.4f}]"
        self.lbl_vector_info.config(text=preview)
        # Tìm kiếm Top 5 qua IVF
        results = self.searcher.search(query_vec, k=5, metric='cosine')
        self.display_results(results)

    def display_intermediate(self, label, img_array, is_gray=False):
        img_resized = cv2.resize(img_array, (120, 120))
        if is_gray:
            img_pil = Image.fromarray(img_resized, 'L')
        else:
            img_pil = Image.fromarray(img_resized, 'RGB')
        img_tk = ImageTk.PhotoImage(img_pil)
        label.config(image=img_tk)
        self.img_refs.append(img_tk)

    def draw_histograms(self, img_hsv, lbp_image, mask):
        for widget in self.hist_frame.winfo_children():
            widget.destroy()
        # Tính HSV Histogram
        hsv_hist = cv2.calcHist([img_hsv], [0, 1, 2], mask, [16, 4, 3], [0, 180, 0, 256, 0, 256])
        hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()
        # Tính LBP Histogram (Chỉ lấy phần chim qua mask)
        lbp_bird_pixels = lbp_image[mask == 255]
        lbp_hist, _ = np.histogram(lbp_bird_pixels, bins=256, range=(0, 256))
        lbp_hist = lbp_hist.astype("float")
        if lbp_hist.sum() > 0: lbp_hist /= lbp_hist.sum()
        # Vẽ 2 đồ thị xếp chồng
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(4, 4.5), dpi=100)
        # Biểu đồ HSV (Màu sắc)
        ax1.plot(hsv_hist, color='blue', linewidth=1.2)
        ax1.set_title('HSV Color Histogram (Màu sắc)', fontsize=9)
        ax1.tick_params(axis='both', which='major', labelsize=7)
        # Biểu đồ LBP (Kết cấu)
        ax2.bar(np.arange(0, 256), lbp_hist, width=1, color='green')
        ax2.set_title('LBP Texture Histogram (Kết cấu)', fontsize=9)
        ax2.tick_params(axis='both', which='major', labelsize=7)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.hist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def display_results(self, results):
        for i, (res, frame_tuple) in enumerate(zip(results, self.result_frames)):
            lbl_img, lbl_info = frame_tuple
            path = res['file_path']
            try:
                img_array = read_image(path)
                img_resized = cv2.resize(img_array, (150, 150))
                img_pil = Image.fromarray(img_resized)
                img_tk = ImageTk.PhotoImage(img_pil)
                lbl_img.config(image=img_tk)
                self.img_refs.append(img_tk)
                distance = res['distance']
                similarity = (1.0 - distance) * 100
                if similarity < 0: similarity = 0
                info_text = f"Top {i+1}\nLoài: {res['species']}\nTương đồng: {similarity:.2f}%"
                lbl_info.config(text=info_text, fg="blue" if i==0 else "black", font=("Arial", 9, "bold" if i==0 else "normal"))
            except Exception as e:
                lbl_info.config(text=f"Lỗi tải ảnh {i+1}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CBIRApp(root)
    root.mainloop()