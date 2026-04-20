# Flood Image Analysis Pipeline

Phân tích ảnh lũ lụt tự động: đo mực nước, phân loại mức độ, tạo báo cáo.

**Stack:** Depth Anything V2 · YOLOv8 · DINOv2 · SegFormer

---

## Cài đặt nhanh

```bash
git clone <repo>
cd flood_pipeline
pip install -r requirements.txt
python main.py
```

Lần đầu chạy sẽ tự động cài thư viện và tải model (~500MB).

---

## Chạy pipeline

### Chế độ tương tác (mặc định)
```bash
python main.py
```
Hỏi từng bước, nhập `b` để quay lại bước trước.

### Chế độ tự động (không cần tương tác)
Sửa `config.yaml`:
```yaml
auto_skip:
  enable: true
  mode: local
  input_path: "C:/anh_lu"
  skip_drive: true
```
Rồi chạy: `python main.py`

### Chạy từ folder ảnh cụ thể
```bash
# Sửa config.yaml rồi chạy, hoặc chọn "Folder LOCAL" trong menu
python main.py
```

---

## Cấu trúc thư mục

```
flood_pipeline/
├── main.py                  # Entry point chính
├── config.yaml              # Cấu hình pipeline
├── setup_env.py             # Cài đặt thư viện + model lần đầu
├── learning_update.py       # CLI quản lý self-learning
│
├── crawlers/                # Tải ảnh từ Bing/Google/Facebook
├── filters/                 # Lọc ảnh: blur, dedup, watermark, enhance
├── depth_analysis/          # Phân tích depth + đo mực nước
├── learning/                # Self-learning: review queue, adaptive thresholds
├── uploader/                # Upload lên Google Drive
└── utils/                   # Config loader, báo cáo, location, constants
```

---

## Output

Kết quả lưu tại `output/<timestamp>/`:
```
output/20240115_143022/
├── 00_raw/                  # Ảnh gốc tải về
├── original_images/         # Ảnh sau khi lọc
├── depth_overlays/          # Ảnh có overlay mực nước
├── depth_maps/              # Depth map grayscale
├── watermark_cleaned/       # Ảnh đã xóa watermark
├── flood_analysis_report.html
├── flood_analysis_report.csv
├── flood_analysis_report.xlsx
└── pipeline_summary.json
```

---

## Self-Learning

```bash
# Xem thống kê
python learning_update.py --stats

# Cập nhật thresholds từ feedback
python learning_update.py --update

# Mở UI review ảnh uncertain
python learning/review_ui.py

# Export training data
python learning_update.py --export
```

---

## Flood Levels

| Level     | Mực nước  | Ý nghĩa                      |
|-----------|-----------|-------------------------------|
| PUDDLE    | 0–15 cm   | Vũng nước nhỏ                 |
| ANKLE     | 15–40 cm  | Ngập mắt cá chân              |
| KNEE      | 40–70 cm  | Ngập đầu gối                  |
| WAIST     | 70–120 cm | Ngập ngang hông               |
| CHEST     | 120–200 cm| Ngập ngang ngực               |
| SUBMERGED | >200 cm   | Ngập hoàn toàn                |

---

## Yêu cầu

- Python 3.10+
- RAM: 8GB+ (khuyến nghị 16GB)
- GPU: CUDA khuyến nghị (chạy được trên CPU, chậm hơn)
- Disk: ~2GB cho models
