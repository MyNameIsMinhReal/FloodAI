# Kiến trúc Pipeline — Flood Image Analysis

## Tổng quan

```
Input (ảnh)
 │
 ├─ Validation          utils/explainability.py, pipeline/orchestrator.py
 │   └─ Kiểm tra định dạng, kích thước, không rỗng
 │
 ├─ Stage 1: Analyze    pipeline/stages/analyze_stage.py
 │   └─ Phát hiện vị trí GPS từ EXIF / địa chỉ
 │
 ├─ Stage 2: Depth      pipeline/stages/depth_stage.py
 │   ├─ Depth Anything V2 → depth map
 │   ├─ YOLOv8 → object detection (vật tham chiếu)
 │   └─ DINOv2 → feature matching (optional)
 │
 ├─ Stage 3: Raincoat   pipeline/stages/raincoat_stage.py
 │   └─ Phát hiện áo mưa → xác nhận môi trường lũ
 │
 ├─ Stage 4: Postprocess pipeline/stages/postprocess_stage.py
 │   ├─ ConfidenceScorer (5-component formula)
 │   ├─ UncertaintyEvaluator → accept/review/reject
 │   ├─ DepthCalibrator → hiệu chỉnh chiều sâu
 │   └─ FloodAlerter → gửi cảnh báo nếu nguy hiểm
 │
 ├─ Stage 5: Store      pipeline/stages/store_stage.py
 │   ├─ Pipeline summary JSON + run_manifest.json (version tracking)
 │   ├─ HTML/CSV/XLSX reports
 │   ├─ GeoJSON export (utils/geojson_exporter.py)
 │   ├─ Explainability overlays (utils/explainability.py)
 │   └─ Google Drive upload (optional)
 │
 └─ Stage 6: Learn      pipeline/stages/learn_stage.py
     ├─ ActiveLearner → phát hiện edge cases
     ├─ ErrorTracker → ghi nhận lỗi
     └─ AdaptiveThresholds → điều chỉnh ngưỡng (với cooldown)
```

## Components chính

### pipeline/orchestrator.py
- `PipelineState` — dataclass chuyển dữ liệu giữa các stage
- `FloodPipeline` — orchestrator linear, nhận `progress_callback`
- `validate_images()` — lọc ảnh không hợp lệ trước khi chạy

### pipeline/job_queue.py
- `JobQueue` — SQLite + thread worker
- Không block Flask request: user upload → `job_id` ngay → xử lý nền
- SSE stream: `GET /api/jobs/<id>/stream`

### pipeline/service.py
- `FloodAnalysisService` — thin wrapper tách web/CLI ra khỏi pipeline
- Web chỉ gọi `service.submit_async(images)`, không tự chạy pipeline

### pipeline/uncertainty.py
- `UncertaintyEvaluator` — phân loại `accept / needs_review / reject`
- Output có `reasons` (điểm tự tin) và `warnings` (cờ cảnh báo)

### utils/
| File | Mục đích |
|---|---|
| `config_loader.py` | Load + flatten YAML config |
| `model_registry.py` | Đọc models/registry.yaml, resolve profile |
| `depth_calibrator.py` | Hiệu chỉnh depth prediction từ reviewer feedback |
| `explainability.py` | Tạo ảnh giải thích với waterline, reference objects |
| `geojson_exporter.py` | Xuất GeoJSON cho bản đồ Leaflet |
| `pipeline_version.py` | Gắn version metadata vào mỗi kết quả |
| `alerting.py` | Auto alerts qua Telegram/Email/Discord |

### learning/
| File | Mục đích |
|---|---|
| `config_promotion.py` | candidate/current/best/rollback config lifecycle |
| `error_tracker.py` | SQLite error logging |
| `active_learner.py` | Phát hiện edge cases cần review |
| `adaptive_thresholds.py` | EMA threshold tuning với cooldown |
| `depth_calibrator.py` | Linear calibration từ human feedback |

## Luồng dữ liệu Web

```
Browser upload ảnh
    ↓ POST /api/v1/jobs (multipart)
Flask api_analyze()
    ↓ save to _upload_tmp/
FloodAnalysisService.submit_async(images)
    ↓ job_id = UUID
JobQueue.submit()
    ↓ SQLite INSERT + Queue.put()
    → HTTP 202 {job_id}

(nền) Worker thread
    ↓ FloodPipeline.run(images)
    ↓ 6 stages
    ↓ JobQueue._update(status="done")
    → SSE push to browser

Browser GET /api/jobs/<id>/stream  (SSE)
    ↓ tiến độ realtime
    ↓ progress bar update
```

## Cấu trúc thư mục output

```
output/
└── 20260519_103045/          ← run_id (timestamp)
    ├── run_manifest.json     ← version, config hash, timing
    ├── pipeline_summary.json ← flood counts, errors
    ├── report.html
    ├── report.csv
    ├── report.xlsx
    ├── flood_results.geojson
    ├── danger_zones.geojson
    ├── original/             ← ảnh gốc copy
    ├── depth_map/            ← depth PNG
    ├── overlay/              ← ảnh có overlay lũ
    └── explain/              ← ảnh giải thích uncertainty
```
