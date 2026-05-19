# Hướng dẫn Deployment

## 1. Yêu cầu

- Python 3.10+
- Docker + Docker Compose (cho production)
- 4GB RAM tối thiểu (8GB+ khuyến nghị với model lớn)
- GPU CUDA (optional, nhưng nhanh hơn 5–10x)

---

## 2. Cài đặt nhanh (Development)

```bash
git clone <repo>
cd flood-pipeline

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Config
cp .env.example .env        # chỉnh sửa .env
# set SECRET_KEY, ADMIN_PASS

# Chạy với config dev (CPU, không Drive)
python main.py run --input ./sample_images --config configs/dev.yaml --safe
```

---

## 3. Web App

```bash
# Dev
python app.py

# Production với Docker
docker-compose up -d

# Xem logs
docker-compose logs -f flood-app
```

---

## 4. Cấu hình `.env`

```env
# Bắt buộc trong production
SECRET_KEY=<random 32 chars>
ADMIN_PASS=<strong password>

# Optional
ADMIN_USER=admin
FLASK_ENV=production
PORT=5000

# Alerts
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
SMTP_HOST=smtp.gmail.com
SMTP_USER=...
SMTP_PASS=...
ALERT_EMAIL_TO=...
DISCORD_WEBHOOK_URL=...

# Google Drive
GOOGLE_MAPS_API_KEY=...
```

---

## 5. SSL (HTTPS)

Tạo self-signed cert (dev):
```bash
mkdir nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem \
  -subj "/CN=localhost"
```

Production: dùng Let's Encrypt / Certbot.

---

## 6. Chọn config phù hợp

| Môi trường | Config | Dùng khi |
|---|---|---|
| Phát triển | `configs/dev.yaml` | Test local, debug |
| CPU-only | `configs/cpu.yaml` | Máy không có GPU |
| GPU | `configs/gpu.yaml` | Server có NVIDIA GPU |
| Demo | `configs/demo.yaml` | Showcase, presentation |
| Production | `configs/prod.yaml` | Deploy thật |

```bash
python main.py run --input ./images --config configs/gpu.yaml
```

---

## 7. Watch Mode (tự động xử lý)

```bash
# Theo dõi folder, tự tạo job khi có ảnh mới
python main.py watch --input /data/incoming --interval 10 --config configs/cpu.yaml
```

---

## 8. Benchmark trước khi deploy

```bash
# So sánh pipeline với baseline
python scripts/baseline_compare.py --dataset datasets/eval --config configs/cpu.yaml

# Evaluation chi tiết
python scripts/evaluate.py --dataset datasets/eval --config configs/cpu.yaml --verbose
```

---

## 9. Docker GPU

```bash
# Cần NVIDIA Container Toolkit
docker-compose --profile gpu up -d

# Kiểm tra GPU available
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## 10. Log rotation

```bash
# Thêm vào cron
0 0 * * * find /app/output -name "*.log" -mtime +30 -delete
0 2 * * 0 find /app/output -maxdepth 1 -type d -mtime +90 -exec rm -rf {} \;
```
