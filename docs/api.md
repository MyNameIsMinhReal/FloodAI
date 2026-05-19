# FloodAI Pipeline — API Reference

Base URL: `https://your-domain.com`

Interactive docs: `GET /api/docs`
OpenAPI JSON:     `GET /openapi.json`

---

## Authentication
Tất cả endpoints `/api/*` (trừ `/api/v1/health`) yêu cầu login session.
Đăng nhập tại `POST /login` với `username` và `password`.

---

## System

### `GET /api/v1/health`
Health check — không cần login.

**Response:**
```json
{
  "status": "ok",
  "python": "3.11.x",
  "jobs": { "pending": 0, "running": 1, "done": 42, "failed": 2 }
}
```

### `GET /api/v1/metrics`
Số liệu tổng hợp về job và pipeline performance.

**Response:**
```json
{
  "jobs": { "total": 50, "done": 45, "failed": 3, "pending": 1, "running": 1 },
  "performance": { "avg_processing_s": 34.5, "total_images_analyzed": 1200 },
  "fail_reasons": [["Out of memory", 2], ["File not found", 1]]
}
```

---

## Jobs

### `POST /api/v1/jobs`
Upload ảnh và gửi vào hàng đợi phân tích nền.

**Form data:**
- `images`: file(s) — jpg/png/webp, tối đa 30MB mỗi file
- `safe`: `"true"` — tắt Drive/Learning

**Response 202:**
```json
{
  "job_id": "20260519_103045_a3b4c5",
  "queued_count": 12,
  "status_url": "/api/jobs/20260519_103045_a3b4c5",
  "stream_url":  "/api/jobs/20260519_103045_a3b4c5/stream"
}
```

### `GET /api/v1/jobs/{job_id}`
Trạng thái hiện tại của job.

**Response:**
```json
{
  "job_id":     "20260519_103045_a3b4c5",
  "status":     "running",
  "stage":      "depth",
  "progress":   62,
  "message":    "depth: 31/50",
  "created_at": "2026-05-19T10:30:45",
  "finished_at": ""
}
```

### `GET /api/jobs/{job_id}/stream`
**Server-Sent Events** — tiến độ realtime.

```js
const es = new EventSource('/api/jobs/<id>/stream');
es.onmessage = e => {
  const d = JSON.parse(e.data);
  // d.progress, d.stage, d.message, d.status
};
```

**Event format:**
```
data: {"job_id":"...", "status":"running", "stage":"depth", "progress":62, "message":"..."}
```

### `GET /api/v1/jobs/{job_id}/result`
Kết quả tóm tắt (chỉ khi status=done).

### `GET /api/v1/jobs/{job_id}/report`
Pipeline summary JSON đầy đủ.

---

## Legacy endpoints

| Route | Alias for |
|---|---|
| `POST /api/analyze` | `POST /api/v1/jobs` |
| `GET /api/jobs` | Danh sách jobs gần đây |
| `GET /api/runs/<run_id>/report` | Report JSON từ output dir |
| `GET /api/health` | `GET /api/v1/health` |

---

## Ví dụ curl

```bash
# Upload ảnh
curl -X POST https://your-domain.com/api/v1/jobs \
  -F "images=@flood1.jpg" \
  -F "images=@flood2.jpg" \
  -F "safe=true" \
  -b "session=..."

# Kiểm tra trạng thái
curl https://your-domain.com/api/v1/jobs/20260519_103045_a3b4c5

# Health check
curl https://your-domain.com/api/v1/health
```
