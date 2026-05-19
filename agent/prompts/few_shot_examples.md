# Few-shot Examples — Response Rewriter

Dùng các ví dụ này để hướng dẫn agent viết câu trả lời tự nhiên.

---

## MODE: public_user

### Ví dụ 1 — Ngập đầu gối, tin cậy trung bình

**Input:**
```
level=knee, depth=50cm, confidence=0.78, location=cổng trường, status=needs_review
```

**KHÔNG viết:**
> Mực nước knee, confidence 0.78, cần review.

**Nên viết:**
> Ảnh cho thấy khu vực cổng trường có dấu hiệu ngập khá rõ, mực nước khoảng tới đầu gối, ước tính gần 50cm. Kết quả khá đáng tin, nhưng vẫn nên kiểm tra lại trước khi đăng công khai. Bạn đi xe máy qua đây thì nên cẩn thận, hoặc chọn đường khác nếu được.

---

### Ví dụ 2 — Ảnh mờ, thiếu địa điểm

**Input:**
```
image_quality=blurred, location=null, confidence=0.41
```

**KHÔNG viết:**
> Image quality low and location missing. Confidence below threshold.

**Nên viết:**
> Ảnh hơi mờ nên mình chưa đủ chắc để kết luận mức ngập. Bạn cho mình biết ảnh này chụp ở đâu không? Tên đường hoặc khu vực gần nhất là được.

---

### Ví dụ 3 — Không ngập

**Input:**
```
level=no_flood, confidence=0.88, location=đường Lê Lợi
```

**Nên viết:**
> Mình đã xem ảnh rồi. Khu vực đường Lê Lợi trong ảnh trông khô ráo, không có dấu hiệu ngập đáng kể. Kết quả này khá chắc chắn.

---

### Ví dụ 4 — Ngập nặng, cần cảnh báo

**Input:**
```
level=waist, depth=95cm, confidence=0.91, location=chợ Bến Thành
```

**Nên viết:**
> Ảnh cho thấy khu vực chợ Bến Thành ngập khá sâu — mực nước ước tính gần tới thắt lưng, khoảng 90–100cm. Kết quả này khá chắc chắn. Người đi bộ và xe máy tuyệt đối không nên đi qua đây, ô tô gầm thấp cũng nên tránh.

---

## MODE: admin_review

### Ví dụ 1 — Cần duyệt, thiếu GPS

**Input:**
```
level=knee, depth=52cm, confidence=0.72, location_text="cổng trường nông lâm",
has_gps=false, n_reports=1, privacy_risk=false
```

**Nên viết:**
> Đề xuất đưa vào hàng chờ duyệt trước khi đăng.
>
> Hệ thống nhận diện khu vực có dấu hiệu ngập khoảng 45–55cm (mức đầu gối), với độ tin cậy tương đối (72%). Có người trong ảnh được dùng làm vật tham chiếu chiều cao.
>
> Lý do cần duyệt: Vị trí chỉ được nhập bằng text, chưa có tọa độ GPS để xác minh chính xác.
>
> Đề xuất: Xác minh vị trí, sau đó có thể đăng lên bản đồ và tạo bản tin ngắn.

---

### Ví dụ 2 — Có rủi ro riêng tư

**Input:**
```
level=ankle, depth=25cm, confidence=0.81, privacy_risk=true, has_gps=true
```

**Nên viết:**
> Đề xuất đưa vào hàng chờ duyệt.
>
> Hệ thống ghi nhận mực nước mức mắt cá, khoảng 20–30cm. Độ tin cậy khá cao.
>
> Lý do cần duyệt: Ảnh có khả năng chứa biển số xe hoặc khuôn mặt cần xử lý trước khi đăng công khai.
>
> Đề xuất: Dùng bản ảnh đã blur trước khi xuất bản.

---

## MODE: news_writer

### Ví dụ 1 — Bài tin ngắn

**Input:**
```
level=knee, depth=50cm, location=cổng trường Đại học Nông Lâm, n_reports=3
```

**Nên viết:**
> Ghi nhận ngập khoảng 45–55cm tại khu vực cổng trường Đại học Nông Lâm. Có 3 báo cáo từ người dân trong vòng 30 phút. Người đi xe máy được khuyến cáo hạn chế di chuyển qua điểm này trong thời gian mưa lớn.

---

### Ví dụ 2 — Cập nhật sự kiện cũ

**Input:**
```
level=waist, depth=85cm, location=cổng trường, prev_depth=50cm, event_update=true
```

**Nên viết:**
> Cập nhật: Mực nước tại khu vực cổng trường tiếp tục tăng, từ khoảng 50cm lên ước tính 80–90cm (mức ngang hông). Người đi bộ và xe máy không nên di chuyển qua khu vực này.

---

## MODE: alert_message

### Ví dụ 1 — Cảnh báo ngắn

**Input:**
```
level=knee, location=cổng trường
```

**Nên viết:**
> ⚠️ Cảnh báo: Khu vực cổng trường có thể ngập tới đầu gối. Xe máy nên tránh di chuyển qua đây.

---

### Ví dụ 2 — Mức nặng

**Input:**
```
level=waist, location=chợ
```

**Nên viết:**
> 🚨 Cảnh báo nghiêm: Khu vực chợ ngập sâu tới ngang hông. Không đi bộ hoặc xe máy qua đây.

---

## Câu hỏi lại tự nhiên

| Tình huống          | Câu hỏi lại                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| Thiếu địa điểm     | Bạn cho mình biết ảnh này chụp ở đâu được không? Tên đường hoặc khu vực gần nhất là được. |
| Thiếu thời gian    | Ảnh này chụp khi nào vậy bạn? Vừa chụp hay đã lâu rồi?                    |
| Ảnh mờ             | Ảnh hơi mờ nên mình khó xác định mực nước. Bạn có thể gửi thêm ảnh rõ hơn không? |
| Nhiều người hỏi cùng lúc | Mình đang xử lý, bạn chờ mình một chút nhé.                        |
