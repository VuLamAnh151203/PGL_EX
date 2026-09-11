# Plan: negative sampling riêng cho image/text

## 1. Mục tiêu và phạm vi

Tạo một phần negative có cấu trúc cho modality BPR, nhằm cung cấp tín hiệu ranking khác nhau cho image/text. Đây là thí nghiệm; không mặc định negative mới là nhãn âm chắc chắn hay sẽ giúp mask specialization.

- Giữ toàn bộ graph train, cách học mask, fusion, I–I và CL của baseline đang tốt.
- Main BPR tiếp tục dùng `neg_items` random hiện tại.
- Modality BPR tiếp tục dùng embedding **sau fusion full/masked và cộng I–I**.
- Chỉ thay negative và trọng số trong auxiliary BPR; không cộng thêm một auxiliary BPR trùng lặp.
- Không thêm graph forward, teacher, hard/semi-hard mining trong phiên bản đầu.

## 2. Candidate pool và bộ lọc

Chuẩn bị pool từ feature image/text gốc, cố định trong training:

| Pool | Cách tạo cho positive item `i` |
| --- | --- |
| Image | Lấy KNN theo text, rồi giữ candidate có image similarity <= phân vị cấu hình trong pool đó |
| Text | Lấy KNN theo image, rồi giữ candidate có text similarity <= phân vị cấu hình trong pool đó |

Cosine similarity dùng feature chuẩn hóa L2. Loại chính `i` trước tính phân vị. Ví dụ `pool_size=100`, `difference_quantile=0.5`: giữ phần similarity thấp hơn hoặc bằng trung vị ở modality mục tiêu. Không diễn giải đây là hard-negative mining.

Khi lấy mẫu cho `(u, i)`:

1. Loại toàn bộ item thuộc lịch sử **train** của `u`.
2. Trong sampler có cấu trúc, loại candidate có số tương tác train dưới `min_item_interactions`. Random baseline không chịu bộ lọc này.
3. Nếu bật CF filter, tính:

   `CF(h,j) = |U(h) ∩ U(j)| / sqrt(|U(h)| * |U(j)|)`

   `U(h)` là tập user tương tác với item `h` trong train. Loại candidate `j` nếu `max(CF(h,j), h trong train_history[u]) > cf_threshold`.

4. Lấy uniform random trong tập hợp lệ. Pool rỗng thì fallback về `neg_items` baseline.

Không dùng validation/test để tạo pool, popularity hoặc CF. CF thấp không chứng minh user không thích item. Xử lý feature không hợp lệ/zero-norm bằng loại khỏi pool có cấu trúc và fallback khi cần.

Tính KNN theo block hoặc cơ chế sẵn có; không giữ ma trận similarity/CF dense kích thước `n_items × n_items`. Cache pool và CF cần dùng; cache phải gắn với feature, train split/item mapping và tham số tạo pool. Không sửa `mm_adj` của model.

## 3. Sampling trong batch và loss

Sau một forward bình thường, với mỗi positive và mỗi modality, độc lập:

```text
negative = neg_items[b]
weight = 1
Nếu sampler được bật và Bernoulli(structured_negative_ratio) = 1:
    lấy candidate hợp lệ từ pool của modality
    nếu có: negative = candidate; weight = structured_negative_weight
    nếu không: giữ negative random và weight = 1
```

Hai modality không bắt buộc chọn negative khác ID. Giữ nguyên positive và negative của main BPR.

Dùng `image_users/image_items` và `text_users/text_items` từ forward hiện tại; không dùng `image_masked_*` hay `text_masked_*`:

```text
margin_m[b] = dot(user_m[u], item_m[pos]) - dot(user_m[u], item_m[negative_m])
loss_m = mean(weight_m * softplus(-margin_m))
aux_bpr = 0.5 * (loss_image + loss_text)
total = main_bpr + aux_bpr_weight * aux_bpr + các loss khác của baseline
```

Giữ reduction theo batch; không chia cho tổng weight. Giữ gradient qua embedding/mask như baseline; sampling và weight không cần gradient. Khi auxiliary BPR tắt, bỏ qua sampler.

## 4. Config và ablation

Tên config đề xuất; các số dưới đây là điểm khởi đầu, cần chọn bằng validation:

```yaml
negative_sampler_mode: random  # random | modality | modality_cf
structured_negative_ratio: 0.2
structured_negative_weight: 1.0
structured_pool_size: 100
structured_difference_quantile: 0.5
structured_min_item_interactions: 5
structured_cf_threshold: 0.2  # chỉ dùng ở modality_cf
```

| Run | Mode | Ratio | Structured weight |
| --- | --- | ---: | ---: |
| A | random | 0 | 1 |
| B | modality | 0.2 | 1 |
| C | modality_cf | 0.2 | 1 |
| D | modality_cf | 0.2 | 0.5 |

Giữ seed, batch size, số tầng và các loss weight khác giống nhau. Nếu D tốt hơn, thêm đối chứng random giảm `aux_bpr_weight` theo mean sample weight thực tế của D, để tách tác động giảm loss khỏi tác động sampling.

## 5. Logging và kiểm tra triển khai

- Theo từng modality: tỷ lệ thử structured, tỷ lệ structured thành công trên toàn batch, tỷ lệ fallback trên số lần thử; mean sample weight.
- Log popularity, margin và auxiliary loss của random/structured; thời gian chuẩn bị pool tách khỏi thời gian mỗi epoch.
- Kiểm tra bằng dữ liệu nhỏ: đúng chiều KNN/filter, không lấy positive train, CF đúng công thức, pool rỗng fallback với weight 1, loss weighting đúng.
- Mode `random` hoặc ratio 0 phải đi đường baseline, không tiêu thụ thêm RNG sampling và không thay đổi loss. Dùng RNG riêng có seed cho sampler mới.
- Số graph forward mỗi train step giữ nguyên. Không tạo pool trong từng batch.
- Chọn checkpoint bằng validation. Sau khi có cấu hình tốt, kiểm tra ranking khi hoán vị soft mask trong từng user: metric tăng không tự chứng minh mask đã specialization.

## 6. Deliverables cho Codex

Implement sampler/preprocessing + tích hợp auxiliary BPR + config A/B/C/D + logging và kiểm tra chức năng phía trên. Giữ mode mặc định `random` để tương thích baseline. Báo cáo thay đổi, kết quả kiểm tra và chi phí phát sinh; không tự chạy grid search lớn.
