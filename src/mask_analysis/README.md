# Phân tích learned mask của PGL_MASKED

Tài liệu này tổng hợp kết quả trong notebook [`first_analysis.ipynb`](./first_analysis.ipynb) cho checkpoint:

```text
PGL_MASKED-baby-seed999-Sep-01-2026-04-06-39
```

Thiết lập chính của checkpoint:

```text
dataset: baby
seed: 999
mask_graph_mode: hard
mask_degree_mode: full
ui_branch_mode: dual
user_embedding_mode: separate
mask_keep_ratio: 0.3
```

Các kết quả dưới đây mô tả mask tại checkpoint cuối. Chúng thể hiện association, không chứng minh rằng một đặc điểm của cạnh là nguyên nhân làm performance tăng hoặc giảm.

## 1. Phạm vi dữ liệu

| Thống kê | Số lượng | Tỷ lệ |
|---|---:|---:|
| Training edges | 118.551 | 100% |
| Selected edges | 35.565 | 30% |
| Unselected edges | 82.986 | 70% |
| Users | 19.445 | — |
| Items | 7.050 | — |

Hard mask xếp hạng các mask score và giữ lại top 30% cạnh. `mask_probability = sigmoid(mask_logit)` được sử dụng như selection score; nó không phải xác suất đã được calibration.

### Phân loại test users theo training edges

Tất cả 19.445 test users đều có training history.

| Nhóm user | Số lượng | Tỷ lệ |
|---|---:|---:|
| Có cả selected và unselected edges | 12.922 | 66,45% |
| Chỉ có selected edges | 1.836 | 9,44% |
| Chỉ có unselected edges | 4.687 | 24,10% |
| Không có training edge | 0 | 0% |

Các phép so sánh selected–unselected trong cùng user chỉ sử dụng 12.922 user có đồng thời hai loại cạnh. Có 6.523 user bị loại khỏi các phép so sánh này.

## 2. Degree của selected và unselected edges

Edge-degree được định nghĩa là:

\[
d_{ui}^{edge}=\sqrt{d_ud_i}
\]

Trọng số propagation chuẩn hóa tương ứng là:

\[
w_{ui}^{norm}=\frac{1}{\sqrt{d_ud_i}}
\]

| Nhóm cạnh | Count | Mean edge-degree | Median | Q25 | Q75 | Q90 |
|---|---:|---:|---:|---:|---:|---:|
| Unselected | 82.986 | 21,210 | 16,733 | 10,488 | 27,148 | 40,792 |
| Selected | 35.565 | **15,338** | **11,832** | **6,928** | **20,445** | **30,594** |

Kiểm định toàn cục:

- Mann–Whitney U: `1.064.536.166`
- p-value: `< 1e-300`
- Rank-biserial effect size: `-0,279`

Selected edges có xu hướng edge-degree thấp hơn và do đó có normalized propagation weight lớn hơn.

### So sánh trong cùng user

- Median selected edge-degree trừ unselected edge-degree: **−1,002**
- Wilcoxon p-value: `1,102e-20`
- Median selected item-degree trừ unselected item-degree: **−7**
- User có selected item-degree thấp hơn: **56,13%**

Hai distribution vẫn overlap đáng kể, vì vậy đây là xu hướng tổng thể chứ không phải mọi user đều chọn item ít phổ biến hơn.

## 3. Quan hệ với user activity

| Selection direction | Users | Mean user-degree | Median | Q25 | Q75 | Q90 |
|---|---:|---:|---:|---:|---:|---:|
| Selected item-degree thấp hơn | 7.253 | 7,004 | 5 | 3 | 8 | 13 |
| Hai nhóm bằng nhau | 19 | 4,579 | 4 | 3 | 5 | 7,2 |
| Selected item-degree cao hơn | 5.650 | 6,235 | 5 | 3 | 7 | 11 |

- Mann–Whitney p-value: `1,515e-04`
- Rank-biserial effect size: `0,038`
- Spearman giữa user-degree và item-degree delta: `0,0006`
- Spearman p-value: `0,945`

Không có quan hệ đơn điệu có ý nghĩa thực tế giữa mức hoạt động của user và hướng chọn item-degree trên toàn bộ user. Một số activity band cao có tỷ lệ chọn item degree thấp lớn hơn, nhưng chúng chứa ít user và chưa đủ để kết luận đây là xu hướng ổn định.

## 4. Modality affinity

Affinity là cosine similarity giữa user modality embedding của masked branch và projected item modality embedding.

| Modality | Unselected mean | Unselected median | Selected mean | Selected median |
|---|---:|---:|---:|---:|
| Visual | 0,465 | 0,478 | **0,570** | **0,592** |
| Textual | 0,485 | 0,497 | **0,608** | **0,620** |

### Modality delta trong cùng user

Với mỗi user:

\[
\Delta_v=\operatorname{mean}(s_v^{selected})-\operatorname{mean}(s_v^{unselected})
\]

\[
\Delta_t=\operatorname{mean}(s_t^{selected})-\operatorname{mean}(s_t^{unselected})
\]

| So sánh | Mean delta | Median delta | Bootstrap 95% CI của median |
|---|---:|---:|---:|
| Visual | 0,0464 | **0,0507** | [0,0480; 0,0532] |
| Textual | 0,0568 | **0,0599** | [0,0568; 0,0629] |
| Visual − textual | −0,0104 | **−0,0105** | [−0,0142; −0,0074] |

Wilcoxon p-value của cả ba so sánh nhỏ hơn độ chính xác hiển thị trong notebook.

| Hướng modality delta | Tỷ lệ user |
|---|---:|
| Visual và textual đều dương | 49,83% |
| Chỉ textual dương | 19,23% |
| Chỉ visual dương | 14,08% |
| Cả hai không dương | 16,86% |

Selected items thường có modality affinity cao hơn unselected items của cùng user. Textual affinity thể hiện xu hướng mạnh hơn visual affinity một chút.

## 5. Overlap giữa degree và affinity

Phân tích được thực hiện ở hai phạm vi:

- `global`: item-degree và raw cosine affinity trên toàn bộ cạnh.
- `within_user`: percentile item-degree và affinity trong các training edges của cùng user.

| Scope | Modality | Spearman rho | Observed high–high | Expected nếu độc lập | Overlap lift |
|---|---|---:|---:|---:|---:|
| Global | Visual | −0,405 | 2,31% | 6,27% | 0,368 |
| Within-user | Visual | −0,410 | 3,95% | 8,95% | 0,441 |
| Global | Textual | −0,473 | 1,85% | 6,27% | 0,295 |
| Within-user | Textual | −0,490 | 3,10% | 8,95% | 0,346 |

Degree và affinity có quan hệ ngược chiều khá rõ. High-degree và high-affinity edges xuất hiện cùng nhau ít hơn nhiều so với kỳ vọng độc lập. Xu hướng này mạnh hơn ở textual modality và vẫn tồn tại khi chỉ so sánh các item của cùng user.

## 6. Mask selection theo tổ hợp degree–affinity

Low và high được xác định bằng bottom/top 25% trong cùng user. Các cạnh nằm ở 50% giữa được bỏ khỏi bảng extreme groups.

### Textual

| Degree | Affinity | Edges | Tỷ lệ trong extreme groups | Selection rate | Selection lift |
|---|---|---:|---:|---:|---:|
| High | High | 3.671 | 7,76% | **45,79%** | **1,526** |
| High | Low | 20.314 | 42,95% | **23,96%** | 0,799 |
| Low | High | 19.191 | 40,58% | **37,21%** | 1,240 |
| Low | Low | 4.118 | 8,71% | 28,73% | 0,958 |

Textual affinity có interaction rõ với degree:

- Khi textual affinity cao, chuyển từ low degree sang high degree làm selection rate tăng từ 37,21% lên 45,79%.
- Khi textual affinity thấp, chuyển sang high degree làm selection rate giảm từ 28,73% xuống 23,96%.
- High-degree + high-textual-affinity là nhóm được ưu tiên mạnh nhất, nhưng nhóm này khá hiếm.

### Visual

| Degree | Affinity | Edges | Tỷ lệ trong extreme groups | Selection rate | Selection lift |
|---|---|---:|---:|---:|---:|
| High | High | 4.677 | 10,12% | 37,27% | 1,242 |
| High | Low | 18.321 | 39,66% | 26,95% | 0,898 |
| Low | High | 18.025 | 39,02% | **38,10%** | **1,270** |
| Low | Low | 5.171 | 11,19% | 26,03% | 0,868 |

Ở visual modality, degree gần như không tạo thêm khác biệt khi đã biết affinity:

- High-affinity edges có selection rate khoảng 37–38% ở cả hai degree groups.
- Low-affinity edges có selection rate khoảng 26–27%.

Visual affinity vì vậy có vẻ là tín hiệu chính, gần như độc lập với degree.

## 7. Kết luận

Các kết quả phù hợp với cách diễn giải sau:

1. Mask không phải một popularity filter đơn giản.
2. Mask có xu hướng giữ các cạnh degree thấp hơn và modality affinity cao hơn.
3. Visual affinity cao làm cạnh dễ được chọn gần như bất kể degree.
4. Textual affinity là tín hiệu mạnh hơn và tương tác với degree: item phổ biến chỉ được ưu tiên mạnh khi textual affinity cũng cao.
5. High-degree + high-affinity là tổ hợp hiếm, nhưng high-degree + high-textual-affinity có selection rate cao nhất.
6. Mask hoạt động gần với một personalized semantic filter, ưu tiên interaction phù hợp với preference của user hơn là chỉ ưu tiên item phổ biến.
7. Với 24,10% test users chỉ có unselected training edges, masked branch không còn neighbor propagation cho user tại checkpoint cuối. Prediction của nhóm này chủ yếu dựa vào full branch, initial embedding qua fusion và multimodal item graph.

## 8. Giới hạn

- Phân tích chỉ sử dụng một checkpoint, một seed và dataset Baby.
- Hard selection được đọc tại checkpoint cuối; Gumbel sampling trong quá trình training có thể từng chọn các cạnh hiện đang unselected.
- Mask score và modality embeddings được học đồng thời nên có thể cùng thích nghi với nhau.
- P-value rất nhỏ một phần do số cạnh lớn; cần xem effect size và độ lớn chênh lệch, không chỉ statistical significance.
- Logistic interaction model chưa chạy vì kernel hiện tại không có `statsmodels`; nhận định interaction đang dựa trên quadrant selection rates.
- Cần lặp lại trên nhiều seed và dataset trước khi xem các kết luận trên là đặc tính ổn định của mô hình.
