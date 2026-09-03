# Top-k spectral analysis của user embedding

## 1. Phạm vi phân tích

Báo cáo này tổng hợp kết quả so sánh user embedding giữa full branch và
hard-masked branch của artifact:

```text
src/saved/PGL_MASKED-baby-seed999-Sep-01-2026-04-06-39-analysis.pt
```

Cấu hình liên quan:

```text
dataset             = baby
ui_branch_mode      = dual
user_embedding_mode = separate
mask_graph_mode     = hard
mask_keep_ratio     = 0.3
num_users           = 19,445
```

Ba phép so sánh được thực hiện:

| Giai đoạn | Full branch | Masked branch | Shape |
|---|---|---|---:|
| Pre-propagation text | `embedding_tables.user_text.weight` | `embedding_tables.second_user_text.weight` | 19,445 x 64 |
| Pre-propagation image | `embedding_tables.user_image.weight` | `embedding_tables.second_user_image.weight` | 19,445 x 64 |
| Post-propagation | `representations.full_users` | `representations.masked_users` | 19,445 x 128 |

Các pre-propagation embedding là tham số đã học tại checkpoint, nhưng chưa đi
qua graph propagation. Chúng không phải embedding ngẫu nhiên tại thời điểm
khởi tạo training.

## 2. Định nghĩa metric

Với ma trận user embedding:

$$
E\in\mathbb{R}^{|U|\times d},
\qquad
E=U\Sigma V^\top,
$$

top-k spectral energy được tính bằng:

$$
C_k(E)=
\frac{\sum_{j=1}^{k}\sigma_j^2}{\|E\|_F^2}.
$$

Trong đó, singular values được sắp xếp giảm dần:

$$
\sigma_1\geq\sigma_2\geq\cdots.
$$

Chênh lệch mức tập trung phổ là:

$$
\Delta C_k=C_k^{masked}-C_k^{full}.
$$

- $\Delta C_k<0$: masked embedding ít tập trung vào top-k directions hơn.
- $\Delta C_k>0$: masked embedding tập trung vào top-k directions mạnh hơn.

User-subspace overlap và feature-subspace overlap:

$$
O_U(k)=
\frac{\|U_{full,k}^{\top}U_{masked,k}\|_F^2}{k},
$$

$$
O_F(k)=
\frac{\|V_{full,k}^{\top}V_{masked,k}\|_F^2}{k}.
$$

Embedding spectral complementarity:

$$
\operatorname{Comp}_k
=1-\frac{O_U(k)+O_F(k)}{2}.
$$

- $\operatorname{Comp}_k\approx0$: dominant subspaces tương tự.
- $\operatorname{Comp}_k$ lớn: hai nhánh biểu diễn users theo các dominant
  directions khác nhau hơn.
- Complementarity là độ khác biệt hình học, không phải phần trăm thông tin mới
  hữu ích cho recommendation.

## 3. Tổng quan độ giống representation

Trong các bảng kết quả bên dưới, metric chuẩn hóa được trình bày dưới dạng
phần trăm. Các cột `Delta C` và `Masked - full` dùng **điểm phần trăm**
(percentage points, viết tắt là `đ.%`), không phải phần trăm thay đổi tương
đối.

| Giai đoạn | Mean cosine (%) | Median cosine (%) | Global norm ratio (%) | Linear CKA (%) | Procrustes similarity (%) | Optimal scale (%) | Scale-aligned error (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pre text | 95.8621% | 96.0495% | 23.3436% | 94.5464% | 97.5181% | 23.4318% | 22.1408% |
| Pre image | 95.8634% | 96.0859% | 22.5343% | 94.5471% | 97.3074% | 22.4862% | 23.0492% |
| Post propagation | 92.8029% | 92.9930% | 21.5321% | 88.5930% | 94.5285% | 21.2967% | 32.6245% |

Global norm ratio là:

$$
\frac{\|E_{masked}\|_F}{\|E_{full}\|_F}.
$$

Các paired-user cosine statistics:

| Giai đoạn | Min (%) | P05 (%) | P25 (%) | Median (%) | P75 (%) | P95 (%) | Max (%) | Mean (%) | Std (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pre text | 87.1350% | 93.2860% | 95.0843% | 96.0495% | 96.8448% | 97.7816% | 99.4203% | 95.8621% | 1.3873% |
| Pre image | 85.0695% | 93.1498% | 95.0601% | 96.0859% | 96.8973% | 97.8165% | 99.1648% | 95.8634% | 1.4548% |
| Post propagation | 81.3827% | 89.5990% | 91.7348% | 92.9930% | 94.0697% | 95.3479% | 97.5129% | 92.8029% | 1.7719% |

Phân bố tỷ lệ norm của từng user, $\|e_u^{masked}\|_2 / \|e_u^{full}\|_2$:

| Giai đoạn | Min (%) | P05 (%) | P25 (%) | Median (%) | P75 (%) | P95 (%) | Max (%) | Mean (%) | Std (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pre text | 9.2590% | 15.4325% | 19.4625% | 22.7391% | 26.0493% | 31.2993% | 57.6594% | 22.9570% | 4.9230% |
| Pre image | 9.4070% | 15.0695% | 18.9106% | 21.9733% | 25.1854% | 30.2577% | 56.8586% | 22.2601% | 4.7886% |
| Post propagation | 7.1687% | 11.7391% | 15.0256% | 18.1536% | 23.7435% | 33.2255% | 63.4096% | 19.9602% | 6.7827% |

Frobenius energy và raw relative difference:

| Giai đoạn | Full energy | Masked energy | Global norm ratio (%) | Raw relative difference (%) |
|---|---:|---:|---:|---:|
| Pre text | 130,359.512174 | 7,103.605431 | 23.3436% | 78.4787% |
| Pre image | 104,040.410841 | 5,283.106170 | 22.5343% | 79.2125% |
| Post propagation | 58,088.016012 | 2,693.152217 | 21.5321% | 81.6500% |

Raw relative difference được tính bằng:

$$
\frac{\|E_{masked}-E_{full}\|_F}{\|E_{full}\|_F}.
$$

Hai nhánh có hướng embedding tương đối giống nhau ngay trước propagation.
Masked tables chỉ có khoảng 22-23% Frobenius norm của full tables. Sau
propagation, cosine, CKA và Procrustes similarity đều giảm, đồng thời norm
ratio giảm xuống khoảng 21.5%.

## 4. Pre-propagation: user text embedding

### 4.1. Top-k spectral energy và complementarity

| k | C_full (%) | C_masked (%) | Delta C (đ.%) | User overlap (%) | Feature overlap (%) | Complementarity (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 42.7570% | 35.3407% | -7.4163 đ.% | 84.4007% | 94.4402% | 10.5796% |
| 16 | 60.5715% | 54.5026% | -6.0690 đ.% | 90.2422% | 98.6288% | 5.5645% |
| 32 | 82.9734% | 80.0151% | -2.9582 đ.% | 90.4376% | 96.5979% | 6.4823% |
| 64 | 100.0000% | 100.0000% | 0.0000 đ.% | 94.4209% | 100.0000% | 2.7896% |

### 4.2. Spectral bands

Các band không chồng lặp được định nghĩa là 1-8, 9-16, 17-32 và 33-64.
Ví dụ:

$$
B_{9:16}=C_{16}-C_8.
$$

| Singular band | Full text (%) | Masked text (%) | Masked - full (đ.%) |
|---|---:|---:|---:|
| 1-8 | 42.7570% | 35.3407% | -7.4163 đ.% |
| 9-16 | 17.8145% | 19.1619% | +1.3474 đ.% |
| 17-32 | 22.4018% | 25.5126% | +3.1107 đ.% |
| 33-64 | 17.0266% | 19.9849% | +2.9582 đ.% |
| Tổng 1-64 | 100.0000% | 100.0000% | 0.0000 đ.% |

Masked text giảm 7.42 điểm phần trăm ở band 1-8. Phần năng lượng này được
phân bổ sang cả ba band phía sau, mạnh nhất tại band 17-32.

## 5. Pre-propagation: user image embedding

### 5.1. Top-k spectral energy và complementarity

| k | C_full (%) | C_masked (%) | Delta C (đ.%) | User overlap (%) | Feature overlap (%) | Complementarity (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 54.8315% | 49.3819% | -5.4497 đ.% | 83.6303% | 92.1168% | 12.1264% |
| 16 | 77.4997% | 73.6659% | -3.8338 đ.% | 90.5053% | 99.0155% | 5.2396% |
| 32 | 92.6882% | 91.1199% | -1.5683 đ.% | 92.5980% | 99.6604% | 3.8708% |
| 64 | 100.0000% | 100.0000% | 0.0000 đ.% | 93.8907% | 100.0000% | 3.0546% |

### 5.2. Spectral bands

| Singular band | Full image (%) | Masked image (%) | Masked - full (đ.%) |
|---|---:|---:|---:|
| 1-8 | 54.8315% | 49.3819% | -5.4497 đ.% |
| 9-16 | 22.6682% | 24.2841% | +1.6159 đ.% |
| 17-32 | 15.1884% | 17.4540% | +2.2655 đ.% |
| 33-64 | 7.3118% | 8.8801% | +1.5683 đ.% |
| Tổng 1-64 | 100.0000% | 100.0000% | 0.0000 đ.% |

Masked image giảm 5.45 điểm phần trăm ở band 1-8 và tăng năng lượng ở toàn
bộ các band phía sau.

Image embedding tập trung phổ mạnh hơn text embedding:

$$
C_{8,full}^{image}=54.8315\%
>C_{8,full}^{text}=42.7570\%.
$$

Điều này cho thấy full image user table bị chi phối bởi một số ít latent
directions mạnh hơn full text user table.

## 6. Post-propagation: final user representation

### 6.1. Top-k spectral energy và complementarity

| k | C_full (%) | C_masked (%) | Delta C (đ.%) | User overlap (%) | Feature overlap (%) | Complementarity (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 42.2996% | 32.3327% | -9.9669 đ.% | 76.9812% | 91.4804% | 15.7692% |
| 16 | 57.5594% | 48.8921% | -8.6673 đ.% | 82.6113% | 95.6681% | 10.8603% |
| 32 | 74.9605% | 69.1647% | -5.7957 đ.% | 86.1474% | 97.7730% | 8.0398% |
| 64 | 91.9145% | 89.9242% | -1.9902 đ.% | 88.1589% | 98.7774% | 6.5318% |

### 6.2. Global spectral-band contribution

| Singular band | Full users (%) | Masked users (%) | Masked - full (đ.%) |
|---|---:|---:|---:|
| 1-8 | 42.2996% | 32.3327% | -9.9669 đ.% |
| 9-16 | 15.2598% | 16.5594% | +1.2996 đ.% |
| 17-32 | 17.4011% | 20.2726% | +2.8715 đ.% |
| 33-64 | 16.9540% | 20.7595% | +3.8055 đ.% |
| Sau 64 | 8.0855% | 10.0758% | +1.9903 đ.% |

### 6.3. Tỷ trọng bên trong riêng top-64

| Singular band | Full users (%) | Masked users (%) | Masked - full (đ.%) |
|---|---:|---:|---:|
| 1-8 | 46.0206% | 35.9555% | -10.0651 đ.% |
| 9-16 | 16.6022% | 18.4148% | +1.8126 đ.% |
| 17-32 | 18.9318% | 22.5441% | +3.6123 đ.% |
| 33-64 | 18.4454% | 23.0855% | +4.6401 đ.% |
| Tổng trong top-64 | 100.0000% | 100.0000% | 0.0000 đ.% |

Post-propagation masked embedding giảm gần 10 điểm phần trăm năng lượng toàn
cục ở band 1-8. Năng lượng tăng mạnh nhất ở band 33-64. Vì final embedding
có 128 chiều nên top-64 chưa bao phủ toàn bộ phổ.

## 7. So sánh pre và post propagation

| Metric | Pre text | Pre image | Post propagation |
|---|---:|---:|---:|
| Delta C tại k=8 (đ.%) | -7.4163 đ.% | -5.4497 đ.% | -9.9669 đ.% |
| Complementarity tại k=8 | 10.5796% | 12.1264% | 15.7692% |
| User overlap tại k=8 | 84.4007% | 83.6303% | 76.9812% |
| Complementarity tại k=64 | 2.7896% | 3.0546% | 6.5318% |
| Mean paired-user cosine | 95.8621% | 95.8634% | 92.8029% |
| Linear CKA | 94.5464% | 94.5471% | 88.5930% |

Sự phân tán phổ đã tồn tại trong các learned user tables trước propagation:
masked text và masked image đều có năng lượng thấp hơn ở band 1-8 và cao hơn
ở các band phía sau.

Graph propagation làm khác biệt tăng thêm:

- Delta C tại k=8 âm mạnh hơn.
- User-subspace overlap giảm.
- Complementarity tăng.
- Paired-user cosine và CKA giảm.

Kết quả phù hợp với nhận định rằng hard-masked graph không tạo ra một latent
space hoàn toàn độc lập, nhưng làm khuếch đại các khác biệt đã có giữa hai bộ
user embedding parameters.

## 8. Kết luận

1. Full user embeddings tập trung mạnh hơn vào các dominant directions đầu.
2. Masked user embeddings phân bổ năng lượng rộng hơn sang các band 9-64.
3. Hiệu ứng xuất hiện ở cả text và image trước propagation.
4. Propagation trên full graph và hard-masked graph làm hai representation
   khác nhau rõ hơn, đặc biệt trong top-8 user subspace.
5. Dù vậy, cosine, CKA và Procrustes similarity vẫn cao. Hai nhánh mang tính
   bổ sung có kiểm soát thay vì hoàn toàn độc lập.
6. Masked embedding có magnitude chỉ khoảng 21-23% full embedding. Top-k
   spectral energy là tỷ lệ chuẩn hóa riêng, nên không thể hiện trực tiếp sự
   chênh lệch magnitude này.

## 9. Lưu ý khi diễn giải

- Pre-propagation embedding có đúng 64 chiều, vì vậy $C_{64}=100\%$ là tất yếu.
- Tại full rank 64 của pre-propagation embedding, feature overlap bằng 1 cũng
  là tất yếu vì toàn bộ feature space đã được lấy. Khi đó complementarity còn
  lại đến từ user-subspace overlap.
- Post-propagation embedding có 128 chiều, nên $C_{64}<100\%$.
- Spectral analysis hiện dùng uncentered embedding để giữ cách tính tương tự
  graph analysis. Singular direction đầu có thể chứa common/mean direction.
- Linear CKA và Procrustes được tính trên centered embeddings để bổ sung góc
  nhìn không phụ thuộc vào translation, phép xoay và isotropic scaling.
- Phổ trải rộng hơn không tự động có nghĩa là tốt hơn. Phần khác biệt có hữu
  ích hay không vẫn cần được xác nhận bằng Recall, nDCG và ablation nhiều seed.
- Báo cáo Markdown nhân các metric chuẩn hóa với 100 để hiển thị phần trăm.
  Script và JSON vẫn giữ giá trị gốc trong khoảng 0-1 để thuận tiện tính toán.

## 10. Lệnh tái lập

```powershell
python src/mask_analysis/spectral_complementarity.py `
  --analysis-file src/saved/PGL_MASKED-baby-seed999-Sep-01-2026-04-06-39-analysis.pt `
  --k-values 8 16 32 64 `
  --analyze-user-embeddings `
  --output-json src/saved/PGL_MASKED-baby-user-spectral.json
```

Kết quả trong JSON nằm tại:

```text
user_embedding_analysis
pre_propagation_user_embedding_analysis.text
pre_propagation_user_embedding_analysis.image
```
