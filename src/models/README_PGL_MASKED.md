# PGL_MASKED: học hai biểu diễn từ full graph và masked graph

## 1. Mục tiêu

`PGL_MASKED` mở rộng PGL bằng cách học một mask trên các user–item interaction
đã tồn tại. Mục tiêu là tạo ra hai góc nhìn của cùng một user–item graph:

1. **Full branch** truyền thông tin trên toàn bộ interaction graph.
2. **Masked branch** truyền thông tin trên graph đã được lọc hoặc gán trọng số
   bởi learnable mask.

Hai representation sau đó được kết hợp để dự đoán mức độ phù hợp giữa user và
item. Model đồng thời tối ưu ranking loss, contrastive loss giữa hai branch và
regularization cho mask.

Model không học mask cho mọi cặp user–item có thể có. Mỗi interaction xuất hiện
trong training graph có đúng một `mask_logit`; những cặp không có interaction
không được thêm vào graph.

Các implementation liên quan:

- [`pgl_masked.py`](./pgl_masked.py): implementation đầy đủ của model.
- [`pgl_masked_3.py`](./pgl_masked_3.py): cùng kiến trúc và forward, nhưng dùng
  custom backward để tránh dense adjacency gradient gây OOM.
- [`../configs/model/PGL_MASKED.yaml`](../configs/model/PGL_MASKED.yaml): config
  của implementation gốc.
- [`../configs/model/PGL_MASKED_3.yaml`](../configs/model/PGL_MASKED_3.yaml):
  config của phiên bản memory-safe.

## 2. Kiến trúc tổng quát

```text
Raw image features ── Linear + normalize ──┐
                                           ├─ Initial item embeddings
Raw text features  ── Linear + normalize ──┘
                                                     │
Initial user embeddings ─────────────────────────────┤
                                                     │
                         ┌───────────────────────────┴──────────────┐
                         │                                          │
                   Full UI graph                              Masked UI graph
                         │                                          │
                    H_full                                     H_masked
                         │                                          │
                         └──────────── Gated fusion ────────────────┘
                                                │
                                          Fused UI embeddings
                                                │
                     Multimodal item graph ──────┤
                                                │
                                    Final user/item embeddings
                                                │
                                  BPR + contrastive + mask loss
```

Với config mặc định:

```yaml
embedding_size: 64
feat_embed_dim: 64
n_ui_layers: 2
n_mm_layers: 1
mask_graph_mode: hard
mask_keep_ratio: 0.3
mask_degree_mode: full
user_embedding_mode: separate
ui_branch_mode: dual
ui_fusion_mode: gated_sum
```

Image và text embedding sau projection đều có 64 chiều. Khi concatenate hai
modality, UI node embedding và multimodal item embedding đều có 128 chiều.

## 3. User–item graph

Training interaction matrix được chuyển thành binary và loại duplicate. Mỗi
interaction `(u, i)` tạo ra hai directed edge trong symmetric graph:

```text
u → i
i → u
```

Item node ID được offset thêm `n_users`. Adjacency vì vậy có kích thước:

```text
(n_users + n_items) × (n_users + n_items)
```

Full graph sử dụng symmetric normalized edge weight:

```math
w_{ui}^{full}=\frac{1}{\sqrt{d_ud_i}}.
```

Tất cả edge indices, normalized weights và adjacency được lưu dưới dạng sparse
COO tensor.

## 4. Learnable interaction mask

Mỗi training interaction có một scalar parameter:

```text
mask_logits.shape = [num_interactions]
```

Mask probability được tính bằng:

```math
p_{ui}=\sigma(l_{ui}).
```

Logit ban đầu được đặt sao cho mọi probability bằng `mask_keep_ratio`:

```math
l_0=\log\frac{r}{1-r}, \qquad \sigma(l_0)=r.
```

Trong đó `r` là `mask_keep_ratio`.

### 4.1. Hard mask

Khi training, đầu mỗi epoch model thêm Gumbel noise vào mask score:

```math
s_{ui}=\frac{l_{ui}}{T}+g_{ui}
```

và giữ lại global top-k interaction, với:

```math
k=\operatorname{round}(r|E|).
```

Đây là **global top-k**, không phải top-k riêng cho từng user. Vì vậy model bảo
đảm tổng số edge được giữ, nhưng không đảm bảo mỗi user đều còn edge trong
masked graph.

Các selected edge dùng straight-through estimator:

```python
hard_mask = 1 + soft_mask - soft_mask.detach()
```

Do đó:

- Forward xem selected edge có mask bằng `1`.
- Backward vẫn truyền gradient về `mask_logits`.
- Unselected edge không nhận ranking/contrastive gradient trong epoch hiện tại,
  nhưng vẫn nhận gradient từ mask regularization.

Các sampled training indices được giữ cố định trong một epoch. Sang epoch tiếp
theo, Gumbel noise mới được sinh và tập selected edge được lấy mẫu lại.

Khi validation/test, model không thêm noise và chọn top-k trực tiếp theo
`mask_logits`.

### 4.2. Soft mask

Với:

```yaml
mask_graph_mode: soft
```

mọi observed interaction đều được giữ. Mask probability trở thành learnable
edge weight. Nếu `mask_degree_mode: full`:

```math
w_{ui}^{masked}=w_{ui}^{full}p_{ui}.
```

Soft mode không dùng `mask_keep_ratio` để xóa một tỷ lệ edge. Tham số này là
target của mean mask probability và là giá trị khởi tạo của probability.

### 4.3. Degree normalization

Hai lựa chọn:

- `full`: dùng degree của full graph rồi nhân mask. Đây là mode nhanh hơn và
  giữ cùng normalization reference giữa hai branch.
- `masked`: tính lại degree từ masked edge weights. Khi mask thay đổi, degree
  normalization cũng thay đổi theo.

## 5. Initial embeddings

Item features được project và chuẩn hóa:

```python
image_features = normalize(image_trs(image_embedding))
text_features = normalize(text_trs(text_embedding))
multimodal_items = concat(image_features, text_features)
```

User representation được ghép từ hai learnable tables:

```python
user_embeddings = concat(user_image, user_text)
```

Với `user_embedding_mode: separate`, masked branch có hai user tables riêng:

```text
Full branch:   user_image + user_text
Masked branch: second_user_image + second_user_text
```

Hai branch vẫn dùng chung initial item features. Với
`user_embedding_mode: shared`, cả hai branch dùng chung user tables.

## 6. Graph propagation

Mỗi UI layer thực hiện:

```math
H^{(l+1)}=AH^{(l)}.
```

Output là trung bình của initial embeddings và tất cả layer outputs:

```math
H_{out}=\frac{1}{L+1}\sum_{l=0}^{L}H^{(l)}.
```

Với `n_ui_layers: 2`:

```math
H_{out}=\frac{H^{(0)}+H^{(1)}+H^{(2)}}{3}.
```

Trong `ui_branch_mode: dual`, propagation được chạy một lần trên full graph và
một lần trên masked graph.

## 7. Fusion hai UI branches

Với `ui_fusion_mode: gated_sum`, full và masked embeddings được concatenate để
tính gate:

```math
g=\sigma(W[H_{full}\Vert H_{masked}]+b).
```

Fused representation là:

```math
H_{fused}=g\odot H_{full}+(1-g)\odot H_{masked}.
```

Gate được tính cho từng node và từng embedding dimension:

- `g` gần 1: ưu tiên full branch.
- `g` gần 0: ưu tiên masked branch.
- `g` gần 0.5: trộn hai branch gần như đều nhau.

Nếu thay đổi mask nhưng recommendation gần như không đổi, một giả thuyết cần
kiểm tra là gate đang bỏ qua masked branch hoặc hai branch đã học representation
quá giống nhau.

`gated_concat` concatenate hai raw branch outputs rồi đưa qua một linear layer.
Trong implementation hiện tại, gate vẫn được tính nhưng không được dùng để tạo
output của mode này. Các thí nghiệm mặc định dùng `gated_sum` nên không chịu ảnh
hưởng của chi tiết đó.

## 8. Multimodal item graph

Model xây hai item–item KNN graphs từ cosine similarity của image và text raw
features. Hai graph được kết hợp:

```math
A_{MM}=\alpha A_{image}+(1-\alpha)A_{text},
```

với `alpha = mm_image_weight`.

Graph sau khi xây được cache trong dataset directory:

```text
mm_adj_freedomdsp_<knn_k>_<10 * mm_image_weight>.pt
```

Multimodal item representation được tính bằng sparse propagation trên graph
này. Final outputs của default dual mode là:

```text
Final users = fused UI user embeddings
Final items = fused UI item embeddings + multimodal item embeddings
```

## 9. Training objective

Tổng loss:

```math
L=L_{BPR}+\lambda_{CL}L_{CL}+\lambda_{mask}L_{mask}.
```

### 9.1. BPR ranking loss

Với positive item `i+` và negative item `i-`:

```math
L_{BPR}=-\log\sigma(s(u,i^+)-s(u,i^-)).
```

Score là dot product giữa final user và item embeddings.

### 9.2. Cross-branch contrastive loss

Trong default dual mode, InfoNCE kéo hai representation của cùng user/item lại
gần nhau và đẩy các representation khác trong batch ra xa:

```text
full user embedding   ↔ masked user embedding
full item embedding   ↔ masked item embedding
```

Code chỉ dùng unique users và unique positive items của batch. `cl_weight` điều
khiển ảnh hưởng của loss này.

### 9.3. Mask regularization

Mask loss gồm hai phần:

```math
L_{mask}=(\operatorname{mean}(p)-r)^2
        +\beta\operatorname{mean}(p(1-p)).
```

- Budget term ép mean probability gần `mask_keep_ratio`.
- Binary term khuyến khích probability tiến về 0 hoặc 1.
- `mask_weight` điều khiển đóng góp tổng thể của mask loss.
- `mask_binary_weight` là hệ số của binary term.

## 10. Prediction và evaluation

Khi full-sort evaluation, model tính:

```math
S=H_UH_I^T.
```

Mỗi evaluation user được chấm điểm với toàn bộ candidate items. Trainer loại
các training-positive items trước khi lấy top-k recommendation.

## 11. Các graph và branch modes

### `mask_graph_mode`

| Mode | Nhánh thứ hai |
|---|---|
| `hard` | Global hard top-k learned mask |
| `soft` | Full observed graph với learnable edge weights |
| `double_full` | Full graph, không có mask |
| `svd` | SVD-derived graph |
| `local_prunning` | Local-pruned graph khi train, full graph khi inference |

### `ui_branch_mode`

| Mode | Cách biểu diễn UI graph |
|---|---|
| `dual` | Full branch + masked/alternative branch + fusion |
| `masked_only` | Chỉ dùng branch thứ hai |
| `dual_modal` | Visual và textual UI branches riêng rồi concatenate/project |

`masked_only` gần single-branch PGL hơn, nhưng loss và mask regularization vẫn
cần được đồng bộ nếu mục tiêu là tạo một ablation chỉ khác PGL ở learned mask.

## 12. Khác biệt với PGL gốc

PGL gốc có một UI graph branch. Với local mode, model dùng local-pruned graph
khi train và full graph khi inference. `PGL_MASKED` bổ sung hoặc thay đổi:

- Full và masked graph branches.
- Learnable interaction logits.
- Hard/soft graph masking.
- Optional separate user embedding tables.
- Gated branch fusion.
- Cross-branch contrastive objective.
- Mask budget và binary regularization.

Vì vậy `PGL_MASKED` không phải thí nghiệm chỉ thay random pruning bằng learned
mask; nhiều thành phần của kiến trúc và objective đã thay đổi đồng thời.

## 13. Vì sao `PGL_MASKED_3` cần thiết?

Trong `PGL_MASKED`, masked adjacency values phụ thuộc vào `mask_logits` và cần
gradient. Native backward của `torch.sparse.mm` có thể tạo intermediate gradient
dense với shape:

```text
n_nodes × n_nodes
```

Trên Clothing:

```text
n_nodes ≈ 62,420
FP32 dense intermediate ≈ 14.52 GiB
```

Điều này gây OOM dù chỉ observed edges có mask parameters.

`PGL_MASKED_3` giữ nguyên native `torch.sparse.mm` ở forward nhưng định nghĩa
custom backward:

```math
\frac{\partial L}{\partial A_{ij}}
=\left\langle\frac{\partial L}{\partial Y_i},X_j\right\rangle
```

chỉ cho các stored edges `(i, j)`. Gradient của node embeddings vẫn là:

```math
\frac{\partial L}{\partial X}=A^T\frac{\partial L}{\partial Y}.
```

Do đó model giữ nguyên kiến trúc, forward và công thức gradient, nhưng memory
không còn tỷ lệ với `n_nodes²`.

## 14. Cách chạy

Từ thư mục `src`:

```bash
python main.py --model PGL_MASKED --dataset baby
```

Với phiên bản tránh OOM:

```bash
python main.py --model PGL_MASKED_3 --dataset clothing
```

Để chạy nhiều `mask_keep_ratio`:

```yaml
mask_keep_ratio: [0.3, 0.5, 0.7]
hyper_parameters: ["mask_keep_ratio"]
```

Chỉ chọn hyperparameter bằng validation metric; không dùng test result để chọn
ratio hoặc checkpoint.

## 15. Checkpoint và analysis artifact

Khi:

```yaml
save_analysis_artifacts: True
restore_best_model: True
```

trainer lưu model và artifact tương ứng với best validation epoch. Artifact bao
gồm:

- Metadata và mask settings.
- User/item ID của mỗi observed UI edge.
- Mask logits và probabilities.
- Hard top-k selection tại `mask_keep_ratio`.
- Trainable embedding tables.
- Full, masked và fused representations.

Ví dụ chạy spectral analysis:

```bash
python mask_analysis/spectral_complementarity.py \
  --analysis-file saved/PGL_MASKED_3-clothing-seed999-...-analysis.pt \
  --k-values 8 16 32 64 \
  --include-hard \
  --random-baseline-runs 10 \
  --analyze-user-embeddings \
  --output-json saved/PGL_MASKED_3-clothing-seed999-spectral.json
```

Các analysis quan trọng:

1. Tỷ lệ user không còn selected edge.
2. Selected edges trên mỗi user và item.
3. Mask probability distribution và mức độ bão hòa.
4. Degree/popularity của selected và unselected edges.
5. Visual/text affinity của selected edges.
6. Similarity và complementarity giữa full và masked representations.
7. Overlap của learned mask giữa nhiều seeds.

Các kết quả analysis mô tả association của learned mask; chúng không tự chứng
minh rằng một đặc điểm edge là nguyên nhân làm recommendation metric tăng hoặc
giảm. Cần kết hợp với controlled ablations như `double_full`, `soft`, nhiều
`keep_ratio` và nhiều seeds.

## 16. Lưu ý thực nghiệm

- Hard top-k là global; user có ít interaction có thể không còn selected edge.
- Metrics được làm tròn bốn chữ số, nên hai run hiển thị giống nhau chưa chắc có
  cùng top-k recommendations.
- Hard Gumbel selection và CUDA reductions không đảm bảo kết quả bitwise giống
  nhau giữa các GPU.
- Báo cáo kết quả nên dùng nhiều seeds và trình bày `mean ± standard deviation`.
- `PGL_MASKED_3` gần `PGL_MASKED` hơn `PGL_MASKED_2` vì nó giữ nguyên sparse
  forward và chỉ thay backward gây OOM.
- Nếu thay đổi `keep_ratio` nhưng metric gần như không đổi, cần kiểm tra fusion
  gate, branch similarity và mức đóng góp của multimodal item graph.
