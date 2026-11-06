# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:06/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   - Record này nói về cả bức ảnh, vì record phân loại các loại xe trong ảnh, nhiều vật thể.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   - Danh sách tên lớp đi kèm với taxonomy/checkpoint, trường taxonomy_name = ImageNet-1K có trong mọi record.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   - Vì tên lớp đứng một mình thì mơ hồ, cần cả ba để định danh chắc chắn
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   - Guideline phải quy định rõ tiêu chí chọn "một" lớp khi ảnh có nhiều vật.
6. Vì sao không được lấy `score` của model làm ground truth?
   - Vì score là độ tin cậy của model, không phải đáp án đúng.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: class_name = person; score = 0.912625; bbox_xyxy = [385.33, 69.24, 498.92, 348.92]; bbox_width = 113.58; bbox_height = 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel? 
   - góc trên-trái (93.17, 187.95), góc dưới-phải (223.01, 320.91), rộng 129.84, cao 132.96
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   - 0.20 có 17 predition
   - 0.60 có 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   - hạ ngưỡng thì ít bỏ sót vật nhưng reviewer phải kiểm tra nhiều hộp hơn
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   - Box phải bao sát toàn bộ phần nhìn thấy được của vật: bốn cạnh chạm vào điểm ngoài cùng của vật trong ảnh, không bao thêm nền hay vật khác; không vẽ phần vật nằm ngoài khung hình, cắt box tại mép ảnh; vật bị che khuất thì box chỉ bao phần nhìn thấy được, không đoán phần bị che; mỗi vật một box riêng, không gộp nhiều vật vào một box; hai box cùng lớp chồng lên nhau vẫn giữ cả hai nếu là hai vật riêng.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   - 

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   - Box chỉ là hình chữ nhật bao ngoài: nó cho biết vật ở đâu và lớn cỡ nào, nhưng bên trong box vẫn lẫn cả nền và vật khác, không cho biết vật thực sự chiếm chỗ nào và dáng ra sao. Polygon/mask cho biên chính xác tới từng pixel: đường viền cong theo đầu, vai, tay, chân người; biết phần nào của box là vật, phần nào là nền; tách được hai vật cùng nằm trong một box; đo được diện tích thật của vật; và thấy được chỗ vật bị khuyết hay lõm vào.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   - Nó là mã định danh duy nhất cho một vật cụ thể trong một ảnh, để phân biệt các vật cùng lớp với nhau. Ảnh kitchen có 4 cái bát cùng class_id = 45, cùng tên bowl, nhưng là 4 vật khác nhau: kitchen-002, kitchen-003, kitchen-006, kitchen-010 — chỉ instance_id mới phân biệt được chúng. Nó không phải class ID vì class ID chỉ nói "đây là loại bát", còn instance_id nói "đây là cái bát nào". Nó cũng không phải tracking ID vì tracking ID theo dõi cùng một vật qua nhiều khung hình video theo thời gian; còn ở đây mỗi ảnh độc lập, kitchen-001 không liên quan gì tới dining-001 hay traffic-001.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   - Vẽ biên sát theo đường viền thật của vật, cong theo hình dáng vật; chỉ tô đúng pixel thuộc về vật, không tô nền; chỗ vật bị vật khác che thì biên dừng tại đường che, không vẽ phần bị che; hai vật chạm nhau thì biên là đường tiếp xúc nhìn thấy được giữa chúng, không để hai mask tô cùng một pixel; vật bị cắt mép ảnh thì mask dừng tại mép ảnh; chi tiết nhỏ nhưng nhìn rõ thì vẽ theo, chi tiết mờ không chắc thì bỏ qua chứ không đoán; mỗi vật một mask riêng, không gộp nhiều vật vào một mask; xong kiểm tra lại bằng cách bật/tắt lớp mask xem biên có ôm sát vật không.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   - mask bàn ăn (dining table 0.58, vùng xanh) tô cả mặt bàn lẫn vùng dưới các bát đặt trên bàn (bowl 0.70, bowl 0.55, bowl 0.74), nên biên giữa mặt bàn và mép bát bị mờ; tương tự mask oven 0.49 (tím) và mask bàn ăn chồng lên nhau ở vùng giữa-trái, cùng một pixel bị hai mask nhận. Hai bát ở góc dưới-trái cũng chạm nhau, mask hồng của chúng đè lên nhau.
   - Trường hợp này nên viết thành quy tắc trong guideline vì nó lặp lại trong hầu hết ảnh đông vật, và có cách giải quyết chuẩn: chỉ vẽ phần nhìn thấy được, vật ở trước (che vật sau) thì thắng các pixel chồng lấn, biên là đường tiếp xúc nhìn thấy được.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn duy nhất cho cả ảnh: (taxonomy_name, class_id, class_name), ví dụ (ImageNet-1K, 468, cab) | Ảnh traffic nhiều chủ đề (xe buýt to nhất, ô tô, taxi) nhưng chỉ được chọn một nhãn; model chọn cab 0.51 trong khi xe buýt mới nổi bật nhất — nhãn đơn không diễn tả được cả cảnh; score 0.51 là độ tin cậy, không phải đáp án đúng | Theo guideline chọn một chủ thể chính theo tiêu chí đã định (lớn nhất/nổi bật nhất), ghi đủ taxonomy + class_id + class_name | Kiểm tra nhãn có đúng tiêu chí chủ thể chính không; kiểm tra bộ ba taxonomy/id/tên có khớp nhau không; ghi nhận trường hợp nhiều chủ đề để bổ sung guideline |
| Phát hiện vật thể | Một box cho mỗi vật: (class, bbox xyxy theo pixel), ví dụ person [385.33, 69.24, 498.92, 348.92], ảnh kitchen 640×427 | Box person 0.61 ở mép trái chỉ thấy một bàn tay (x_min ≈ 0) — một bàn tay có tính là person không? Ngưỡng 0.35 cho 11 box, 0.60 chỉ còn 6 — hạ ngưỡng tăng số box phải dọn; các box điểm thấp (bowl 0.38, cup 0.38) dễ sai | Vẽ một box ôm sát cho mỗi vật nhìn thấy được, chỉ phần trong ảnh, cắt box tại mép ảnh, không đoán phần bị che | Kiểm tra từng box (vật có thật không, đúng lớp không, box có ôm sát không); xóa box sai/trùng; kiểm tra trường hợp cắt mép/che khuất; đưa trường hợp mơ hồ (bàn tay) lên Lab Coach |
| Instance segmentation | Một polygon/mask cho mỗi vật: (instance_id, class, danh sách điểm polygon), ví dụ kitchen-001 person, 348 điểm | Mask bàn ăn (dining table 0.58) tô cả vùng dưới các bát trên bàn; mask oven 0.49 chồng lên mask bàn ăn — cùng một pixel bị hai mask nhận; biên giữa mặt bàn và mép bát mờ; hai bát góc dưới-trái chạm nhau, mask đè lên nhau | Vẽ biên sát theo phần nhìn thấy được; vật trước che vật sau thì vật trước thắng pixel chồng; mỗi vật một mask riêng | Kiểm tra biên mask có ôm sát vật không; kiểm tra không có pixel nào thuộc hai mask; kiểm tra tách instance (4 bát = 4 instance_id khác nhau); kiểm tra xử lý che khuất theo quy tắc |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   - Chỉ sử dụng dữ liệu lab đã được cấp trong phạm vi bài lab, không chia sẻ, tải lên hay lưu trữ ở bên ngoài, và không đưa bất kỳ thông tin cá nhân nào vào báo cáo hay repo.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   - Dừng ngay, không gán nhãn và không upload ảnh/dữ liệu đó, rồi báo cho Lab Coach kèm mã ảnh (sample_id) để coach xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
