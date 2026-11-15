# Luyện phỏng vấn — Project RAG Chatbot hỏi đáp tài liệu nội bộ (tiếng Việt)

Tài liệu này viết cho chính bạn để luyện nói. Mọi con số lấy từ `evals/*.json` và code hiện tại của repo (không phải số "đẹp" bịa thêm). Cách dùng: che phần **Trả lời**, tự trả lời thành tiếng trong 30–60 giây, rồi so lại. Đừng học thuộc, hãy hiểu để diễn đạt bằng lời của mình.

---

## 0. Pitch 60 giây (chuẩn bị sẵn, nói ở câu "giới thiệu về project này")

> Đây là chatbot hỏi đáp tài liệu nội bộ bằng tiếng Việt, chạy hoàn toàn local. Người dùng hỏi bằng ngôn ngữ tự nhiên, ví dụ "quà tặng bao nhiêu thì phải khai báo?", hệ thống trả lời **kèm nguồn** (file, trang, mục) và **từ chối** nếu tài liệu không có thông tin thay vì bịa.
>
> Mình dùng kiến trúc RAG: chia PDF theo từng mục, tìm kiếm kết hợp vector (bge-m3 + FAISS) và BM25, chấm lại bằng cross-encoder reranker, rồi mới đưa vào LLM chạy bằng Ollama. Điểm mình làm kỹ nhất là **đo lường**: mình tự dựng bộ 53 câu hỏi có nhãn, đo từng thay đổi một. Nhờ vậy độ chính xác câu trả lời đi từ khoảng 89% lên 100% trên bộ eval, còn tỉ lệ bịa khi không có thông tin từ 7% về 0%.
>
> Hệ thống có giao diện web streaming, REST API bằng FastAPI, đóng gói Docker, CI chạy lint và hơn 100 test.

**Ba điều muốn người nghe nhớ:** (1) tôi hiểu pipeline RAG từ đầu đến cuối, (2) tôi cải tiến dựa trên số liệu chứ không dựa cảm giác, (3) tôi biết rõ giới hạn của hệ thống mình làm ra.

---

## 1. Tổng quan và động lực

**Q1. Project này giải quyết vấn đề gì? Người dùng là ai?**

**Trả lời:** Nhân viên thường phải mở nhiều file PDF quy định (nội quy, bảo mật, làm việc từ xa, nghỉ phép, quy tắc ứng xử) để tìm một chi tiết nhỏ như "báo nghỉ ốm trước mấy giờ". Chatbot cho phép hỏi trực tiếp và trả lời có trích dẫn để người dùng tự kiểm chứng. Dữ liệu nội bộ nên mình cho chạy local: tài liệu và câu hỏi không gửi ra dịch vụ bên ngoài.

**Q2. Tại sao dùng RAG mà không fine-tune hay nhét hết tài liệu vào prompt?**

**Trả lời:**
- **Fine-tune** dạy model hành vi/phong cách, không phải cách tốt để nạp kiến thức thay đổi. Tài liệu sửa là phải train lại, và không trích dẫn được nguồn.
- **Nhét hết vào prompt**: 5 file nhỏ thì còn được, nhưng không scale. Context càng dài càng tốn và model dễ bỏ sót ý ở giữa. Máy local lại chỉ có `num_ctx=4096`.
- **RAG**: cập nhật bằng cách thay file, chỉ đưa vài đoạn liên quan vào prompt, và biết chính xác đoạn nào được dùng nên làm được citation.

**Q3. Vai trò của bạn trong project? Bạn tự làm những gì?**

**Trả lời:** Đây là project cá nhân, mình làm toàn bộ: ingest PDF, chunking theo mục, retrieval, reranker, pipeline hỏi đáp, API, giao diện web, Docker/CI và bộ đánh giá. Mình dùng LangChain cho các phần "hạ tầng" (loader, splitter, wrapper FAISS, ChatOllama), còn phần quyết định chất lượng như BM25, RRF, ngưỡng từ chối, tạo nguồn trích dẫn, viết lại câu hỏi thì mình tự viết.

---

## 2. Kiến trúc và luồng xử lý

**Q4. Mô tả luồng khi người dùng gửi một câu hỏi.**

**Trả lời:**
1. (Nếu cần) **Viết lại câu hỏi** thành câu độc lập, có dấu. Chỉ chạy khi có lịch sử hội thoại hoặc câu có vẻ gõ không dấu.
2. **Tìm ứng viên**: vector search (bge-m3 + FAISS) và BM25 chạy song song, gộp bằng Reciprocal Rank Fusion, lấy khoảng 10 ứng viên.
3. **Rerank** bằng cross-encoder `bge-reranker-v2-m3`, giữ top 5.
4. **Áp ngưỡng**: không chunk nào đạt điểm ≥ 0.20 thì trả "không tìm thấy" ngay, **không gọi LLM**.
5. Còn chunk đạt ngưỡng thì dựng prompt (quy tắc + ngữ cảnh + câu hỏi), gọi LLM (`qwen2.5:7b-instruct`, temperature 0).
6. Nếu LLM tự từ chối bằng cách diễn đạt khác thì regex nhận ra và chuẩn hóa về một câu từ chối. Cuối cùng **code tự gắn nguồn** từ metadata của chunk.

**Q5. Vì sao không gọi LLM khi retrieval không đủ điểm?**

**Trả lời:** Ba lý do: (1) nhanh hơn, vì bỏ một lượt sinh văn bản; (2) an toàn hơn, vì nếu cho LLM một ngữ cảnh không liên quan thì nó vẫn có thể trả lời bằng kiến thức nội tại của nó, còn ở đây nhánh từ chối không để nó có cơ hội đó; (3) hành vi dễ đoán và dễ test, đây là một nhánh xác định chứ không phụ thuộc "tâm trạng" của model.

**Q6. Vì sao tách thành package `ragbot/` nhiều module thay vì một file?**

**Trả lời:** Ban đầu mình có một file `chatbot.py`. Khi thêm eval, API, hybrid search thì tách theo trách nhiệm: `ingest` (đọc và chia), `indexing` (build/cache index), `lexical` (BM25), `rerank`, `retrieval` (kết hợp), `pipeline` (luồng hỏi đáp), `api`, `cli`, `evaluate`, `config`. Lợi ích thực tế là test từng phần độc lập bằng stub: 108 test chạy không cần tải model hay Ollama. `RagPipeline` nhận retriever và chain qua constructor nên dễ thay bằng đối tượng giả.

---

## 3. Xử lý tài liệu và chunking

**Q7. Bạn đọc PDF bằng gì và có gặp vấn đề gì?**

**Trả lời:** Dùng `unstructured` ở chế độ `paged` để mỗi trang là một Document có số trang (cần cho citation), với `languages=["vie"]`. Vấn đề gặp phải: (1) PDF ngắt dòng cứng giữa câu nên mình viết hàm `_reflow` nối các dòng chưa kết thúc bằng dấu câu; (2) loader trả metadata nhiễu (tọa độ, parent_id) nên mình chỉ giữ `{source, page}`; (3) PDF scan không có text layer thì trả rỗng, mình log cảnh báo và báo lỗi rõ ràng, hướng dẫn cài Tesseract tiếng Việt.

**Q8. Chunking của bạn hoạt động thế nào? Vì sao chia theo mục?**

**Trả lời:** Các tài liệu quy định có cấu trúc mục đánh số "1. …, 2. …". Mình dùng regex nhận tiêu đề, mỗi mục là một chunk, và **lặp lại tên công ty + tên tài liệu ở đầu mọi chunk**. Lý do lặp tiêu đề: nhiều file có mục trùng tên (ví dụ "Nghỉ phép"), nếu thiếu tên tài liệu thì embedding và BM25 không phân biệt được chunk thuộc file nào. Mục nào dài hơn `chunk_size=1200` ký tự thì cắt tiếp bằng `RecursiveCharacterTextSplitter`. Trang không có mục đánh số thì fallback về cắt theo độ dài.

**Q9. Làm sao regex tiêu đề không nhận nhầm "1.000.000 VNĐ" là tiêu đề mục?**

**Trả lời:** Hai lớp bảo vệ: (1) regex yêu cầu dấu chấm theo sau bởi **khoảng trắng** (`^\d{1,2}\.\s+...`), nên "1.000.000" không khớp; (2) mình đòi **số mục tăng dần** (biến `expected`): chỉ chấp nhận "2." sau khi đã thấy "1.". Một dòng bắt đầu bằng số ngẫu nhiên giữa văn bản sẽ không khớp thứ tự nên bị bỏ qua.

**Q10. Chunking theo mục có điểm yếu gì? Bạn có đo không?**

**Trả lời:** Điểm yếu chính: nó giả định tài liệu có tiêu đề đánh số, tài liệu khác cấu trúc thì rơi về cắt theo độ dài (README đã ghi rõ hạn chế này). Mình có đo ablation (bảng ở mục 5): riêng việc đổi từ cắt cố định sang chia theo mục, khi vẫn dùng vector thuần, **hit@1 tăng từ 0.846 lên 0.949**, nhưng hit@5 lại giảm nhẹ (0.974 → 0.949) vì hai câu gõ không dấu bị trượt hoàn toàn. Tức là chunking tốt giúp xếp hạng chính xác hơn nhưng không tự sửa được điểm yếu của embedding với tiếng Việt không dấu. Phần đó giải quyết bằng BM25 bỏ dấu và query rewriting.

---

## 4. Embedding, vector search, BM25, reranker

**Q11. Project có dùng embedding và vector database không?**

**Trả lời:** Có embedding, và có vector index nhưng không phải "vector database" theo nghĩa một server riêng. Embedding dùng `BAAI/bge-m3` (đa ngôn ngữ, hỗ trợ tiếng Việt tốt), vector được chuẩn hóa. Chỉ mục dùng **FAISS**, vốn là **thư viện** tìm kiếm vector chạy trong tiến trình, không phải database có server, phân quyền hay filter metadata. Với 5 tài liệu (vài chục chunk), brute-force exact search (`IndexFlat`) là quá đủ và chính xác tuyệt đối, không cần ANN. Nếu corpus lên hàng trăm nghìn chunk hoặc cần filter theo quyền truy cập thì mình sẽ chuyển sang Qdrant/pgvector/Weaviate.

**Q12. Điểm similarity của bạn là cosine thật chứ? Có gì đặc biệt?**

**Trả lời:** Có một chi tiết dễ sai: `DistanceStrategy.COSINE` của LangChain thực chất dùng `IndexFlatL2` rồi tính `1 - distance`, nên điểm không phải cosine thật và có thể âm. Mình dùng embedding đã chuẩn hóa (`normalize_embeddings=True`) cùng `MAX_INNER_PRODUCT`. Khi vector có độ dài 1, tích vô hướng chính là cosine, nên ngưỡng có ý nghĩa và so sánh được.

**Q13. Vì sao cần BM25 khi đã có embedding?**

**Trả lời:** Hai cách bổ sung nhau. Embedding hiểu cách diễn đạt khác ("remote" ≈ "làm việc từ xa") nhưng yếu với số liệu, mã, thuật ngữ chính xác và đặc biệt yếu với **tiếng Việt gõ không dấu** ("mat khau toi thieu"). BM25 so khớp từ khóa trên văn bản đã **bỏ dấu**, nên không bị ảnh hưởng bởi dấu. Mình tự cài BM25 khoảng 50 dòng (k1=1.5, b=0.75) với tokenizer unigram + bigram để giữ được cụm như "mat khau", "lam viec". Lý do tự cài: corpus nhỏ, tránh thêm dependency, và kiểm soát tokenizer tiếng Việt.

**Q14. Reciprocal Rank Fusion là gì? Vì sao dùng nó mà không cộng điểm?**

**Trả lời:** RRF gộp hai bảng xếp hạng bằng công thức `score = Σ 1 / (60 + rank)` trên từng bảng. Điểm của BM25 (không giới hạn) và cosine (trong [−1, 1]) khác thang đo hoàn toàn, cộng hoặc trung bình trực tiếp sẽ bị một bên lấn át và cần chuẩn hóa. RRF chỉ dùng **thứ hạng**, nên không cần chuẩn hóa và rất bền. Hằng số 60 là giá trị chuẩn trong paper gốc.

**Q15. Reranker cross-encoder khác bi-encoder (embedding) ở đâu? Vì sao không dùng reranker làm bước tìm kiếm duy nhất?**

**Trả lời:** Bi-encoder mã hóa câu hỏi và chunk **riêng biệt** thành vector rồi so sánh, nên nhanh và index trước được nhưng ít chính xác. Cross-encoder đưa **cặp (câu hỏi, chunk) cùng vào model**, nên đọc được tương tác giữa hai bên và chấm chính xác hơn, nhưng phải chạy lại cho từng cặp nên chậm. Vì vậy mô hình chuẩn là: tìm rộng và rẻ (hybrid, 10 ứng viên) rồi chấm lại hẹp và đắt (rerank lấy top 5).

**Q16. Bạn dùng điểm reranker để làm gì ngoài việc xếp hạng?**

**Trả lời:** Dùng làm **ngưỡng từ chối**. Reranker trả xác suất trong [0, 1] có ý nghĩa nhất quán hơn cosine, nên mình đặt `rerank_threshold = 0.20`: không chunk nào đạt thì coi như tài liệu không có thông tin. Ngưỡng này mình chọn bằng cách quét trên bộ eval (hàm `sweep_thresholds`) chứ không đoán.

**Q17. Query rewriting để làm gì và rủi ro là gì?**

**Trả lời:** Hai việc: (1) câu hỏi nối tiếp như "còn nghỉ khẩn cấp thì sao?" không tự đứng được, phải thay đại từ/ý bỏ lửng bằng nội dung từ lịch sử; (2) câu không dấu được thêm dấu. Chỉ gọi LLM khi cần (có lịch sử hoặc câu toàn ASCII từ 3 từ trở lên) nên không tốn thêm độ trễ cho câu thường. Rủi ro là model nhỏ viết lại sai hoặc lan man, nên mình có: giới hạn `num_predict=128`, hàm `clean_rewrite` bỏ tiền tố/ngoặc kép, và nếu kết quả rỗng hoặc dài bất thường thì **giữ nguyên câu gốc**. Hạn chế đã nêu trong README: nếu tắt rewriting thì câu không dấu giảm độ chính xác rõ rệt.

---

## 5. Đánh giá chất lượng (phần mạnh nhất, hãy nói thật kỹ)

**Q18. Bạn đánh giá hệ thống thế nào? Bộ dữ liệu ra sao?**

**Trả lời:** Mình tự dựng `evals/dataset.jsonl` gồm **53 câu hỏi**: 39 câu có đáp án, 14 câu không có đáp án. Phân loại: 30 factoid, 8 near-domain (gần chủ đề nhưng tài liệu không có), 6 paraphrase, 6 off-topic, 2 không dấu, 1 đa nguồn. Mỗi câu có nhãn file gold, đoạn bằng chứng, và nhóm từ khóa bắt buộc (`must_include`, cho phép nhiều cách diễn đạt như "17:00" / "17h"). Mình tách đánh giá thành 3 lớp:
- **Retrieval**: hit@1/3/5, MRR, source recall, context precision.
- **Ngưỡng**: quét threshold xem có tách được câu có/không có đáp án không.
- **Generation**: accuracy, false-refusal rate, hallucination rate, citation accuracy, faithfulness (LLM-judge), latency.

**Q19. Hãy kể hành trình cải thiện bằng số liệu.**

**Trả lời:** Bảng retrieval (39 câu có đáp án, các biến thể chạy riêng từng thay đổi):

| Cấu hình | hit@1 | hit@5 | MRR | Context precision |
|---|---|---|---|---|
| A. Chunk cố định + vector | 0.846 | 0.974 | 0.900 | 0.43 |
| B. Chunk theo mục + vector | 0.949 | 0.949 | 0.949 | 0.36 |
| C. Chunk theo mục + hybrid | 0.923 | 0.974 | 0.941 | 0.39 |
| D. Chunk theo mục + vector + rerank | 0.949 | 0.974 | 0.962 | 0.96 |
| **E. Chunk theo mục + hybrid + rerank** | **0.974** | **1.000** | **0.983** | **0.96** |

Bảng toàn pipeline (53 câu):

| Mốc | Model | Accuracy | Bịa khi không có info | Từ chối nhầm | Faithfulness |
|---|---|---|---|---|---|
| Baseline | llama3.2:3b, chunk cố định, vector | 88.7% | 7.1% | 10.3% | 0.944 |
| Sau bước 4 (section + hybrid + rerank) | qwen2.5:7b | 96.2% | 0% | 5.1% | 1.0 |
| Sau bước 5 (query rewriting) | qwen2.5:7b | 100% (53/53) | 0% | 0% | chưa đo |

Thông điệp chính: **rerank** là bước kéo context precision từ ~0.4 lên ~0.96 (đưa ít rác hơn vào prompt), **hybrid** vá điểm yếu tiếng Việt không dấu, còn **rewriting** đưa nhóm không dấu từ 0% lên 100%.

**Q20. 100% accuracy nghe quá đẹp. Bạn có tin con số đó không?**

**Trả lời:** Không tin theo nghĩa "hệ thống đúng 100% ngoài đời". Mình nói thẳng các lý do: (1) bộ eval chỉ **53 câu** do chính mình viết, nhiều nhóm rất nhỏ (không dấu chỉ 2 câu, đa nguồn 1 câu), nên khoảng tin cậy rất rộng; (2) mình **chỉnh ngưỡng và cấu hình trên cùng bộ này**, tức có nguy cơ overfit (không có tập held-out riêng); (3) corpus chỉ 5 tài liệu mẫu, sạch và có cấu trúc. Con số có ý nghĩa như một **bài kiểm tra hồi quy** (đổi thiết kế mà số tụt là biết ngay) và để so sánh tương đối giữa các cấu hình, không phải cam kết chất lượng production. Bước tiếp theo đúng đắn là thu thập câu hỏi thật từ người dùng và tách tập dev/test.

**Q21. Faithfulness được đo thế nào? Có điểm yếu gì?**

**Trả lời:** Mình dùng LLM-judge: đưa (câu hỏi, ngữ cảnh, câu trả lời) cho model và yêu cầu trả JSON `{"faithful": true/false}` (đúng khi mọi thông tin trong câu trả lời đều được ngữ cảnh hỗ trợ). Điểm yếu: judge là `qwen2.5:7b` **cùng họ với model sinh** nên có thể thiên vị chính nó, và model nhỏ làm judge không đáng tin tuyệt đối. Ở lần chạy cuối mình tắt judge nên faithfulness không có số, và mình ghi đúng là "chưa đo" thay vì điền số cũ. Với con số 1.0 ở bước 4 cũng nên nhìn như tín hiệu tham khảo.

**Q22. `hit@k`, `MRR`, `context precision` nghĩa là gì?**

**Trả lời:**
- **hit@k**: tỉ lệ câu hỏi mà chunk đúng (đúng file gold và chứa đoạn bằng chứng) nằm trong top-k.
- **MRR**: trung bình của `1/rank` của chunk đúng đầu tiên, thưởng cho việc xếp chunk đúng càng cao càng tốt.
- **Context precision**: trong số chunk đưa vào prompt, bao nhiêu phần là chunk liên quan. Chỉ số này quan trọng vì chunk rác làm LLM dễ lạc đề và tốn context.

**Q23. Vì sao bạn đo "tỉ lệ từ chối nhầm" và "tỉ lệ bịa" riêng?**

**Trả lời:** Đây là hai loại lỗi đánh đổi lẫn nhau. Ngưỡng cao thì ít bịa nhưng từ chối nhầm nhiều; ngưỡng thấp thì ngược lại. Với chatbot tài liệu nội quy, **trả lời sai nguy hiểm hơn từ chối** (nhân viên làm theo thông tin sai), nên mình ưu tiên ngưỡng khiến hallucination = 0 và chấp nhận một ít từ chối nhầm, rồi dùng retrieval tốt hơn để kéo từ chối nhầm xuống.

**Q24. Làm sao đo "citation accuracy"?**

**Trả lời:** Với mỗi câu trả lời có đáp án và không bị từ chối, kiểm tra xem file được hiển thị trong phần "Nguồn" có nằm trong danh sách file gold hay không. Kết quả 100% ở các lần chạy. Lưu ý đây đo mức **file**, chưa đo đúng trang/mục.

**Q25. Nếu lấy một chỉ số để báo cáo cho sếp, bạn chọn gì?**

**Trả lời:** Mình chọn nhóm chỉ số "độ tin cậy" thay vì một số tổng: tỉ lệ bịa khi không có thông tin (0%), citation accuracy (100%) và accuracy trên câu có đáp án. Sếp quan tâm "có đáng tin để dùng không", mà độ tin cậy nằm ở việc hệ thống biết nói "không biết" và dẫn nguồn đúng.

---

## 6. Prompt, hallucination, citation

**Q26. Làm sao bạn giảm hallucination?**

**Trả lời:** Nhiều lớp, không dựa vào một lớp nào: (1) retrieval chất lượng cao và **ngưỡng từ chối trước khi gọi LLM**; (2) prompt có quy tắc rõ: chỉ dùng ngữ cảnh, không có thì trả đúng một câu cố định, không suy đoán; (3) `temperature=0`; (4) nhận diện lời từ chối diễn đạt khác bằng regex rồi chuẩn hóa; (5) nguồn do code tạo từ metadata thật, LLM không được tự ghi nguồn.

**Q27. Vì sao nguồn trích dẫn do code tạo thay vì để LLM ghi?**

**Trả lời:** Model nhỏ hay ghi sai hoặc bịa tên file/trang. Mình lấy nguồn thẳng từ metadata của chunk đã truy hồi (`source`, `page`, `section`), nên nguồn **luôn có thật**. Thêm nữa, hàm `select_sources` chỉ giữ chunk có điểm sát top-1 (trong biên 0.08), vì chunk xếp sau chỉ là ứng viên phụ, liệt kê hết sẽ khiến người dùng tưởng câu trả lời lấy từ cả những file không liên quan.

**Q28. Regex nhận lời từ chối có gì rủi ro?**

**Trả lời:** Hai chiều: bỏ sót cách diễn đạt mới (LLM nói "tài liệu chưa nêu rõ…") hoặc bắt nhầm câu trả lời hợp lệ có chứa cụm như "không có thông tin thay đổi". Mình giảm rủi ro bằng cách giới hạn mẫu quanh các cụm "không tìm thấy/không đề cập … thông tin" và có test cho các biến thể. Đây là giải pháp thực dụng; hướng tốt hơn là cho LLM trả JSON có trường `answerable` hoặc dùng structured output.

**Q29. `num_ctx=4096` có ý nghĩa gì?**

**Trả lời:** Ollama **âm thầm cắt phần đầu prompt** khi vượt context window, tức là mất luôn phần quy tắc ở đầu mà không báo lỗi, rất khó debug. Mình đặt `num_ctx` tường minh và giới hạn top-k, kích thước chunk sao cho prompt luôn nằm gọn trong 4096 token.

**Q30. Prompt injection: nếu tài liệu hoặc câu hỏi chứa "bỏ qua mọi quy tắc trước đó" thì sao?**

**Trả lời:** Mình chưa có phòng thủ chuyên biệt, và sẽ nói thẳng điều đó. Giảm nhẹ sẵn có: LLM chỉ nhận ngữ cảnh từ tài liệu do quản trị viên đặt vào thư mục, câu hỏi bị giới hạn 1000 ký tự, nguồn do code tạo nên không bị giả mạo bằng prompt. Nếu làm tiếp: tách rõ ranh giới ngữ cảnh/chỉ dẫn trong prompt, lọc đầu vào/đầu ra, và kiểm tra citation sau khi sinh.

---

## 7. Backend, API, triển khai

**Q31. Vì sao FastAPI? Streaming làm thế nào?**

**Trả lời:** FastAPI cho validation bằng Pydantic (giới hạn câu hỏi 1000 ký tự, lịch sử tối đa 40 tin), tài liệu OpenAPI tự sinh và dễ test với `TestClient`. Endpoint `/chat/stream` dùng **Server-Sent Events** với các sự kiện `rewrite`, `token`, `done`, `error`. Sự kiện `done` mang Answer cuối cùng (đã có nguồn và đã chuẩn hóa lời từ chối), vì phần token ghép lại có thể là câu từ chối diễn đạt khác. Vì header 200 đã gửi trước khi sinh, lỗi giữa chừng phải báo bằng sự kiện `error` chứ không đổi được status code.

**Q32. Server có lưu phiên hội thoại không? Vì sao?**

**Trả lời:** Không, thiết kế **stateless**: client giữ lịch sử và gửi lại mỗi lần (tối đa 6 lượt gần nhất được dùng làm ngữ cảnh). Ưu điểm: không cần database phiên, dễ scale ngang, không lo lộ lịch sử. Nhược điểm: payload lớn dần và chưa có người dùng riêng. Đây là một mục trong "hướng phát triển".

**Q33. Bạn xử lý đồng thời (concurrency) như thế nào?**

**Trả lời:** Hiện mọi lượt hỏi **xếp hàng qua một `threading.Lock`**. Lý do: embedding, reranker và LLM dùng chung một CPU/GPU và không đảm bảo thread-safe, nên chạy song song có thể tranh tài nguyên hoặc lỗi. Hệ quả: chỉ phù hợp một nhóm nhỏ dùng nội bộ. Để mở rộng mình sẽ tách phần suy luận (Ollama/vLLM) thành server riêng, chạy nhiều worker và dùng hàng đợi.

**Q34. Bảo mật API ở mức nào?**

**Trả lời:** Một khóa chung qua header `X-API-Key` (bật bằng `RAG_API_KEY`), so sánh bằng `secrets.compare_digest` để tránh timing attack. `/health` và trang chủ không cần khóa. Giới hạn: chưa có xác thực theo từng người dùng, chưa có rate limiting. Tải index dùng `pickle` (`allow_dangerous_deserialization`) nên chỉ an toàn khi thư mục index do chính app tạo ra, mình ghi rõ điều này trong code và README.

**Q35. Cơ chế cache index hoạt động ra sao?**

**Trả lời:** Embedding toàn bộ chunk bằng bge-m3 trên CPU rất chậm, nên mình lưu FAISS index kèm một **fingerprint**: SHA-256 của nội dung từng PDF cộng các tham số ảnh hưởng đến index (embedding model, kiểu chunk, kích thước, overlap, phiên bản schema). Lần chạy sau nếu fingerprint khớp thì nạp lại, bỏ qua cả đọc PDF lẫn embedding. Ghi index vào thư mục `.tmp` rồi **đổi tên** để một lần ghi dở không để lại cache hỏng. Cache hỏng thì ghi cảnh báo và dựng lại, không làm chết app. Biến `INDEX_SCHEMA_VERSION` để bắt buộc dựng lại khi mình đổi logic chia tài liệu mà fingerprint không phản ánh.

**Q36. Docker và CI bạn cấu hình thế nào?**

**Trả lời:**
- **Dockerfile**: Python 3.10-slim, cài `torch` bản CPU trước để pip không kéo bản CUDA hàng GB, chạy bằng user không phải root, `HEALTHCHECK` gọi `/health` với `start-period` dài vì nạp 3 model mất vài chục giây, volume `/data` giữ model HuggingFace và FAISS index để không tải lại.
- **docker-compose**: hai service `ollama` và `app`; app chờ Ollama healthy; `restart: on-failure` vì app thoát nếu Ollama chưa có model, và tự thử lại sau khi `ollama pull`. Có sẵn khối GPU để bỏ comment.
- **GitHub Actions**: workflow CI chạy `ruff check`, `ruff format --check`, `pytest` (test dùng stub, không cần model); workflow build Docker chỉ chạy khi các file ảnh hưởng đến image thay đổi (path filter) vì image nặng.

**Q37. Config quản lý ra sao?**

**Trả lời:** `pydantic-settings` đọc biến môi trường tiền tố `RAG_` và file `.env`. Có validator kiểm tra khoảng giá trị (ví dụ `chunk_overlap < chunk_size`, `rerank_threshold` trong [0,1]) nên cấu hình sai sẽ **fail ngay lúc khởi động** với thông báo rõ. Đường dẫn tương đối được tính từ thư mục gốc dự án để chạy từ đâu cũng được. Settings được cache bằng `lru_cache`.

---

## 8. Testing và chất lượng code

**Q38. Bạn test một hệ thống dùng LLM thế nào?**

**Trả lời:** Phân hai tầng. **Tầng test đơn vị** (108 test, chạy trong vài giây, không cần model/Ollama): kiểm thử logic xác định bằng stub, như tách mục, regex tiêu đề, BM25, RRF, chọn nguồn, nhận diện từ chối, `clean_rewrite`, fingerprint cache, validate config, API (kể cả SSE, API key, lỗi). **Tầng đánh giá** (`python -m ragbot.evaluate`): chạy model thật trên bộ 53 câu, vì hành vi của LLM không đảm bảo bằng assert thông thường. Tách hai tầng giúp CI nhanh và ổn định, còn chất lượng AI thì đo bằng metric.

**Q39. Lỗi khó nhất bạn gặp là gì và xử lý ra sao?**

**Trả lời (chọn một chuyện thật mà bạn nhớ rõ; gợi ý từ code):**
- *Câu không dấu truy hồi sai:* embedding hiểu kém "mat khau toi thieu" → mình thêm BM25 bỏ dấu và query rewriting; số liệu: nhóm không dấu từ 0% lên 100% accuracy.
- *Điểm "cosine" bất thường:* phát hiện `DistanceStrategy.COSINE` của LangChain không phải cosine thật → đổi sang embedding chuẩn hóa + inner product.
- *Prompt bị cắt âm thầm:* Ollama bỏ phần đầu prompt khi quá context → đặt `num_ctx` tường minh.

Khi kể, theo cấu trúc: **triệu chứng → cách xác định nguyên nhân → cách sửa → cách kiểm chứng bằng số**.

---

## 9. Hạn chế và hướng phát triển (đừng né, người phỏng vấn rất thích câu này)

**Q40. Hạn chế lớn nhất của project là gì?**

**Trả lời:** Theo thứ tự mình thấy quan trọng:
1. **Bộ eval nhỏ và tự viết**, ngưỡng chỉnh trên chính nó nên có nguy cơ overfit; chưa có dữ liệu người dùng thật.
2. **Chunking phụ thuộc cấu trúc "1. Tiêu đề"**; PDF có bảng, cột, ảnh scan sẽ kém hơn.
3. **Mỗi lần thêm/sửa PDF phải dựng lại toàn bộ index** (chưa incremental).
4. **Một khóa API chung, hàng đợi một luồng**, chưa phục vụ nhiều người đồng thời.
5. Chưa có phòng thủ prompt injection chuyên biệt, chưa có phân quyền theo tài liệu.

**Q41. Nếu có thêm 2 tuần, bạn làm gì trước?**

**Trả lời:** Thứ tự ưu tiên của mình: (1) mở rộng bộ eval với câu hỏi do người khác viết và tách tập dev/test, vì mọi cải tiến khác phụ thuộc vào đo lường đáng tin; (2) index incremental theo hash từng file; (3) tách LLM server và chạy nhiều worker; (4) lưu phiên và xác thực theo người dùng; (5) thêm logging/metrics (độ trễ từng bước, tỉ lệ từ chối) để theo dõi trên production.

**Q42. Nếu corpus lớn lên 10.000 tài liệu thì thay đổi gì?**

**Trả lời:**
- Vector store: chuyển FAISS flat sang index xấp xỉ (HNSW/IVF) hoặc DB có sẵn như Qdrant/pgvector, có filter metadata và cập nhật incremental.
- BM25: dùng Elasticsearch/OpenSearch thay vì tự tính trong bộ nhớ.
- Ingest: chạy bất đồng bộ theo job, lưu từng chunk kèm hash để cập nhật từng phần.
- Retrieval: giữ mô hình "rộng rồi rerank", nhưng tăng `fetch_k`, cân nhắc cache embedding câu hỏi và batch rerank trên GPU.
- Phân quyền: lọc theo quyền người dùng **trước** khi truy hồi, không phải sau.

**Q43. Làm sao biết chatbot đang trả lời sai trên production?**

**Trả lời:** Ghi log ẩn danh (câu hỏi đã viết lại, điểm top-1, có từ chối không, chunk nào được chọn), thêm nút phản hồi đúng/sai trên giao diện, theo dõi tỉ lệ từ chối theo thời gian (tăng đột biến thường nghĩa là tài liệu thiếu hoặc retrieval hỏng), và định kỳ lấy mẫu câu bị từ chối/điểm sát ngưỡng để bổ sung vào bộ eval.

---

## 10. Câu hỏi "bẫy" thường gặp

**Q44. "Sao không dùng ChatGPT/Claude API cho chất lượng cao hơn?"**

**Trả lời:** Yêu cầu của bài toán là **dữ liệu nội bộ không rời máy**, nên mình chọn model local. Đổi lại là chất lượng và tốc độ phụ thuộc phần cứng và model nhỏ (7B). Kiến trúc tách riêng LLM qua `ChatOllama`, nên nếu chính sách cho phép thì đổi sang API ngoài chỉ cần thay một đối tượng, các phần retrieval, ngưỡng, nguồn, eval giữ nguyên.

**Q45. "LangChain có cần thiết không?"**

**Trả lời:** Mình dùng có chọn lọc: loader, text splitter, wrapper FAISS, prompt template, ChatOllama. Phần quyết định chất lượng thì tự viết (BM25, RRF, retriever, ngưỡng, sinh nguồn, rewrite) vì cần kiểm soát và test từng bước. Mình cũng chủ động tránh bẫy của abstraction, ví dụ tự kiểm tra thấy `DistanceStrategy.COSINE` không đúng như tên gọi.

**Q46. "Khác gì so với chatbot dùng LangChain mẫu trên mạng?"**

**Trả lời:** Bản mẫu thường dừng ở load → split cố định → FAISS → LLM. Phần mình thêm và **chứng minh bằng số**: chunk theo cấu trúc, hybrid retrieval, reranker, ngưỡng từ chối trước LLM, query rewriting cho tiếng Việt không dấu, citation do code tạo, bộ eval có nhãn, cache index có fingerprint, API streaming, Docker, CI. Điểm phân biệt lớn nhất là có vòng lặp **đo → thay đổi → đo lại**.

**Q47. "Nếu người dùng hỏi câu có đáp án ở hai tài liệu khác nhau?"**

**Trả lời:** Top-5 chunk sau rerank có thể đến từ nhiều file và đều được đưa vào prompt, `select_sources` hiển thị mọi chunk sát top-1. Mình thành thật là bộ eval chỉ có **1 câu đa nguồn**, nên chưa đủ để khẳng định chất lượng ở nhóm này. Đó là chỗ mình sẽ mở rộng eval trước khi tin.

**Q48. "Tại sao `qwen2.5:7b` mà không phải model khác?"**

**Trả lời:** Mình thử hai model trên cùng bộ eval (3 lần chạy mỗi model): `qwen2.5:7b` ổn định ở **0.962**, còn `llama3.2:3b` dao động **0.943–0.962**. Qwen hỗ trợ tiếng Việt tốt hơn và ổn định hơn nên được chọn làm mặc định. Model nhẹ vẫn dùng được qua biến cấu hình cho máy yếu. Lưu ý: với 53 câu, chênh lệch này nhỏ, mình xem nó là xu hướng chứ chưa phải kết luận thống kê.

---

## 11. Câu hỏi nên hỏi ngược lại nhà tuyển dụng

- Hệ thống AI của team đang đo chất lượng bằng cách nào? Có bộ eval chung không?
- Dữ liệu là tài liệu có cấu trúc hay phi cấu trúc? Có yêu cầu phân quyền theo tài liệu không?
- Team dùng LLM qua API hay tự host? Ràng buộc về dữ liệu/chi phí ra sao?
- Vị trí này kỳ vọng nghiêng về xây pipeline AI hay tích hợp/backend?

---

## 12. Lưu ý riêng khi trả lời (đọc trước giờ phỏng vấn)

1. **Đừng nói "chạy production".** Nói đúng: "project cá nhân, đã đóng gói Docker + CI, thiết kế cho nhóm nhỏ dùng nội bộ".
2. **Đừng tuyên bố "100% chính xác".** Nói: "100% trên bộ eval 53 câu do mình viết, dùng để chống hồi quy, mình biết rõ nó nhỏ và có nguy cơ overfit".
3. **Về con số ngưỡng 0.20:** comment trong `ragbot/config.py` nêu "0.154 / 0.257", trong khi báo cáo sweep ghi câu không-có-đáp-án cao nhất là 0.2474 và ngưỡng cân bằng tốt nhất là 0.28. Hai chỗ chưa khớp (có thể do chạy ở thời điểm khác nhau). **Đừng trích các số lẻ này**; chỉ nói "chọn ngưỡng 0.20 bằng cách quét trên bộ eval, ưu tiên không bịa". Trước khi phỏng vấn nên chạy lại `python -m ragbot.evaluate --mode full` để có số mới thống nhất và sửa comment.
4. **FAISS là thư viện, không phải database.** Câu "có dùng vector database không?" rất hay bị hỏi, hãy trả lời đúng như Q11.
5. **Nắm ít nhất một câu chuyện debug thật** (Q39), kể có số liệu trước/sau.
6. Mở sẵn repo để chỉ vào code khi được yêu cầu: `ragbot/pipeline.py` (luồng chính), `ragbot/retrieval.py` (hybrid + rerank), `ragbot/ingest.py` (chia theo mục), `ragbot/evaluate.py` (đánh giá).
