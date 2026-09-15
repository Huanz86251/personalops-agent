read_file 可直接读取任务 PDF/Office 和已登记下载，本地解析及自动 OCR，不发送原始 PDF 给模型。需要指定输出或 OCR 策略时用 attachment_to_text；图片文字用 ocr_image。检查警告和 next_page；OCR 不等于图表理解。
根据问题读取相关页并保留页码；需要全文结论时继续读取后续页，说明无法识别的部分。
