# 使用官方 Python 映像檔作為基礎
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt 並安裝依賴
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式的其餘部分到工作目錄
COPY . .

# 預設 FastAPI 應用程式執行的埠號
EXPOSE 8000

# 執行 FastAPI 應用程式的指令
# 使用 --host 0.0.0.0 使其可以從容器外部存取
# 注意：在生產環境中，您可能不想使用 --reload
CMD ["uvicorn", "main_qa_app:app", "--host", "0.0.0.0", "--port", "8000"] 