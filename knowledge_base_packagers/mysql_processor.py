'''
Module to process patient records from a MySQL database.
'''
import mysql.connector

# --- Configuration --- (使用者必須自行修改這些預留位置)
DB_CONFIG = {
    'host': 'YOUR_DB_HOST',             # 例如 'localhost' 或 IP 位址
    'user': 'YOUR_DB_USER',             # 例如 'ai_agent_user'
    'password': 'YOUR_DB_PASSWORD',     # 資料庫使用者密碼
    'database': 'YOUR_DB_NAME'          # 資料庫名稱，例如 'hospital_records'
}

# 表格與欄位設定 (使用者必須自行修改)
# 假設您的病例資料在名為 'patient_cases' 的表格中
TABLE_NAME = 'patient_cases'
# 假設欄位名稱如下，請根據您的實際情況修改
COLUMN_ID = 'case_id'                    # 病例的唯一ID
COLUMN_CHIEF_COMPLAINT = 'chief_complaint' # 主訴
COLUMN_ASSESSMENT = 'assessment_result'    # 評估結果
COLUMN_TREATMENT_PLAN = 'treatment_plan'   # 治療計畫
COLUMN_LAST_MODIFIED = 'last_modified_date' # (選用) 用於增量更新的最後修改日期欄位

# --- Text Splitting Configuration (與其他 processor 相似，但通常病例記錄本身較短，可能不需要積極切分) ---
# 對於病例資料，通常一條記錄就是一個上下文單元，除非內容非常長。
# 這裡的切分可能更多是為了統一格式，或者如果單一欄位文本過長才需要。
CHUNK_SIZE_CHARS = 1500 # 設定較大值，因為我們希望盡可能保留完整病例描述
CHUNK_OVERLAP_CHARS = 150

def format_case_record_to_text(record, column_names):
    '''Formats a database record (tuple) into a readable text string.'''
    record_dict = dict(zip(column_names, record))
    
    # 根據實際欄位建立描述性文本
    # 您可以根據需求調整這個格式化字串
    text_parts = []
    if record_dict.get(COLUMN_CHIEF_COMPLAINT):
        text_parts.append(f"病患主訴：{record_dict[COLUMN_CHIEF_COMPLAINT]}")
    if record_dict.get(COLUMN_ASSESSMENT):
        text_parts.append(f"評估結果：{record_dict[COLUMN_ASSESSMENT]}")
    if record_dict.get(COLUMN_TREATMENT_PLAN):
        text_parts.append(f"治療計畫：{record_dict[COLUMN_TREATMENT_PLAN]}")
    
    if not text_parts:
        return None # 如果重要欄位都為空，則不處理此記錄
        
    return "；".join(text_parts) + "。"

def split_record_text_into_chunks(text, record_id, table_name):
    '''
    Splits formatted record text if it's too long. 
    Usually, one record is one chunk unless exceptionally long.
    '''
    if not text:
        return []

    # 對於資料庫記錄，通常我們將整個格式化後的文本視為一個 chunk
    # 除非它真的非常長，超過 CHUNK_SIZE_CHARS
    if len(text) <= CHUNK_SIZE_CHARS:
        return [{
            "text": text,
            "source": f"{table_name}_record_id_{record_id}",
            "type": "mysql_record",
            "record_id": record_id
            # 可以考慮加入 "chunk_char_start": 0, "chunk_char_end": len(text) 如果需要
        }]
    else: # 如果文本過長，則套用類似的切分邏輯 (較少見於結構化數據)
        chunks = []
        start_index = 0
        doc_len = len(text)
        while start_index < doc_len:
            end_index = min(start_index + CHUNK_SIZE_CHARS, doc_len)
            # (簡易切分，不特別處理句子邊界，因為原始數據可能無明確句子結構)
            chunk_text = text[start_index:end_index]
            chunks.append({
                "text": chunk_text,
                "source": f"{table_name}_record_id_{record_id}_part{len(chunks)+1}",
                "type": "mysql_record_chunk",
                "record_id": record_id,
                "chunk_char_start": start_index,
                "chunk_char_end": end_index
            })
            if end_index == doc_len:
                break
            start_index = max(0, end_index - CHUNK_OVERLAP_CHARS)
            if start_index >= end_index:
                start_index = end_index
        return chunks

def process_mysql_records(db_config=None, last_sync_time=None):
    '''
    Fetches records from MySQL, formats them, and (if needed) splits into chunks.
    `last_sync_time` (datetime object or string in YYYY-MM-DD HH:MM:SS format) 
    can be used for incremental updates if a suitable timestamp column exists.
    '''
    config = db_config if db_config else DB_CONFIG
    all_text_chunks = []
    
    # 檢查是否有預設的敏感資訊，如果使用者未修改則提示並返回
    if config['host'] == 'YOUR_DB_HOST' or \
       config['user'] == 'YOUR_DB_USER' or \
       config['password'] == 'YOUR_DB_PASSWORD' or \
       config['database'] == 'YOUR_DB_NAME':
        print("ERROR: Database configuration in mysql_processor.py contains placeholder values.")
        print("Please update DB_CONFIG with your actual MySQL connection details.")
        return all_text_chunks

    try:
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        
        query = f"SELECT {COLUMN_ID}, {COLUMN_CHIEF_COMPLAINT}, {COLUMN_ASSESSMENT}, {COLUMN_TREATMENT_PLAN} FROM {TABLE_NAME}"
        
        # 增量更新邏輯 (選用)
        # 如果提供了 last_sync_time 且設定了 COLUMN_LAST_MODIFIED
        # 假設 last_sync_time 是一個 datetime 物件或 YYYY-MM-DD HH:MM:SS 格式的字串
        params = []
        if last_sync_time and COLUMN_LAST_MODIFIED:
            query += f" WHERE {COLUMN_LAST_MODIFIED} > %s"
            params.append(last_sync_time) 
            print(f"Fetching records from {TABLE_NAME} modified after: {last_sync_time}")
        else:
            print(f"Fetching all records from table: {TABLE_NAME}")

        # 加入一些限制，避免一次拉取過多資料，例如 LIMIT 或 WHERE 條件
        # query += " LIMIT 10000" # 開發測試時可以加上，實際部署時需考慮分批處理

        cursor.execute(query, tuple(params))
        
        column_names_for_formatting = [COLUMN_ID, COLUMN_CHIEF_COMPLAINT, COLUMN_ASSESSMENT, COLUMN_TREATMENT_PLAN]
        fetched_records = 0
        processed_chunks_count = 0

        for row in cursor:
            fetched_records += 1
            record_id = row[0] # 假設第一欄是 ID
            formatted_text = format_case_record_to_text(row, column_names_for_formatting) # 傳遞原始 row 給格式化函數
            
            if formatted_text:
                chunks = split_record_text_into_chunks(formatted_text, record_id, TABLE_NAME)
                if chunks:
                    all_text_chunks.extend(chunks)
                    processed_chunks_count += len(chunks)
            
            if fetched_records % 500 == 0:
                print(f"  Processed {fetched_records} records so far...")

        print(f"Fetched {fetched_records} records from MySQL.")
        print(f"Total text chunks created from MySQL records: {processed_chunks_count}")

    except mysql.connector.Error as err:
        print(f"Error connecting to or querying MySQL database: {err}")
        # 根據錯誤類型可以做更細緻的處理
        if err.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
            print("MySQL Error: Access denied. Check username and password.")
        elif err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
            print("MySQL Error: Database does not exist.")
        else:
            print(err)
    except Exception as e:
        print(f"An unexpected error occurred during MySQL processing: {e}")
    finally:
        if 'cnx' in locals() and cnx.is_connected():
            cursor.close()
            cnx.close()
            print("MySQL connection closed.")
            
    return all_text_chunks

# --- Main execution for testing ---
if __name__ == '__main__':
    print("--- Testing MySQL Record Processor ---")
    # 為了測試，您需要：
    # 1. 修改上面的 DB_CONFIG, TABLE_NAME, 和 COLUMN_* 常數為您資料庫的實際設定。
    # 2. 確保您的 MySQL 伺服器正在運行且網路可達。
    # 3. 執行 python knowledge_base_packagers/mysql_processor.py
    
    # 執行前請務必確認 DB_CONFIG 中的預留位置已被替換！
    if DB_CONFIG['host'] == 'YOUR_DB_HOST':
        print("\nWARNING: DB_CONFIG in mysql_processor.py still contains placeholder values.")
        print("The script will not attempt to connect to the database.")
        print("Please update DB_CONFIG with your actual MySQL connection details to test.")
    else:
        # 測試獲取所有資料
        print("\n--- Test: Processing all records ---")
        processed_chunks = process_mysql_records()
        if processed_chunks:
            print(f"Successfully processed {len(processed_chunks)} chunks from MySQL.")
            # print("First few chunks:")
            # for i, chunk in enumerate(processed_chunks[:min(3, len(processed_chunks))]):
            #     print(f"--- Chunk {i+1} (Source: {chunk['source']}, Record ID: {chunk.get('record_id')}) ---")
            #     print(chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"])
        else:
            print("No chunks were processed from MySQL. Check configurations and database connection.")
        
        # 範例：測試增量更新 (需要您的表格有 COLUMN_LAST_MODIFIED 欄位)
        # from datetime import datetime, timedelta
        # print("\n--- Test: Processing records since yesterday (requires COLUMN_LAST_MODIFIED) ---")
        # one_day_ago = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        # if COLUMN_LAST_MODIFIED == 'last_modified_date': # 假設您的欄位名稱是這個
        #     incremental_chunks = process_mysql_records(last_sync_time=one_day_ago)
        #     if incremental_chunks:
        #         print(f"Successfully processed {len(incremental_chunks)} incremental chunks from MySQL.")
        #     else:
        #         print("No incremental chunks processed, or COLUMN_LAST_MODIFIED is not set up correctly.")
        # else:
        #     print("Skipping incremental test as COLUMN_LAST_MODIFIED is not configured or is default.")

    print("--- MySQL Record Processor Test Complete ---") 