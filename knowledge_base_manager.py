'''
Knowledge Base Manager

Handles the creation, updating, and querying of the vector knowledge base.
It orchestrates data loading from various sources (initially website scraper),
embeds the text chunks, and stores/retrieves them from a vector store (FAISS).
'''
import os
import faiss
import numpy as np
import pickle # For saving/loading FAISS index and doc metadata
from openai import OpenAI # Updated import for openai > v1.0
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

# Assuming website_scraper is in the knowledge_base_packagers directory
# and this script might be run from the project root or a similar level.
# Adjust the import path if necessary based on your project structure.
from knowledge_base_packagers import website_scraper
from knowledge_base_packagers import word_processor # 新增匯入
from knowledge_base_packagers import google_drive_processor # 新增 Google Drive 匯入
from knowledge_base_packagers import smb_processor # 新增 SMB 匯入

# Attempt to import a configuration URL from the scraper for display purposes
try:
    from knowledge_base_packagers.website_scraper import SITEMAP_URL_FOR_STATUS_DISPLAY
    s_config_url = SITEMAP_URL_FOR_STATUS_DISPLAY
except ImportError:
    s_config_url = "(Scraper SITEMAP_URL_FOR_STATUS_DISPLAY not found)" # Fallback

# --- Configuration --- (使用者需根據實際情況修改)
OPENAI_API_KEY = "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" # 強烈建議使用環境變數設定 API Key
if not OPENAI_API_KEY:
    print("警告：OPENAI_API_KEY 環境變數未設定。您需要在執行前設定它。")
    # raise ValueError("OPENAI_API_KEY environment variable not set.")

EMBEDDING_MODEL = "text-embedding-ada-002" # OpenAI 的嵌入模型

# 向量儲存的相關路徑
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "knowledge_base.index")
DOC_METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "doc_metadata.pkl")
KB_STATUS_FILE = os.path.join(VECTOR_STORE_DIR, "kb_status.json") # 知識庫狀態檔案
SYNC_LOG_FILE = os.path.join(VECTOR_STORE_DIR, "sync_log.json") # 新增：同步日誌檔案
MAX_SYNC_LOG_ENTRIES = 50 # 新增：日誌檔案中保留的最大條目數量

# 預設的知識庫狀態結構
DEFAULT_KB_STATUS = {
    "overall_status": {
        "last_full_rebuild_timestamp": None,
        "last_any_sync_timestamp": None, # 新增：任何來源最後同步時間
        "total_indexed_vectors": 0,
        "message": "尚未進行任何同步作業。"
    },
    "sources": {
        "website": {
            "last_sync_timestamp": None,
            "status": "pending", # pending, success, error, success_no_new_data, etc.
            "message": "等待同步",
            "processed_items": 0,
            "embedded_items": 0,
            "target_config_url": None # 新增：用於顯示爬蟲配置的目標 URL
        },
        "word_documents": { # 新增 Word 文件來源狀態
            "last_sync_timestamp": None,
            "status": "pending",
            "message": "等待同步",
            "processed_items": 0,
            "embedded_items": 0,
            "target_config_url": None # 將會設為 word_processor.WORD_DOCS_DIR
        },
        "google_drive": { # 新增 Google Drive 來源狀態
            "last_sync_timestamp": None,
            "status": "pending",
            "message": "等待同步",
            "processed_items": 0,
            "embedded_items": 0,
            "target_config_url": "Google Drive Folder ID / Credentials Path (see google_drive_processor.py)" # 預留設定路徑提示
        },
        "smb": { # 新增 SMB 來源狀態
            "last_sync_timestamp": None,
            "status": "pending",
            "message": "等待同步",
            "processed_items": 0,
            "embedded_items": 0,
            "target_config_url": "SMB Share Path / Credentials (see smb_processor.py)" # 預留設定路徑提示
        },
        # Future sources like mysql can be added here
        # "mysql": { ... }
    }
}

# 確保向量儲存目錄存在
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# 初始化 OpenAI client
# client = OpenAI(api_key=OPENAI_API_KEY) # client 會在需要時才初始化，避免 KEY 未設定時直接報錯

class KnowledgeBaseManager:
    def __init__(self, openai_api_key=OPENAI_API_KEY):
        if not openai_api_key:
            raise ValueError("OpenAI API Key is required to initialize KnowledgeBaseManager.")
        self.client = OpenAI(api_key=openai_api_key)
        self.index = None
        self.doc_metadata = [] # Stores metadata like source, title for each vector
        self._load_vector_store()
        # Ensure sync log file exists (can be empty list)
        if not os.path.exists(SYNC_LOG_FILE):
            with open(SYNC_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)
        # 初始化 word_documents 的 target_config_url
        # 這樣即使在首次 _read_kb_status 之前，DEFAULT_KB_STATUS 也是最新的
        if DEFAULT_KB_STATUS["sources"].get("word_documents"):
             DEFAULT_KB_STATUS["sources"]["word_documents"]["target_config_url"] = word_processor.WORD_DOCS_DIR
        # 初始化新增來源的 target_config_url (如果它們的處理器中有定義的設定變數)
        # 由於 google_drive_processor 和 smb_processor 中的設定是註解掉的 placeholder，
        # 這裡我們暫時不直接從模組中讀取，而是使用 DEFAULT_KB_STATUS 中已定義的提示字串。
        # 如果未來這些處理器模組中定義了可匯出的設定變數 (例如 GOOGLE_DRIVE_FOLDER_ID)，
        # 則可以像 word_processor.WORD_DOCS_DIR 一樣更新它們。
        # 例如:
        # if DEFAULT_KB_STATUS["sources"].get("google_drive") and hasattr(google_drive_processor, 'GOOGLE_DRIVE_FOLDER_ID'):
        #      DEFAULT_KB_STATUS["sources"]["google_drive"]["target_config_url"] = google_drive_processor.GOOGLE_DRIVE_FOLDER_ID
        # else:
        #      DEFAULT_KB_STATUS["sources"]["google_drive"]["target_config_url"] = "Google Drive Folder ID (config in processor)"

        # 目前，DEFAULT_KB_STATUS 中已經為新的 sources 提供了 target_config_url 的預留文字。

    def _get_embedding(self, text, model=EMBEDDING_MODEL):
        '''Generates embedding for a given text using OpenAI.'''
        try:
            response = self.client.embeddings.create(input=[text.replace("\n", " ")], model=model)
            return response.data[0].embedding
        except Exception as e:
            print(f"生成嵌入時發生錯誤: {e}")
            return None

    def _load_vector_store(self):
        '''Loads the FAISS index and document metadata from disk if they exist.'''
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOC_METADATA_PATH):
            try:
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                with open(DOC_METADATA_PATH, 'rb') as f:
                    self.doc_metadata = pickle.load(f)
                print(f"成功從 {VECTOR_STORE_DIR} 載入現有的向量知識庫。索引中有 {self.index.ntotal} 個向量。")
            except Exception as e:
                print(f"載入向量儲存時發生錯誤: {e}. 將會建立新的儲存。")
                self.index = None
                self.doc_metadata = []
        else:
            print("找不到現有的向量知識庫，將會建立新的。")

    def _save_vector_store(self):
        '''Saves the FAISS index and document metadata to disk.'''
        if self.index is not None:
            try:
                faiss.write_index(self.index, FAISS_INDEX_PATH)
                with open(DOC_METADATA_PATH, 'wb') as f:
                    pickle.dump(self.doc_metadata, f)
                print(f"向量知識庫已成功儲存至 {VECTOR_STORE_DIR}。")
            except Exception as e:
                print(f"儲存向量知識庫時發生錯誤: {e}")
        else:
            print("索引未被初始化，無法儲存。")

    def _read_kb_status(self) -> dict:
        """Reads the current KB status from KB_STATUS_FILE, returns default if not found or error."""
        default_status_copy = DEFAULT_KB_STATUS.copy()
        if os.path.exists(KB_STATUS_FILE):
            try:
                with open(KB_STATUS_FILE, 'r', encoding='utf-8') as f:
                    loaded_status = json.load(f)
                    # 確保基本結構存在，特別是 overall_status 和 sources
                    if not isinstance(loaded_status, dict) or \
                       "overall_status" not in loaded_status or \
                       "sources" not in loaded_status or \
                       not isinstance(loaded_status["overall_status"], dict) or \
                       not isinstance(loaded_status["sources"], dict):
                        print(f"警告：{KB_STATUS_FILE} 內容結構不完整或無效。將與預設狀態合併。")
                        # Merge loaded status with default to ensure all keys are present
                        # This is a shallow merge, for deeper merge, more complex logic is needed
                        # but for top-level keys, this should help.
                        # A better approach might be to validate and fill missing keys recursively.
                        # For now, prioritize ensuring 'overall_status' and 'sources' exist.
                        
                        merged_status = default_status_copy
                        if isinstance(loaded_status, dict):
                            # If loaded_status is a dict, try to preserve what's there if keys match
                            if "overall_status" in loaded_status and isinstance(loaded_status["overall_status"], dict):
                                merged_status["overall_status"].update(loaded_status["overall_status"])
                            if "sources" in loaded_status and isinstance(loaded_status["sources"], dict):
                                # For sources, ensure each known source from default exists
                                for src_key, src_default_val in DEFAULT_KB_STATUS["sources"].items():
                                    if src_key not in merged_status["sources"]:
                                        merged_status["sources"][src_key] = src_default_val.copy()
                                    # 特別處理 target_config_url 的初始化
                                    if src_key == "word_documents" and merged_status["sources"][src_key].get("target_config_url") is None:
                                        merged_status["sources"][src_key]["target_config_url"] = word_processor.WORD_DOCS_DIR
                                    elif src_key == "website" and merged_status["sources"][src_key].get("target_config_url") is None:
                                        # s_config_url 應該在檔案頂部定義
                                        merged_status["sources"][src_key]["target_config_url"] = s_config_url 
                                    # 新增 google_drive 和 smb 的 target_config_url 初始化 (如果為 None)
                                    elif src_key == "google_drive" and merged_status["sources"][src_key].get("target_config_url") is None:
                                        merged_status["sources"][src_key]["target_config_url"] = DEFAULT_KB_STATUS["sources"]["google_drive"]["target_config_url"]
                                    elif src_key == "smb" and merged_status["sources"][src_key].get("target_config_url") is None:
                                        merged_status["sources"][src_key]["target_config_url"] = DEFAULT_KB_STATUS["sources"]["smb"]["target_config_url"]
                                        
                                    if src_key in loaded_status["sources"] and isinstance(loaded_status["sources"][src_key], dict):
                                        merged_status["sources"][src_key].update(loaded_status["sources"][src_key])
                        return merged_status
                    
                    # 確保所有在 DEFAULT_KB_STATUS 中定義的 sources 都存在於 loaded_status
                    # 並確保它們的 target_config_url 被正確初始化
                    for default_src_key, default_src_value in DEFAULT_KB_STATUS["sources"].items():
                        if default_src_key not in loaded_status["sources"]:
                            loaded_status["sources"][default_src_key] = default_src_value.copy()
                            if default_src_key == "word_documents":
                                loaded_status["sources"][default_src_key]["target_config_url"] = word_processor.WORD_DOCS_DIR
                            elif default_src_key == "website":
                                loaded_status["sources"][default_src_key]["target_config_url"] = s_config_url
                            # 新增: 初始化新來源的 target_config_url (如果它們在讀取的 status 中不存在)
                            elif default_src_key == "google_drive":
                                loaded_status["sources"][default_src_key]["target_config_url"] = DEFAULT_KB_STATUS["sources"]["google_drive"]["target_config_url"]
                            elif default_src_key == "smb":
                                loaded_status["sources"][default_src_key]["target_config_url"] = DEFAULT_KB_STATUS["sources"]["smb"]["target_config_url"]
                        elif loaded_status["sources"][default_src_key].get("target_config_url") is None: # 如果存在但 URL 未設定
                            if default_src_key == "word_documents":
                                loaded_status["sources"][default_src_key]["target_config_url"] = word_processor.WORD_DOCS_DIR
                            elif default_src_key == "website":
                                loaded_status["sources"][default_src_key]["target_config_url"] = s_config_url
                            # 新增: 設定新來源的 target_config_url (如果已存在但 URL 為 None)
                            elif default_src_key == "google_drive":
                                loaded_status["sources"][default_src_key]["target_config_url"] = DEFAULT_KB_STATUS["sources"]["google_drive"]["target_config_url"]
                            elif default_src_key == "smb":
                                loaded_status["sources"][default_src_key]["target_config_url"] = DEFAULT_KB_STATUS["sources"]["smb"]["target_config_url"]

                    return loaded_status
            except (json.JSONDecodeError, IOError) as e:
                print(f"讀取知識庫狀態檔案 {KB_STATUS_FILE} 時發生錯誤: {e}. 將使用預設狀態。")
                return default_status_copy
        return default_status_copy

    def _write_kb_status(self, status_data: dict):
        """Helper function to write the kb_status.json file."""
        try:
            with open(KB_STATUS_FILE, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, ensure_ascii=False, indent=4)
            # print(f"知識庫狀態已更新至 {KB_STATUS_FILE}") # Can be too verbose
        except Exception as e:
            print(f"更新知識庫狀態檔案時發生錯誤: {e}")

    def _add_sync_log_entry(self, operation_type: str, source_name: str, status: str, message: str):
        """Adds an entry to the sync log file, keeping only the last MAX_SYNC_LOG_ENTRIES."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation_type": operation_type, # e.g., "full_rebuild", "source_sync_website", "source_sync_word"
            "source_name": source_name, # Specific source like "website" or "overall" for full rebuild
            "status": status, # e.g., "started", "success", "error", "partial_success", "no_new_data"
            "message": message
        }
        
        logs: List[Dict[str, Any]] = []
        try:
            if os.path.exists(SYNC_LOG_FILE):
                with open(SYNC_LOG_FILE, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
                    if not isinstance(logs, list): # Ensure it's a list
                        logs = []
        except (json.JSONDecodeError, IOError) as e:
            print(f"讀取同步日誌 {SYNC_LOG_FILE} 時發生錯誤: {e}. 將建立新的日誌列表。")
            logs = []

        logs.insert(0, log_entry) # Add new entry to the beginning (for chronological order when reading)
        
        # Keep only the last MAX_SYNC_LOG_ENTRIES
        if len(logs) > MAX_SYNC_LOG_ENTRIES:
            logs = logs[:MAX_SYNC_LOG_ENTRIES]
            
        try:
            with open(SYNC_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"寫入同步日誌 {SYNC_LOG_FILE} 時發生錯誤: {e}")

    def update_knowledge_base(self, force_rebuild=False, source_filter: Optional[str] = None):
        """
        Updates the knowledge base by processing new documents and embedding them.
        Can perform a full rebuild or update from a specific source.

        Args:
            force_rebuild (bool): If True, clears existing index for the affected scope.
                                  If source_filter is None, clears everything.
                                  If source_filter is specified, it's more complex;
                                  currently, for simplicity, force_rebuild with a source_filter
                                  will re-process that source and add to existing index
                                  after potentially removing old items from that source (TODO for removal).
                                  For now, force_rebuild+source_filter means re-scrape and re-embed that source.
            source_filter (Optional[str]): If specified (e.g., "website", "word"),
                                           only processes data from that source.
                                           If None, processes all configured sources.
        """
        operation_description = f"force_rebuild={force_rebuild}, source_filter={source_filter if source_filter else 'all'}"
        log_operation_type = "full_rebuild" if not source_filter else f"source_sync_{source_filter}"
        log_source_name_for_op = source_filter if source_filter else "overall"

        print(f"開始更新知識庫... {operation_description}")
        self._add_sync_log_entry(log_operation_type, log_source_name_for_op, "started", f"知識庫更新開始: {operation_description}")
        
        current_kb_status = self._read_kb_status()
        operation_timestamp = datetime.now().isoformat()

        # Determine which sources to process
        sources_to_process = []
        if source_filter:
            if source_filter in current_kb_status["sources"]:
                sources_to_process.append(source_filter)
            else:
                print(f"警告：未知的資料來源過濾器 '{source_filter}'，將不處理任何特定來源。")
                # Update status for this unknown filter attempt? Or just log and return?
                return
        else: # Process all known sources
            sources_to_process = list(current_kb_status["sources"].keys())

        if not sources_to_process:
            print("沒有定義或選擇任何資料來源進行處理。")
            # Log this event
            message = "沒有定義或選擇任何資料來源進行處理。"
            self._add_sync_log_entry(log_operation_type, log_source_name_for_op, "skipped", message)
            # Update status file (minimal update to reflect attempt)
            current_kb_status["overall_status"]["last_any_sync_timestamp"] = operation_timestamp
            if not source_filter: current_kb_status["overall_status"]["last_full_rebuild_timestamp"] = operation_timestamp
            current_kb_status["overall_status"]["message"] = message
            self._write_kb_status(current_kb_status)
            return

        # If it's a full rebuild (no source_filter) and force_rebuild is True
        if not source_filter and force_rebuild:
            print("執行完整強制重建：清空現有索引和中繼資料。")
            self.index = None
            self.doc_metadata = []
            # Reset parts of the status that are about a full build
            current_kb_status["overall_status"]["total_indexed_vectors"] = 0
            for src in current_kb_status["sources"]: # Reset all source specific stats
                current_kb_status["sources"][src].update({
                    "status": "pending_rebuild", "message": "等待完整重建",
                    "processed_items": 0, "embedded_items": 0
                })

        all_raw_chunks_from_sources = []
        source_specific_stats = {src: {"raw_count": 0, "unique_count": 0, "embedded_count": 0, "status": "pending", "message": ""} for src in sources_to_process}

        for current_source in sources_to_process:
            print(f"--- 開始處理來源: {current_source} ---")
            source_stats = source_specific_stats[current_source]
            source_config = current_kb_status["sources"].get(current_source, {}) # Get current config/status for the source
            source_status_update = {"status": "processing", "message": "正在處理中...", "last_sync_timestamp": operation_timestamp}

            raw_chunks_this_source = []
            try:
                if current_source == "website":
                    # 在呼叫 process_all_documents 之前更新狀態
                    current_kb_status["sources"]["website"].update(source_status_update)
                    self._write_kb_status(current_kb_status) # Write intermediate status
                    raw_chunks_this_source = website_scraper.process_all_documents()
                elif current_source == "word_documents":
                     # 在呼叫 process_all_documents 之前更新狀態
                    current_kb_status["sources"]["word_documents"].update(source_status_update)
                    self._write_kb_status(current_kb_status) # Write intermediate status
                    raw_chunks_this_source = word_processor.process_all_documents()
                elif current_source == "google_drive": # 新增 Google Drive 處理
                    current_kb_status["sources"]["google_drive"].update(source_status_update)
                    self._write_kb_status(current_kb_status)
                    raw_chunks_this_source = google_drive_processor.process_all_documents()
                elif current_source == "smb": # 新增 SMB 處理
                    current_kb_status["sources"]["smb"].update(source_status_update)
                    self._write_kb_status(current_kb_status)
                    raw_chunks_this_source = smb_processor.process_all_documents()
                # Add other sources here with elif current_source == "new_source_key":
                else:
                    print(f"未知的資料來源類型: {current_source}，跳過。")
                    source_stats["status"] = "error"
                    source_stats["message"] = f"未知的資料來源類型 {current_source}"
                    # Update status for this specific source
                    current_kb_status["sources"][current_source].update({
                        "status": "error", 
                        "message": f"未知的資料來源類型",
                        "last_sync_timestamp": operation_timestamp 
                    })
                    continue # Skip to next source

                source_stats["raw_count"] = len(raw_chunks_this_source)
                print(f"來源 {current_source} 原始提取了 {source_stats['raw_count']} 個區塊。")
                if not raw_chunks_this_source:
                    source_stats["status"] = "success_no_new_data"
                    source_stats["message"] = "成功處理，但未提取到新的資料區塊。"
                else:
                    source_stats["status"] = "success_processed" # Temporary status
                    source_stats["message"] = f"成功提取 {source_stats['raw_count']} 個原始區塊，等待嵌入。"
                
                # Update status immediately after processing this source
                current_kb_status["sources"][current_source].update({
                    "status": source_stats["status"],
                    "message": source_stats["message"],
                    "processed_items": source_stats["raw_count"], # 'processed' refers to raw chunks extracted
                    "embedded_items": 0 # Reset embedded count for this sync, will be updated later
                })

            except Exception as e:
                error_message = f"處理來源 {current_source} 時發生錯誤: {e}"
                print(error_message)
                self._add_sync_log_entry(log_operation_type, current_source, "error", error_message)
                source_stats["status"] = "error"
                source_stats["message"] = str(e)
                current_kb_status["sources"][current_source].update({
                    "status": "error", "message": str(e), "last_sync_timestamp": operation_timestamp,
                    "processed_items": 0, "embedded_items": 0
                })
            
            all_raw_chunks_from_sources.extend(raw_chunks_this_source)
            # Write status after each source is processed (or attempted)
            self._write_kb_status(current_kb_status)

        # --- 全局處理和嵌入 ---
        if not all_raw_chunks_from_sources and not force_rebuild: # Only print if not a forced clean rebuild
            print("所有來源均未提取到新的文本區塊。知識庫未作更改 (除非是強制重建)。")
            # Update overall status
            current_kb_status["overall_status"]["last_any_sync_timestamp"] = operation_timestamp
            if not source_filter: current_kb_status["overall_status"]["last_full_rebuild_timestamp"] = operation_timestamp # If it was meant to be a full run
            final_message = "所有來源均未提取到新的文本區塊。"
            if force_rebuild and not source_filter : final_message = "完整強制重建完成，但未從任何來源提取到資料。"

            current_kb_status["overall_status"]["message"] = final_message
            # Ensure individual source statuses reflect no new data if they were successful but empty
            for src in sources_to_process:
                if current_kb_status["sources"][src]["status"] == "success_processed": # If it was marked as processed but resulted in no global chunks
                     current_kb_status["sources"][src]["status"] = "success_no_new_data"
                     current_kb_status["sources"][src]["message"] = "成功處理，但未提取到新的資料區塊 (或未被納入最終列表)。"
            self._write_kb_status(current_kb_status)
            self._add_sync_log_entry(log_operation_type, log_source_name_for_op, "success_no_new_data", final_message)
            return
        
        print(f"從所有已處理來源總共收集到 {len(all_raw_chunks_from_sources)} 個原始文本區塊。準備進行嵌入。")

        # 篩選掉重複的區塊 (基於 text 和 source) - 這是一個簡單的去重
        # TODO: 更細緻的去重策略，例如考慮到如果 source_filter 被使用，
        #       我們可能不希望移除其他來源先前已存在的相同內容，除非是 force_rebuild。
        #       目前的邏輯是：如果 force_rebuild + source_filter，舊的 source data 實際上不會被移除，
        #       新的會被加入。如果只是 source_filter (no force_rebuild)，新的會被加入。
        #       如果 full force_rebuild，所有東西都會被清空重建。
        
        # 如果不是針對特定來源的強制重建，那麼在加入新區塊前，需要考慮如何處理舊的同來源區塊。
        # 目前的實現是直接加入。若要實現 source-specific rebuild，需要先從 self.doc_metadata 和 self.index 中移除該來源的舊條目。
        # 這部分邏輯比較複雜，暫時簡化處理：force_rebuild 清空所有，否則追加。

        unique_chunks_to_embed = []
        if not force_rebuild or source_filter: # If not a full clean rebuild, check for existing
            existing_chunk_signatures = set()
            if self.doc_metadata:
                for meta in self.doc_metadata:
                    # Signature considers text and a more specific source identifier if available
                    # For example, meta might have 'source_file_path' or similar unique ID
                    # For now, using meta['text'] and meta['source'] as a basic signature
                    existing_chunk_signatures.add((meta.get('text', ''), meta.get('source', '')))

            for chunk in all_raw_chunks_from_sources:
                chunk_signature = (chunk.get('text', ''), chunk.get('source', ''))
                if chunk_signature not in existing_chunk_signatures:
                    unique_chunks_to_embed.append(chunk)
                    existing_chunk_signatures.add(chunk_signature) # Add to set to avoid duplicates from current batch
            
            print(f"過濾後，有 {len(unique_chunks_to_embed)} 個新的唯一區塊需要嵌入。")
            if not unique_chunks_to_embed and all_raw_chunks_from_sources: # Had raw, but all were duplicates
                final_message = "提取到資料區塊，但均為現有知識庫中已存在的重複內容。"
                print(final_message)
                current_kb_status["overall_status"]["last_any_sync_timestamp"] = operation_timestamp
                if not source_filter: current_kb_status["overall_status"]["last_full_rebuild_timestamp"] = operation_timestamp
                current_kb_status["overall_status"]["message"] = final_message
                # Update source statuses to reflect duplicates found if they successfully processed data
                for src in sources_to_process:
                    if current_kb_status["sources"][src]["status"] == "success_processed":
                        current_kb_status["sources"][src]["status"] = "success_duplicates_found"
                        current_kb_status["sources"][src]["message"] = "成功提取資料，但均為重複內容。"
                        current_kb_status["sources"][src]["embedded_items"] = 0 # No new items embedded
                self._write_kb_status(current_kb_status)
                self._add_sync_log_entry(log_operation_type, log_source_name_for_op, "success_duplicates_found", final_message)
                return
        else: # Full force_rebuild, so all raw chunks are "unique" for this new index
            unique_chunks_to_embed = all_raw_chunks_from_sources
            print(f"完整重建模式：所有 {len(unique_chunks_to_embed)} 個提取的區塊將被嵌入。")


        if not unique_chunks_to_embed:
            final_message = "沒有新的唯一文本區塊可供嵌入。"
            if force_rebuild and not source_filter: final_message = "完整強制重建完成，但未找到可嵌入的新資料。"
            print(final_message)
            current_kb_status["overall_status"]["last_any_sync_timestamp"] = operation_timestamp
            if not source_filter: current_kb_status["overall_status"]["last_full_rebuild_timestamp"] = operation_timestamp
            current_kb_status["overall_status"]["message"] = final_message
            self._write_kb_status(current_kb_status)
            self._add_sync_log_entry(log_operation_type, log_source_name_for_op, "success_no_new_embeddings", final_message)
            return

        # Embedding the new unique chunks
        new_embeddings = []
        new_metadata_for_embeddings = []
        embedded_count_by_source = {src: 0 for src in sources_to_process}

        for i, chunk_data in enumerate(unique_chunks_to_embed):
            print(f"  正在嵌入區塊 {i+1}/{len(unique_chunks_to_embed)} (來源: {chunk_data.get('type', 'N/A')}, 標題: {chunk_data.get('title', 'N/A')})...")
            embedding = self._get_embedding(chunk_data["text"])
            if embedding is not None:
                new_embeddings.append(embedding)
                new_metadata_for_embeddings.append(chunk_data) # Store the whole chunk_data as metadata
                # Increment embedded count for the source of this chunk
                source_type = chunk_data.get("type") # e.g., "website_page", "word_document", "google_drive_document", "smb_document"
                # Map source_type back to source_key used in current_kb_status["sources"]
                source_key_for_stats = None
                if source_type == "website_page": source_key_for_stats = "website"
                elif source_type == "word_document": source_key_for_stats = "word_documents"
                elif source_type == "google_drive_document": source_key_for_stats = "google_drive"
                elif source_type == "smb_document": source_key_for_stats = "smb"
                
                if source_key_for_stats and source_key_for_stats in embedded_count_by_source:
                    embedded_count_by_source[source_key_for_stats] += 1
            else:
                print(f"    警告：無法為來源 '{chunk_data.get('source')}' 的區塊生成嵌入，已跳過。")
                # Log this specific chunk embedding failure if necessary

        if not new_embeddings:
            final_message = "所有提取到的唯一區塊都未能成功嵌入。"
            print(final_message)
            current_kb_status["overall_status"]["last_any_sync_timestamp"] = operation_timestamp
            if not source_filter: current_kb_status["overall_status"]["last_full_rebuild_timestamp"] = operation_timestamp
            current_kb_status["overall_status"]["message"] = final_message
            # Update source statuses if they had unique chunks but none embedded
            for src_key in sources_to_process:
                # Check if this source contributed to unique_chunks_to_embed
                source_had_unique_chunks = any(chunk.get("type") and 
                                               ((chunk.get("type") == "website_page" and src_key == "website") or \
                                                (chunk.get("type") == "word_document" and src_key == "word_documents") or \
                                                (chunk.get("type") == "google_drive_document" and src_key == "google_drive") or \
                                                (chunk.get("type") == "smb_document" and src_key == "smb")) \
                                               for chunk in unique_chunks_to_embed)
                if source_had_unique_chunks and current_kb_status["sources"][src_key]["status"] not in ["error", "success_no_new_data"]:
                    current_kb_status["sources"][src_key]["status"] = "error_embedding"
                    current_kb_status["sources"][src_key]["message"] = "提取到唯一區塊，但嵌入失敗。"
                    current_kb_status["sources"][src_key]["embedded_items"] = 0

            self._write_kb_status(current_kb_status)
            self._add_sync_log_entry(log_operation_type, log_source_name_for_op, "error_embedding_all", final_message)
            return

        # Convert to numpy array for FAISS
        new_embeddings_np = np.array(new_embeddings, dtype=np.float32)

        # Update FAISS index
        if self.index is None or (not source_filter and force_rebuild): # If new index or full forced rebuild
            if new_embeddings_np.shape[0] > 0: # Ensure there's something to build an index with
                dimension = new_embeddings_np.shape[1]
                self.index = faiss.IndexFlatL2(dimension) # Using L2 distance
                self.index.add(new_embeddings_np)
                self.doc_metadata = new_metadata_for_embeddings
                print(f"已建立新的 FAISS 索引，包含 {self.index.ntotal} 個向量。")
            else: # This case should ideally be caught earlier
                print("沒有可供建立新索引的嵌入。")
                # Status updates should have happened already
                return 
        else: # Adding to existing index
            if new_embeddings_np.shape[0] > 0: # Ensure there's something to add
                self.index.add(new_embeddings_np)
                self.doc_metadata.extend(new_metadata_for_embeddings)
                print(f"已將 {new_embeddings_np.shape[0]} 個新向量加入現有 FAISS 索引。總數: {self.index.ntotal}。")
            else:
                print("沒有新的嵌入可以加入現有索引。") 
                # This implies unique_chunks_to_embed was empty or all failed to embed,
                # which should have been handled by earlier return statements.

        # Save the updated index and metadata
        self._save_vector_store()

        # Final status update
        current_kb_status["overall_status"]["last_any_sync_timestamp"] = operation_timestamp
        if not source_filter: current_kb_status["overall_status"]["last_full_rebuild_timestamp"] = operation_timestamp
        current_kb_status["overall_status"]["total_indexed_vectors"] = self.index.ntotal if self.index else 0
        
        final_overall_message = f"知識庫更新成功。新增了 {len(new_embeddings)} 個嵌入。總共 {current_kb_status['overall_status']['total_indexed_vectors']} 個向量。"
        if not source_filter and force_rebuild:
            final_overall_message = f"知識庫完整重建成功。總共 {current_kb_status['overall_status']['total_indexed_vectors']} 個向量。"
        elif source_filter and force_rebuild: # This implies a source-specific re-embed.
            final_overall_message = f"來源 '{source_filter}' 資料已成功重新嵌入 ({len(new_embeddings)} 個區塊)。總知識庫大小: {current_kb_status['overall_status']['total_indexed_vectors']}。"
        elif source_filter and not force_rebuild:
            final_overall_message = f"來源 '{source_filter}' 資料已成功同步並新增 {len(new_embeddings)} 個嵌入。總知識庫大小: {current_kb_status['overall_status']['total_indexed_vectors']}。"


        current_kb_status["overall_status"]["message"] = final_overall_message
        print(final_overall_message)

        # Update individual source statuses with embedded counts
        for src_key, count in embedded_count_by_source.items():
            if src_key in current_kb_status["sources"]:
                current_kb_status["sources"][src_key]["embedded_items"] = count
                if count > 0 :
                    current_kb_status["sources"][src_key]["status"] = "success"
                    current_kb_status["sources"][src_key]["message"] = f"成功同步並嵌入 {count} 個區塊。"
                elif current_kb_status["sources"][src_key]["status"] not in ["error", "success_no_new_data", "success_duplicates_found"]:
                    # If it was 'processing' or 'success_processed' but 0 embedded from this source
                    # (e.g. all its chunks failed to embed, or it had no unique chunks for embedding)
                    # This needs careful state tracking. If it had unique chunks, but they all failed, it's an error_embedding.
                    # If it had raw chunks, but none were unique, it's success_duplicates_found.
                    # If it had no raw chunks, it's success_no_new_data.
                    # The status should have been set earlier for these cases.
                    # This final update mainly confirms the 'success' if count > 0.
                    pass


        self._write_kb_status(current_kb_status)
        self._add_sync_log_entry(log_operation_type, log_source_name_for_op, "success", final_overall_message)

        print("知識庫更新流程完成。")

    def search_knowledge_base(self, query_text, k=5):
        '''Searches the knowledge base for text chunks similar to the query_text.
        
        Args:
            query_text (str): The user's query.
            k (int): The number of top similar chunks to retrieve.
            
        Returns:
            list: A list of dictionaries, where each dictionary contains
                  the 'text', 'source', 'title', and 'score' (similarity) 
                  of a retrieved chunk.
        '''
        if self.index is None or self.index.ntotal == 0:
            print("知識庫為空或未初始化。無法執行搜尋。")
            return []

        print(f"正在搜尋知識庫，查詢: '{query_text[:100]}...'")
        query_embedding = self._get_embedding(query_text)
        if query_embedding is None:
            print("無法為查詢生成嵌入，搜尋中止。")
            return []

        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        # FAISS 搜尋 (L2 距離，所以距離越小越相似)
        distances, indices = self.index.search(query_embedding_np, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx == -1: # FAISS 可能返回 -1 如果找不到足够的鄰居
                continue
            distance = distances[0][i]
            metadata = self.doc_metadata[idx]
            results.append({
                "text": metadata["text"],
                "source": metadata["source"],
                "title": metadata["title"],
                "score": float(distance), # L2 距離，越小越好
                "original_chunk_info": metadata.get("original_chunk_info")
            })
        
        print(f"搜尋完成，找到 {len(results)} 個結果。")
        return results

    def get_kb_status(self):
        """Returns the current status of the knowledge base."""
        return self._read_kb_status()

# --- Main execution for testing --- 
if __name__ == '__main__':
    print("--- 測試 KnowledgeBaseManager ---")
    
    # !!! 重要：執行此測試前，請確保您的 OPENAI_API_KEY 環境變數已設定 !!!
    # 例如: export OPENAI_API_KEY='sk-your_actual_key_here' (Linux/macOS)
    #       set OPENAI_API_KEY=sk-your_actual_key_here (Windows CMD)
    #       $Env:OPENAI_API_KEY='sk-your_actual_key_here' (Windows PowerShell)
    if not OPENAI_API_KEY:
        print("\n!!! 錯誤：OPENAI_API_KEY 環境變數未設定。請設定後再執行測試。")
        print("測試中止。")
    else:
        try:
            kb_manager = KnowledgeBaseManager(openai_api_key=OPENAI_API_KEY)
            
            # 1. 更新/建立知識庫 (這會觸發 website_scraper)
            # 設定 force_rebuild=True 會清空並重建知識庫
            # 首次執行時，或者 website_scraper.py 的設定有變動時，建議設為 True
            print("\n--- 步驟 1: 更新知識庫 ---")
            kb_manager.update_knowledge_base(force_rebuild=True) # 第一次通常建議 force_rebuild
            
            # 2. 測試搜尋功能
            if kb_manager.index and kb_manager.index.ntotal > 0:
                print("\n--- 步驟 2: 測試搜尋 ---")
                test_queries = [
                    "肩膀痛怎麼辦？",
                    "關於五十肩的治療",
                    "最新的運動傷害資訊"
                ]
                for query in test_queries:
                    print(f"\n查詢: {query}")
                    search_results = kb_manager.search_knowledge_base(query, k=3)
                    if search_results:
                        for i, res in enumerate(search_results):
                            print(f"  結果 {i+1}:")
                            print(f"    標題: {res['title']}")
                            print(f"    來源: {res['source']}")
                            print(f"    分數 (L2距離): {res['score']:.4f}") # 距離越小越相關
                            text_snippet = res['text'][:150].replace('\n', ' ')
                            print(f"    文本片段: {text_snippet}...")
                    else:
                        print("  未找到相關結果。")
            else:
                print("\n知識庫為空，跳過搜尋測試。請檢查 update_knowledge_base 步驟的日誌。")
                
        except ValueError as ve:
            print(f"初始化 KnowledgeBaseManager 失敗: {ve}")
        except Exception as e:
            print(f"測試過程中發生未預期的錯誤: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- KnowledgeBaseManager 測試完成 ---") 