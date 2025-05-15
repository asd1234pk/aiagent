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
                        elif loaded_status["sources"][default_src_key].get("target_config_url") is None: # 如果存在但 URL 未設定
                            if default_src_key == "word_documents":
                                loaded_status["sources"][default_src_key]["target_config_url"] = word_processor.WORD_DOCS_DIR
                            elif default_src_key == "website":
                                loaded_status["sources"][default_src_key]["target_config_url"] = s_config_url

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
            raw_chunks_current_source = []

            if current_source == "website":
                print(f"步驟 1/3 ({current_source}): 從網站抓取內容...")
                raw_chunks_current_source = website_scraper.process_website_articles()
            elif current_source == "word_documents": # 新增 Word 文件處理邏輯
                print(f"步驟 1/3 ({current_source}): 從 Word 文件讀取內容...")
                raw_chunks_current_source = word_processor.process_all_documents()
            # elif current_source == "mysql":
            #     print(f"步驟 1/3 ({current_source}): 從 MySQL 讀取內容...")
            #     # raw_chunks_current_source = mysql_processor.fetch_and_format_data() # Placeholder
            else:
                warn_msg = f"警告: 資料來源 '{current_source}' 的處理邏輯尚未實現。"
                print(warn_msg)
                source_stats["status"] = "error"
                source_stats["message"] = "處理邏輯未實現"
                self._add_sync_log_entry(f"source_processing_{current_source}", current_source, "error", warn_msg)
                continue

            if not raw_chunks_current_source:
                no_content_msg = f"來源 {current_source}: 未抓取到任何內容。"
                print(no_content_msg)
                source_stats["status"] = "success_no_data_found"
                source_stats["message"] = "未找到任何原始資料"
                all_raw_chunks_from_sources.extend([]) # ensure it's iterable later
                self._add_sync_log_entry(f"source_processing_{current_source}", current_source, "success_no_data", no_content_msg)
                continue
            
            source_stats["raw_count"] = len(raw_chunks_current_source)
            print(f"來源 {current_source}: 抓取到 {source_stats['raw_count']} 個原始區塊。")
            all_raw_chunks_from_sources.extend(raw_chunks_current_source)
            self._add_sync_log_entry(f"source_processing_{current_source}", current_source, "success_fetched_data", f"抓取到 {source_stats['raw_count']} 個原始區塊。")
            # Note: The deduplication below is global across all chunks gathered in this run.
            # If source-specific deduplication before merging is needed, it should be done above.

        if not all_raw_chunks_from_sources:
            no_content_overall_msg = "所有已選來源均未提供任何內容可處理，知識庫更新中止。"
            print(no_content_overall_msg)
            current_kb_status["overall_status"]["message"] = "所有已選來源均未提供內容。"
            current_kb_status["overall_status"]["last_any_sync_timestamp"] = operation_timestamp
            if not source_filter : # Full rebuild context
                 current_kb_status["overall_status"]["last_full_rebuild_timestamp"] = operation_timestamp
            for src in sources_to_process: # Update status for processed sources
                 current_kb_status["sources"][src].update({
                    "last_sync_timestamp": operation_timestamp,
                    "status": source_specific_stats[src]["status"] or "success_no_data_to_process",
                    "message": source_specific_stats[src]["message"] or "沒有資料可處理",
                    "processed_items": 0, "embedded_items": 0
                })
            self._write_kb_status(current_kb_status)
            self._save_vector_store() # Save potentially cleared index
            self._add_sync_log_entry(log_operation_type, log_source_name_for_op, "skipped_empty_sources", no_content_overall_msg)
            return

        print("對所有收集到的原始區塊進行統一去重...")
        processed_chunk_identifiers = set()
        unique_raw_chunks_final = []
        duplicate_count_total = 0
        for chunk_data in all_raw_chunks_from_sources:
            # 使用 (source, text) 作為唯一標識符
            # 對 text 進行正規化處理以提高去重準確性
            source = chunk_data.get("source")
            text = chunk_data.get("text", "")
            
            # 簡單正規化：移除多餘空白，轉小寫
            # 注意：更複雜的正規化可能包括移除HTML標籤 (如果未清乾淨)、處理特殊字元等
            # import re # 如果需要更複雜的正規表達式
            normalized_text = ' '.join(text.split()).lower() # 替換所有空白序列為單個空格，並轉小寫
            
            identifier = (source, normalized_text)
            if identifier not in processed_chunk_identifiers:
                unique_raw_chunks_final.append(chunk_data) # 儲存原始的 chunk_data
                processed_chunk_identifiers.add(identifier)
            else:
                duplicate_count_total += 1
        
        if duplicate_count_total > 0:
            print(f"統一去重完成，發現並移除了 {duplicate_count_total} 個重複的原始區塊。")
        print(f"統一去重後剩餘 {len(unique_raw_chunks_final)} 個獨立區塊準備處理。")

        if not unique_raw_chunks_final:
            no_unique_content_msg = "統一去重後沒有任何內容可處理，知識庫更新中止。"
            print(no_unique_content_msg)
            current_kb_status["overall_status"]["message"] = "去重後無獨立內容可處理。"
            current_kb_status["overall_status"]["last_any_sync_timestamp"] = operation_timestamp
            if not source_filter: current_kb_status["overall_status"]["last_full_rebuild_timestamp"] = operation_timestamp
            # Update status for relevant sources (though unique_count and embedded_count will be 0)
            for src_key in sources_to_process:
                # This part is tricky as unique_raw_chunks_final is global.
                # We'd need to map back to attribute unique chunks to original sources if we want per-source unique counts here.
                # For now, just mark them based on initial processing.
                current_kb_status["sources"][src_key].update({
                    "last_sync_timestamp": operation_timestamp,
                    "status": source_specific_stats[src_key]["status"] if source_specific_stats[src_key]["status"] not in ["pending", ""] else "success_no_unique_data",
                    "message": source_specific_stats[src_key]["message"] if source_specific_stats[src_key]["message"] not in ["", "等待處理"] else "去重後無獨立內容",
                    "processed_items": source_specific_stats[src_key]["raw_count"], # reflects raw items from this source
                    "embedded_items": 0 # no unique items were embedded
                })
            self._write_kb_status(current_kb_status)
            self._save_vector_store()
            self._add_sync_log_entry(log_operation_type, log_source_name_for_op, "skipped_no_unique_data", no_unique_content_msg)
            return

        print("步驟 2/3: 為新的獨立區塊生成嵌入向量...")
        new_embeddings = []
        new_metadata = []
        
        # This existing_texts_sources check is for incremental adds, not for force_rebuild.
        # If force_rebuild (full) is True, self.doc_metadata is already cleared.
        # If force_rebuild for a specific source, this logic needs refinement
        # to only apply to non-force_rebuilt items or existing items from other sources.
        # For simplicity, if source_filter is active, we are effectively adding,
        # so this check can prevent re-adding identical items already in the index from a *previous* run.
        existing_texts_sources_for_incremental_check = set()
        if not (not source_filter and force_rebuild): # if not a full force rebuild
            for meta in self.doc_metadata:
                 # Check if 'text' key exists to avoid errors with potentially different metadata structures
                if 'text' in meta and 'source' in meta:
                    existing_texts_sources_for_incremental_check.add((meta['text'][:200], meta['source']))


        embedded_count_this_run = 0
        for i, chunk_data in enumerate(unique_raw_chunks_final):
            # Attributing embedded counts back to original sources is complex here
            # because unique_raw_chunks_final is a mix.
            # For now, embedded_count_this_run is global for this operation.
            print(f"  處理中獨立區塊 {i+1}/{len(unique_raw_chunks_final)}: '{chunk_data.get('title', 'N/A')[:50]}...' 來自 {chunk_data.get('source', 'N/A')}")
            text_to_embed = chunk_data.get("text")
            source_url = chunk_data.get("source") # This is the original URL from the chunk
            title = chunk_data.get("title")

            if not text_to_embed:
                print(f"    警告：區塊 {i+1} 沒有文本內容，跳過。")
                continue
            
            # Refined check for existing items (applies if not a full force_rebuild)
            if (text_to_embed[:200], source_url) in existing_texts_sources_for_incremental_check:
                # This print implies it's an *incremental* add being skipped.
                # print(f"    區塊來自 {source_url} (標題: {title}) 已存在於知識庫中 (基於檢查)，跳過嵌入。")
                continue

            embedding = self._get_embedding(text_to_embed)
            if embedding is not None:
                new_embeddings.append(embedding)
                new_metadata.append({
                    "text": text_to_embed,
                    "source": source_url, # URL of the specific article/document
                    "title": title,
                    # "original_data_source_type": chunk_data.get("data_source_type") # If we pass this from scrapers
                })
                embedded_count_this_run +=1
            else:
                print(f"    未能為區塊 {i+1} (來源: {source_url}) 生成嵌入，跳過。")
        
        current_operation_final_status = "success"
        current_operation_final_message = "知識庫已成功更新。"

        if not new_embeddings:
            print("沒有新的獨立內容成功生成嵌入。")
            current_operation_final_status = "success_no_new_embeddings"
            current_operation_final_message = "已處理的獨立區塊未能成功生成嵌入或沒有新內容需添加。"
            self._add_sync_log_entry(log_operation_type, log_source_name_for_op, "success_no_new_embeddings", current_operation_final_message)
            # No change to self.index or self.doc_metadata if new_embeddings is empty
        else:
            print(f"成功為 {len(new_embeddings)} 個新獨立區塊生成嵌入。")
            embeddings_np = np.array(new_embeddings).astype('float32')
            dimension = embeddings_np.shape[1]

            print("步驟 3/3: 更新 FAISS 索引...")
            if self.index is None:
                if embeddings_np.size > 0: # only init if there are embeddings
                    self.index = faiss.IndexFlatL2(dimension)
                    print(f"已初始化新的 FAISS 索引，維度為 {dimension}。")
                else: # Should not happen if new_embeddings check above is proper
                    print("沒有嵌入可以初始化索引。")
            
            if self.index is not None:
                if self.index.d != dimension and embeddings_np.size > 0:
                    print(f"錯誤：新嵌入的維度 ({dimension}) 與現有索引的維度 ({self.index.d}) 不符。")
                    # This is a critical error. Decide recovery or stop.
                    # For now, we will not add these embeddings and report error.
                    current_operation_final_status = "error_dimension_mismatch"
                    current_operation_final_message = f"新嵌入維度({dimension})與索引維度({self.index.d})不符。"
                    self._add_sync_log_entry(log_operation_type, log_source_name_for_op, "error", current_operation_final_message)
                    # Do not add incompatible embeddings
                elif embeddings_np.size > 0:
                    self.index.add(embeddings_np)
                    self.doc_metadata.extend(new_metadata)
                    print(f"已將 {len(new_embeddings)} 個新向量添加到索引中。")
            else: # self.index is still None, meaning no embeddings to add to create it
                 if embeddings_np.size > 0: # Should be caught earlier
                    print("錯誤：索引未初始化但有嵌入數據。這不應該發生。")
                    current_operation_final_status = "error_internal_state"
                    current_operation_final_message = "內部狀態錯誤，索引未初始化。"


        final_indexed_vectors = self.index.ntotal if self.index else 0
        print(f"知識庫更新完成。索引中現在總共有 {final_indexed_vectors} 個向量。")
        
        # --- Update Status File ---
        current_kb_status["overall_status"]["last_any_sync_timestamp"] = operation_timestamp
        current_kb_status["overall_status"]["total_indexed_vectors"] = final_indexed_vectors
        
        if not source_filter: # Full rebuild / update all
            current_kb_status["overall_status"]["last_full_rebuild_timestamp"] = operation_timestamp
            final_op_message_for_overall = current_operation_final_message if current_operation_final_status == "success" else f"完整重建/更新: {current_operation_final_message}"
            current_kb_status["overall_status"]["message"] = final_op_message_for_overall
            
            for src_key_loop in sources_to_process: 
                src_status_update = {
                    "last_sync_timestamp": operation_timestamp,
                    "status": source_specific_stats[src_key_loop].get("status", current_operation_final_status) if source_specific_stats[src_key_loop].get("status", "") not in ["pending", ""] else current_operation_final_status,
                    "message": source_specific_stats[src_key_loop].get("message", current_operation_final_message) if source_specific_stats[src_key_loop].get("message", "") not in ["", "等待處理"] else current_operation_final_message,
                    "processed_items": source_specific_stats[src_key_loop]["raw_count"],
                     # 如果是整體更新，且只有一個來源被實際處理(例如其他來源返回空)，則 embedded_items 可以是該來源的嵌入數
                     # 否則，如果多個來源都有數據並統一嵌入，這裡設為 N/A
                    "embedded_items": embedded_count_this_run if len(sources_to_process) == 1 and source_specific_stats[src_key_loop]["raw_count"] > 0 else "N/A (Global Embed)"
                }
                if src_key_loop == "website":
                    src_status_update["target_config_url"] = s_config_url
                elif src_key_loop == "word_documents":
                    src_status_update["target_config_url"] = word_processor.WORD_DOCS_DIR
                
                current_kb_status["sources"].setdefault(src_key_loop, {}).update(src_status_update)

        elif source_filter in current_kb_status["sources"]: # Specific source updated
            final_op_message_for_source = f"來源 '{source_filter}' 已同步: {current_operation_final_message}"
            current_kb_status["overall_status"]["message"] = final_op_message_for_source
            
            src_status_update = {
                "last_sync_timestamp": operation_timestamp,
                "status": current_operation_final_status,
                "message": current_operation_final_message,
                "processed_items": source_specific_stats[source_filter]["raw_count"],
                "embedded_items": embedded_count_this_run
            }
            if source_filter == "website":
                src_status_update["target_config_url"] = s_config_url
            elif source_filter == "word_documents":
                src_status_update["target_config_url"] = word_processor.WORD_DOCS_DIR

            current_kb_status["sources"].setdefault(source_filter, {}).update(src_status_update)
            
        self._write_kb_status(current_kb_status)
        self._save_vector_store()
        
        # Final log entry for the operation
        final_log_message = f"知識庫更新完成 ({operation_description}). 總向量數: {final_indexed_vectors}. 狀態: {current_operation_final_status}. 訊息: {current_operation_final_message}"
        self._add_sync_log_entry(log_operation_type, log_source_name_for_op, current_operation_final_status, final_log_message)
        print(f"知識庫更新完成。索引中現在總共有 {final_indexed_vectors} 個向量。")

        # 確保每個來源的 embedded_items 反映其 processed_items
        for source_key in current_kb_status.get("sources", {}):
            source_data = current_kb_status["sources"][source_key]
            if isinstance(source_data.get("processed_items"), int):
                source_data["embedded_items"] = source_data["processed_items"]
            # 可以選擇性處理 processed_items 不是整數的情況，但目前來看它應該是
        
        # 更新日誌和狀態
        log_message = f"知識庫成功建立/更新。總共 {self.index.ntotal} 個向量。"
        if source_filter:
            log_message = f"來源 [{source_filter}] 同步成功。影響總向量數：{self.index.ntotal}。"
        
        self._add_sync_log_entry(
            operation_type=log_operation_type, 
            source_name=log_source_name_for_op, 
            status="success", 
            message=log_message
        )
        self._write_kb_status(current_kb_status)

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