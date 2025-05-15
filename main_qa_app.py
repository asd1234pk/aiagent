'''
Main Q&A Application for the Medical Assistant (with FastAPI)

This application orchestrates the RAG (Retrieval Augmented Generation) process.
It uses the KnowledgeBaseManager to retrieve relevant context from the knowledge base
and then uses an OpenAI LLM to generate answers based on the user's question and the retrieved context.
It also exposes an API endpoint using FastAPI.
'''
import os
from openai import OpenAI
from knowledge_base_manager import KnowledgeBaseManager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

# --- Configuration ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set. API calls will fail.")

LLM_MODEL = "gpt-4o" # Default LLM model, can also be part of prompt_settings.json if desired
FEEDBACK_FILE = "feedbacks.json"
PROMPT_SETTINGS_FILE = "prompt_settings.json" # Define the prompt settings file

# Default prompt settings (if file is missing or invalid)
DEFAULT_SYSTEM_MESSAGE = "您是一個基礎的 AI 助理。請根據上下文回答問題。"
DEFAULT_USER_PROMPT_TEMPLATE = "上下文資訊：\n{context_str}\n---\n使用者的問題：{user_question}\n---\nAI 助理的回答："
DEFAULT_TEMPERATURE = 0.7

# Global variable to hold the app instance
# This will be populated by the lifespan manager
assistant_app_instance = None

# --- FastAPI App Setup ---
# Create a dictionary to hold application state, including the assistant app instance
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Lifespan: Initializing Medical Assistant App...")
    # Directly assign the created instance to app_state
    app_state['assistant_app'] = MedicalAssistantApp()
    # No need for a local assistant_app_instance variable here that might cause confusion
    print("Lifespan: Medical Assistant App initialized.")
    yield
    # Clean up resources if needed when app shuts down
    print("Lifespan: Cleaning up resources...")
    app_state.clear()
    print("Lifespan: Cleanup complete.")

app = FastAPI(lifespan=lifespan)

# --- CORS Middleware Configuration ---
# Add CORSMiddleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# --- End CORS Middleware Configuration ---

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list # List of source dictionaries

class PromptSettingsRequest(BaseModel):
    system_message: str
    user_prompt_template: str
    temperature: float

class PromptSettingsResponse(BaseModel):
    system_message: str
    user_prompt_template: str
    temperature: float
    last_updated_by: Optional[str] = None
    last_updated_at: Optional[str] = None

class FeedbackRequest(BaseModel):
    question: Optional[str] = None # The original question, if available
    answer: str # The AI's answer that received feedback
    feedback_type: str # e.g., "helpful", "unhelpful", "incorrect", "other"
    details: Optional[str] = None # User's detailed textual feedback
    session_id: Optional[str] = None # Optional session ID for tracking

class FeedbackResponse(BaseModel): # For responding with the created feedback
    id: str
    question: Optional[str] = None
    answer: str
    feedback_type: str
    details: Optional[str] = None
    timestamp: str
    status: str
    admin_notes: Optional[str] = None

class FeedbackItem(FeedbackRequest): # Inherits fields from FeedbackRequest for display
    id: str
    timestamp: str
    status: str
    admin_notes: Optional[str] = None # New field for admin notes

class FeedbackUpdateRequest(BaseModel):
    status: Optional[str] = None
    admin_notes: Optional[str] = None

class MedicalAssistantApp:
    def __init__(self, knowledge_base_dir="vector_store"):
        '''
        Initializes the Medical Assistant application.

        Args:
            knowledge_base_dir (str): Directory where the FAISS index and metadata are stored.
        '''
        # print("Initializing Medical Assistant App...") # Moved to lifespan
        self.kb_manager = KnowledgeBaseManager(openai_api_key=OPENAI_API_KEY)
                                               
        if self.kb_manager.index is None:
            print("WARNING: Knowledge base index is not loaded. Search might fail or be empty.")
            print("Please ensure 'knowledge_base_manager.py' has run successfully.")

        if OPENAI_API_KEY:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.client = None
            print("WARNING: OpenAI client not initialized due to missing API key. LLM calls will fail.")
        # print("Medical Assistant App initialized.") # Moved to lifespan

        self._load_prompt_settings()

    def _load_prompt_settings(self):
        try:
            if os.path.exists(PROMPT_SETTINGS_FILE):
                with open(PROMPT_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    self.system_message = settings.get("system_message", DEFAULT_SYSTEM_MESSAGE)
                    self.user_prompt_template = settings.get("user_prompt_template", DEFAULT_USER_PROMPT_TEMPLATE)
                    self.temperature = settings.get("temperature", DEFAULT_TEMPERATURE)
                    print(f"Successfully loaded prompt settings from {PROMPT_SETTINGS_FILE}.")
                    print(f"Loaded temperature: {self.temperature}")
            else:
                print(f"Warning: {PROMPT_SETTINGS_FILE} not found. Using default prompt settings.")
                self._use_default_prompt_settings()
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {PROMPT_SETTINGS_FILE}. Using default prompt settings.")
            self._use_default_prompt_settings()
        except Exception as e:
            print(f"Error loading {PROMPT_SETTINGS_FILE}: {e}. Using default prompt settings.")
            self._use_default_prompt_settings()

    def _use_default_prompt_settings(self):
        self.system_message = DEFAULT_SYSTEM_MESSAGE
        self.user_prompt_template = DEFAULT_USER_PROMPT_TEMPLATE
        self.temperature = DEFAULT_TEMPERATURE

    def reload_prompt_settings(self): # Method to allow reloading settings later if needed
        print("Reloading prompt settings...")
        self._load_prompt_settings()

    def trigger_kb_rebuild(self):
        """
        Triggers a full rebuild of the knowledge base (all sources).
        This can be a long-running operation.
        """
        print("Full knowledge base rebuild triggered in MedicalAssistantApp...")
        try:
            self.kb_manager.update_knowledge_base(force_rebuild=True, source_filter=None)
            print("Full knowledge base rebuild process completed successfully via MedicalAssistantApp.")
            return {"status": "success", "message": "知識庫已成功觸發完整重建並完成。"}
        except Exception as e:
            print(f"Error during full knowledge base rebuild: {e}")
            return {"status": "error", "message": f"知識庫完整重建過程中發生錯誤: {e}"}

    def trigger_source_specific_sync(self, source_name: str):
        """
        Triggers a sync/update for a specific data source.
        Args:
            source_name (str): The name of the source to sync (e.g., 'website').
        """
        print(f"Sync triggered for source '{source_name}' in MedicalAssistantApp...")
        try:
            # force_rebuild=False for typical incremental sync of a source.
            # If a source needs a full re-scrape, KnowledgeBaseManager might handle that
            # internally or we might need a different flag.
            self.kb_manager.update_knowledge_base(force_rebuild=False, source_filter=source_name)
            print(f"Sync process for source '{source_name}' completed successfully.")
            return {"status": "success", "message": f"資料來源 '{source_name}' 已成功觸發同步並完成。"}
        except Exception as e:
            print(f"Error during sync for source '{source_name}': {e}")
            return {"status": "error", "message": f"資料來源 '{source_name}' 同步過程中發生錯誤: {e}"}

    def answer_question(self, user_question, top_k=3):
        '''
        Answers a user's question using RAG.

        Args:
            user_question (str): The question from the user.
            top_k (int): The number of relevant chunks to retrieve from the knowledge base.

        Returns:
            tuple: (answer_text, list_of_source_documents)
                   Returns (None, []) if an error occurs or no answer can be generated.
        '''
        if self.kb_manager.index is None or not self.kb_manager.doc_metadata:
            print("Knowledge base is not available or empty. Cannot answer questions.")
            return "知識庫尚未建立或為空，無法回答問題。", []

        # print(f"\nReceived question: {user_question}") # Logging for API can be different
        # print(f"Searching knowledge base for top {top_k} relevant documents...")
        
        retrieved_docs = self.kb_manager.search_knowledge_base(user_question, k=top_k)

        if not retrieved_docs:
            # print("No relevant documents found in the knowledge base for this question.")
            return "抱歉，我目前找不到與您問題直接相關的資訊。", []

        # print(f"Found {len(retrieved_docs)} relevant documents.")
        
        context_parts = []
        sources_for_response = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"相關資訊片段 {i+1} (來源: {doc['source']}, 標題: {doc.get('title', 'N/A')}):\n{doc['text']}\n---\n")
            sources_for_response.append({
                "title": doc.get('title', 'N/A'),
                "url": doc['source'],
                "score": doc.get('score', 0) 
            })
        
        context_str = "\n".join(context_parts)

        # Use loaded or default prompt settings
        current_prompt = self.user_prompt_template.format(
            context_str=context_str,
            user_question=user_question
        )

        if not self.client:
            print("ERROR: OpenAI client not available. Cannot generate LLM answer.")
            return "AI 模型客戶端未初始化，無法生成回答。請檢查 API 金鑰設定。", sources_for_response

        # print("Generating answer using LLM...")
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": current_prompt}
                ],
                temperature=self.temperature, 
            )
            answer = response.choices[0].message.content.strip()
            # print(f"LLM Answer: {answer}")
            return answer, sources_for_response
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"呼叫 AI 模型時發生錯誤: {e}", sources_for_response

# Function to load feedbacks (similar to existing logic in get_all_feedbacks)
def load_feedbacks_from_file() -> List[FeedbackItem]:
    if not os.path.exists(FEEDBACK_FILE):
        return []
    try:
        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            return [FeedbackItem(**item) for item in json.load(f)]
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading feedbacks: {e}")
        return []

# Function to save feedbacks
def save_feedbacks_to_file(feedbacks: List[FeedbackItem]):
    try:
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump([fb.model_dump() for fb in feedbacks], f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"Error saving feedbacks: {e}")

# Ensure assistant_app_instance is available for routes
# This relies on the lifespan event populating app_state['assistant_app']
def get_assistant_app() -> MedicalAssistantApp:
    if 'assistant_app' not in app_state or app_state['assistant_app'] is None:
        # This case should ideally not happen if lifespan is working correctly
        # and requests are made after startup.
        print("CRITICAL: MedicalAssistantApp not initialized in app_state!")
        # Fallback or error, though this path suggests a deeper issue
        # For now, let's try to initialize it here as a last resort, though this is not ideal
        # as it might not have the full context or might be re-initialized multiple times.
        # A better approach would be to raise an HTTPException to indicate server error.
        raise HTTPException(status_code=503, detail="Assistant service not ready. Please try again shortly.")
    return app_state['assistant_app']

@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    # Get the assistant app instance from app_state populated by lifespan manager
    assistant_app = get_assistant_app()
    if not assistant_app: # Should be handled by get_assistant_app raising HTTPException
        raise HTTPException(status_code=503, detail="AI Service not available.")
    
    answer_text, sources = assistant_app.answer_question(request.question)
    if answer_text is None:
        raise HTTPException(status_code=500, detail="Error generating answer")
    return AnswerResponse(answer=answer_text, sources=sources)

@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback_data: FeedbackRequest):
    feedbacks = load_feedbacks_from_file()
    
    new_feedback_id = f"fb_{int(datetime.now().timestamp() * 1000)}_{len(feedbacks) + 1}"
    
    new_feedback = FeedbackItem(
        id=new_feedback_id,
        timestamp=datetime.now().isoformat(),
        question=feedback_data.question,
        answer=feedback_data.answer,
        feedback_type=feedback_data.feedback_type,
        details=feedback_data.details,
        session_id=feedback_data.session_id, # Make sure this is included if present in FeedbackRequest
        status="pending_review", # Default status for new feedback
        admin_notes=None
    )
    
    feedbacks.append(new_feedback)
    save_feedbacks_to_file(feedbacks)
    
    # Return the created feedback item, matching FeedbackResponse model
    return FeedbackResponse(
        id=new_feedback.id,
        question=new_feedback.question,
        answer=new_feedback.answer,
        feedback_type=new_feedback.feedback_type,
        details=new_feedback.details,
        timestamp=new_feedback.timestamp,
        status=new_feedback.status,
        admin_notes=new_feedback.admin_notes
    )

@app.get("/api/admin/prompt-settings", response_model=PromptSettingsResponse)
async def get_prompt_settings():
    try:
        if os.path.exists(PROMPT_SETTINGS_FILE):
            with open(PROMPT_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                return PromptSettingsResponse(
                    system_message=settings.get("system_message", DEFAULT_SYSTEM_MESSAGE),
                    user_prompt_template=settings.get("user_prompt_template", DEFAULT_USER_PROMPT_TEMPLATE),
                    temperature=settings.get("temperature", DEFAULT_TEMPERATURE),
                    last_updated_by=settings.get("last_updated_by"),
                    last_updated_at=settings.get("last_updated_at")
                )
        else:
            # If file doesn't exist, maybe return defaults or an error
            # For an editor, returning current defaults that would be saved is fine.
            return PromptSettingsResponse(
                system_message=DEFAULT_SYSTEM_MESSAGE,
                user_prompt_template=DEFAULT_USER_PROMPT_TEMPLATE,
                temperature=DEFAULT_TEMPERATURE,
                last_updated_by="system_default_not_saved",
                last_updated_at=None
            )
    except Exception as e:
        print(f"Error reading {PROMPT_SETTINGS_FILE} for admin: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read prompt settings: {e}")

@app.post("/api/admin/prompt-settings", response_model=PromptSettingsResponse)
async def update_prompt_settings(new_settings: PromptSettingsRequest):
    try:
        # Validate temperature (basic example)
        if not (0.0 <= new_settings.temperature <= 2.0): # OpenAI typical range for temperature
            raise HTTPException(status_code=400, detail="Temperature must be between 0.0 and 2.0.")

        updated_data = {
            "system_message": new_settings.system_message,
            "user_prompt_template": new_settings.user_prompt_template,
            "temperature": new_settings.temperature,
            "last_updated_by": "admin", # Placeholder, integrate auth later
            "last_updated_at": datetime.now().isoformat()
        }
        
        with open(PROMPT_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=4)
        
        # Reload settings in the running MedicalAssistantApp instance
        assistant_app = app_state.get('assistant_app')
        if assistant_app and hasattr(assistant_app, 'reload_prompt_settings'):
            assistant_app.reload_prompt_settings()
            print("Prompt settings reloaded in running MedicalAssistantApp instance.")
        else:
            print("Warning: Could not find running MedicalAssistantApp instance or reload_prompt_settings method to apply new settings immediately.")

        return PromptSettingsResponse(**updated_data) # Return the saved data

    except HTTPException as he: # Re-raise HTTPExceptions to ensure proper client response
        raise he
    except Exception as e:
        print(f"Error writing {PROMPT_SETTINGS_FILE} for admin: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update prompt settings: {e}")

# --- Admin API Endpoints for Feedback Review ---

@app.get("/api/admin/feedbacks", response_model=List[FeedbackItem])
async def get_all_feedbacks():
    if not os.path.exists(FEEDBACK_FILE):
        return []
    try:
        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedbacks_data = json.load(f)
            # Ensure it's a list, and optionally validate structure if needed
            if not isinstance(feedbacks_data, list):
                print(f"Warning: {FEEDBACK_FILE} does not contain a list.")
                return []
            # Sort by timestamp, newest first, for better review order
            feedbacks_data.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return [FeedbackItem(**fb) for fb in feedbacks_data] # Validate each item against the model
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {FEEDBACK_FILE} when getting all feedbacks.")
        return [] # Or raise HTTPException
    except Exception as e:
        print(f"Error reading {FEEDBACK_FILE} for admin feedback review: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read feedbacks: {e}")

@app.put("/api/admin/feedbacks/{feedback_id}", response_model=FeedbackItem)
async def update_feedback_status(feedback_id: str, update_data: FeedbackUpdateRequest):
    if not os.path.exists(FEEDBACK_FILE):
        raise HTTPException(status_code=404, detail="Feedback file not found.")

    updated = False
    updated_feedback_item = None
    try:
        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedbacks = json.load(f)
            if not isinstance(feedbacks, list):
                 raise HTTPException(status_code=500, detail="Feedback data is corrupted (not a list).")

        for i, fb in enumerate(feedbacks):
            if fb.get("id") == feedback_id:
                if update_data.status is not None:
                    fb["status"] = update_data.status
                if update_data.admin_notes is not None:
                    fb["admin_notes"] = update_data.admin_notes
                fb["last_updated_by_admin_at"] = datetime.now().isoformat() # Add admin update timestamp
                updated_feedback_item = FeedbackItem(**fb)
                feedbacks[i] = fb # Update the item in the list
                updated = True
                break
        
        if not updated:
            raise HTTPException(status_code=404, detail=f"Feedback with id {feedback_id} not found.")

        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedbacks, f, ensure_ascii=False, indent=4)
        
        return updated_feedback_item

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to decode feedback data.")
    except HTTPException as he: # Re-raise HTTPExceptions
        raise he
    except Exception as e:
        print(f"Error updating feedback {feedback_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update feedback: {e}")

# --- Admin API Endpoint for Knowledge Base Management ---
class KBRebuildResponse(BaseModel):
    status: str
    message: str

@app.post("/api/admin/knowledgebase/rebuild", response_model=KBRebuildResponse)
async def rebuild_knowledge_base_endpoint():
    current_assistant_app = app_state.get('assistant_app')
    if not current_assistant_app:
        print("ERROR in /api/admin/knowledgebase/rebuild: MedicalAssistantApp instance not found in app_state.")
        raise HTTPException(status_code=503, detail="醫療助理應用程式未正確初始化或不可用。")

    # This operation can be time-consuming.
    # For a production app, consider running this in a background task (e.g., using Celery with FastAPI).
    # For simplicity in this stage, we'll run it synchronously. The client will wait.
    print("API endpoint /api/admin/knowledgebase/rebuild called.")
    result = current_assistant_app.trigger_kb_rebuild()
    
    if result["status"] == "error":
        # Log the error on the server for more details if needed
        print(f"Rebuild API endpoint returning error: {result['message']}")
        raise HTTPException(status_code=500, detail=result["message"])
    
    print(f"Rebuild API endpoint returning success: {result['message']}")
    return KBRebuildResponse(status=result["status"], message=result["message"])

class OverallStatusModel(BaseModel):
    last_full_rebuild_timestamp: Optional[str] = None
    last_any_sync_timestamp: Optional[str] = None
    total_indexed_vectors: Optional[int] = None
    message: Optional[str] = None

class SourceStatusModel(BaseModel):
    last_sync_timestamp: Optional[str] = None
    status: Optional[str] = None
    message: Optional[str] = None
    processed_items: Optional[int] = None
    embedded_items: Optional[Union[int, str]] = None 
    target_config_url: Optional[str] = None

class KBStatusResponse(BaseModel):
    overall_status: Optional[OverallStatusModel] = None
    sources: Optional[Dict[str, SourceStatusModel]] = None
    status: Optional[str] = None 
    message: Optional[str] = None 
    error_message: Optional[str] = None 

@app.post("/api/admin/sync/{source_name}", response_model=KBRebuildResponse) 
async def sync_source_endpoint(source_name: str):
    current_assistant_app = app_state.get('assistant_app')
    if not current_assistant_app:
        print(f"ERROR in /api/admin/sync/{source_name}: MedicalAssistantApp instance not found.")
        raise HTTPException(status_code=503, detail="醫療助理應用程式未正確初始化或不可用。")

    # 更新支援的資料來源名稱列表
    if source_name not in ["website", "word_documents"]: 
        raise HTTPException(status_code=400, detail=f"不支援的資料來源名稱: {source_name}。目前僅支援 'website', 'word_documents'。")

    print(f"API endpoint /api/admin/sync/{source_name} called.")
    result = current_assistant_app.trigger_source_specific_sync(source_name)
    
    if result["status"] == "error":
        print(f"Sync API for source '{source_name}' returning error: {result['message']}")
        raise HTTPException(status_code=500, detail=result["message"])
    
    print(f"Sync API for source '{source_name}' returning success: {result['message']}")
    return KBRebuildResponse(status=result["status"], message=result["message"])

@app.get("/api/admin/knowledgebase/status", response_model=KBStatusResponse)
async def get_knowledge_base_status():
    kb_status_file_path = os.path.join("vector_store", "kb_status.json") 

    if not os.path.exists(kb_status_file_path):
        # Return the default structure but indicate it's not found
        default_status_for_not_found = DEFAULT_KB_STATUS.copy() # Assuming DEFAULT_KB_STATUS is defined in this file or imported
        return KBStatusResponse(
            overall_status=OverallStatusModel(**default_status_for_not_found["overall_status"]),
            sources={k: SourceStatusModel(**v) for k,v in default_status_for_not_found["sources"].items()},
            status="file_not_found", 
            message="知識庫狀態檔案不存在，可能尚未執行過任何同步作業。"
        )
    try:
        with open(kb_status_file_path, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
            # Validate and structure the data according to KBStatusResponse
            # This assumes status_data directly matches the nested structure
            return KBStatusResponse(
                overall_status=OverallStatusModel(**status_data.get("overall_status", {})),
                sources={k: SourceStatusModel(**v) for k,v in status_data.get("sources", {}).items()},
                status="success_loaded"
            )
    except json.JSONDecodeError:
        return KBStatusResponse(
            status="error_decode",
            message="無法解析知識庫狀態檔案內容。",
            error_message="JSONDecodeError"
        )
    except Exception as e:
        return KBStatusResponse(
            status="error_read",
            message=f"讀取知識庫狀態檔案時發生錯誤: {e}",
            error_message=str(e)
        )

# Make DEFAULT_KB_STATUS available to the API endpoint if it's not already
# (This would typically be defined at the top of the file or imported)
DEFAULT_KB_STATUS = {
    "overall_status": {
        "last_full_rebuild_timestamp": None,
        "last_any_sync_timestamp": None, 
        "total_indexed_vectors": 0,
        "message": "尚未進行任何同步作業。"
    },
    "sources": {
        "website": {
            "last_sync_timestamp": None,
            "status": "pending", 
            "message": "等待同步",
            "processed_items": 0,
            "embedded_items": 0,
            "target_config_url": None 
        },
        "word_documents": { # 新增 word_documents 的預設狀態
            "last_sync_timestamp": None,
            "status": "pending", 
            "message": "等待同步",
            "processed_items": 0,
            "embedded_items": 0,
            "target_config_url": None # 在 KnowledgeBaseManager 中會被實際路徑覆蓋
        }
        # mysql 之後也會加在這裡
    }
}

class SyncLogEntryModel(BaseModel):
    timestamp: str
    operation_type: str
    source_name: str
    status: str
    message: str

class SyncLogResponse(BaseModel):
    logs: List[SyncLogEntryModel]

@app.get("/api/admin/knowledgebase/sync-log", response_model=SyncLogResponse)
async def get_sync_log():
    sync_log_file_path = os.path.join("vector_store", "sync_log.json")
    logs_to_return = []
    if os.path.exists(sync_log_file_path):
        try:
            with open(sync_log_file_path, 'r', encoding='utf-8') as f:
                raw_logs = json.load(f)
                if isinstance(raw_logs, list):
                    # Validate each log entry with Pydantic model
                    for log_item in raw_logs:
                        try:
                            logs_to_return.append(SyncLogEntryModel(**log_item))
                        except Exception as pydantic_error: # Catch Pydantic validation error specifically if needed
                            print(f"Warning: Skipping log entry due to validation error: {pydantic_error}. Entry: {log_item}")
                else:
                    print(f"Warning: {sync_log_file_path} does not contain a list. Returning empty log.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {sync_log_file_path}. Returning empty log.")
            # Optionally raise HTTPException or return an error structure in SyncLogResponse
        except Exception as e:
            print(f"Error reading {sync_log_file_path}: {e}. Returning empty log.")
            # Optionally raise HTTPException
    return SyncLogResponse(logs=logs_to_return)

# --- Main execution for testing (commented out for FastAPI) ---
# if __name__ == '__main__':
#     print("--- 測試 Medical Assistant QA App ---")
#     
#     if not OPENAI_API_KEY:
#         print("請先設定 OPENAI_API_KEY 環境變數才能執行測試。")
#     else:
#         # This direct instantiation is now handled by lifespan manager for FastAPI
#         # assistant_app_direct_test = MedicalAssistantApp() 
# 
#         if assistant_app_direct_test.kb_manager.index is not None and assistant_app_direct_test.kb_manager.doc_metadata:
#             test_questions = [
#                 "肩膀痛怎麼辦？",
#                 "請問五十肩的治療方式有哪些？",
#                 "跑步膝蓋外側痛的原因？",
#                 "我想了解關於脊椎側彎的資訊",
#                 "什麼是足底筋膜炎？",
#                 "醫院的地址在哪裡？" 
#             ]
# 
#             for q in test_questions:
#                 answer, sources_info = assistant_app_direct_test.answer_question(q)
#                 print(f"\n問題: {q}")
#                 print(f"回答: {answer}")
#                 if sources_info:
#                     print("參考來源:")
#                     for i, src in enumerate(sources_info):
#                         print(f"  {i+1}. 標題: {src['title']}, 來源: {src['source_url']}, (分數: {src.get('score', 'N/A'):.4f})")
#                 print("-" * 30)
#         else:
#             print("知識庫未準備好，無法進行問答測試。請先執行 knowledge_base_manager.py 來建立知識庫。")
# 
#     print("--- Medical Assistant QA App 測試完成 ---")

# To run this FastAPI app:
# 1. Ensure OPENAI_API_KEY is set in your environment.
# 2. Install uvicorn: pip install uvicorn[standard]
# 3. Run: uvicorn main_qa_app:app --reload
#    (assuming your file is named main_qa_app.py and 'app' is the FastAPI instance)
# 4. Access the API at http://127.0.0.1:8000/docs for interactive documentation.

# ''' # Removed extra triple quote at the end of the file 