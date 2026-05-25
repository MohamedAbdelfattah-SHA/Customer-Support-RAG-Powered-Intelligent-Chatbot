import re
import json
import os
import sys#for docker
import shutil
import pandas as pd
import csv
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Ensure current directory is in path for imports
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Security, Depends, UploadFile, File, Form
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from config import Config
import evaluation
import time
from datetime import datetime
import secrets
from openai.types.chat import ChatCompletionMessageParam
from live_metrics import (
    initialize_log_file,
    generate_interaction_id,
    compute_retrieval_quality,
    detect_hallucination,
    log_interaction,
    update_user_satisfaction,
    get_dashboard_metrics
)
from rag_experiment_results import load_rag_experiment_results

# ========== ادوات القراءه==========
import pymupdf4llm
from unstructured.partition.auto import partition

# ========== إعدادات مساعدة (Docker Safe Paths) ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Route dynamic generated files to the DATA_DIR so they persist in Azure volumes
PENDING_FILE = os.path.join(Config.DATA_DIR, "pending_files.json")
BACKUP_DB_DIR = Config.CHROMA_DB_DIR + "_backup"
LOG_FILE = os.path.join(Config.DATA_DIR, "live_metrics_log.csv")
GOLDEN_PATH = os.path.join(Config.DATA_DIR, "golden_dataset.xlsx")
INTERNAL_SYSTEM_FILES = {
    "golden_dataset.xlsx",
    "live_metrics_log.csv",
    "pending_files.json",
    "generated_questions.xlsx",
    "mlflow.db",
    "pso_best_params.json",
    "pso_best_params.tmp.json",
    "user_queries.jsonl",
}


def is_internal_system_file(filename: str) -> bool:
    name = os.path.basename(str(filename)).lower()
    return name in INTERNAL_SYSTEM_FILES

EMBEDDING_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def load_pending_files():
    if os.path.exists(PENDING_FILE):
        with open(PENDING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_pending_files(files_list):
    os.makedirs(os.path.dirname(PENDING_FILE), exist_ok=True)
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump(files_list, f, ensure_ascii=False)

def add_pending_file(filename):
    pending = load_pending_files()
    if filename not in pending:
        pending.append(filename)
        save_pending_files(pending)

def clear_pending_files():
    if os.path.exists(PENDING_FILE):
        os.remove(PENDING_FILE)

def is_list_query(query: str) -> bool:
    list_keywords = [
        "اذكر", "عدّد", "قائمة", "كل", "جميع", "تفاصيل", "عدد", "كم", "ما هي", "ما هى",
        "list", "all", "details", "mention", "enumerate", "how many"
    ]
    return any(kw in query.lower() for kw in list_keywords)

# ---------- استخراج النص من الملفات ----------
def read_file_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == '.pdf':
            text = pymupdf4llm.to_markdown(file_path)
        elif ext in ['.docx', '.txt']:
            elements = partition(filename=file_path)
            text = "\n\n".join([el.text for el in elements if hasattr(el, 'text')])
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            text = df.to_string(index=False)
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            text = df.to_string(index=False)
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {str(e)}")
        text = ""
        
    # Safely ensure 'text' is a string before stripping, fallback to empty string if None
    if text is None:
        return ""
        
    return str(text).strip()

# ---------- توليد أسئلة متعددة اللغات ----------
def generate_questions_for_file(file_path: str, language: str, count: int = 5) -> List[Dict]:
    content = read_file_text(file_path)
    if not content:
        return []
    content = content[:5000]
    language_map = {
        "arabic": "العربية",
        "english": "الإنجليزية",
        "franco": "العربية بحروف إنجليزية (Franco)"
    }
    
    prompt = f"""أنت خبير في استخراج الأسئلة والأجوبة من مستندات.
من النص التالي، استخرج {count} أزواج من الأسئلة والأجوبة التي يمكن أن يسألها عميل حقيقي.
يجب أن تكون الإجابة موجودة فعلاً في النص.

- يجب أن تكون الأسئلة والأجوبة باللغة {language_map[language]}.
- **اكتب الإجابة في جملة واحدة فقط، مختصرة ومباشرة**، كما لو كنت موظف خدمة عملاء يجيب بسرعة دون أي حشو أو تفاصيل زائدة.
- لا تستخدم جداول أو تنسيقات إضافية.

أخرج النتيجة كمصفوفة JSON فقط. كل عنصر يجب أن يحتوي على:
- "query": السؤال.
- "expected_answer": الإجابة في جملة واحدة.
- "intent": تصنيف السؤال (branches, policy, catalog, general).
- "keywords": كلمات مفتاحية من السؤال.

النص:
{content}
"""
    try:
        # 1. Wrap the Config variables in str() to guarantee they are strings
        llm = AzureOpenAI(
            azure_endpoint=str(Config.AZURE_ENDPOINT),
            api_key=str(Config.AZURE_API_KEY),
            api_version=str(Config.AZURE_API_VERSION)
        )
        resp = llm.chat.completions.create(
            model=str(Config.AZURE_DEPLOYMENT_NAME), # Wrapped in str() here too just in case
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        output = resp.choices[0].message.content
        
        # 2. Add this fallback to ensure 'output' is never None before regex
        output = output or "" 
        
        match = re.search(r'\[.*\]', output, re.DOTALL)
        
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                for item in data:
                    item.setdefault("intent", "general")
                    item.setdefault("keywords", "")
                valid = [q for q in data if "query" in q and "expected_answer" in q]
                return valid
    except Exception as e:
        print(f"[EXTRACT] Error for {file_path} ({language}): {e}")
    return []

def extract_questions_from_files(file_paths: List[str], multilingual: bool = True) -> List[Dict]:
    if not file_paths:
        return []

    filtered_paths = [
        p for p in file_paths
        if not is_internal_system_file(os.path.basename(p))
    ]

    if not filtered_paths:
        return []

    all_questions = []

    for p in filtered_paths:
        if multilingual:
            all_questions.extend(generate_questions_for_file(p, language="arabic", count=15))
            all_questions.extend(generate_questions_for_file(p, language="english", count=12))
            all_questions.extend(generate_questions_for_file(p, language="franco", count=3))
        else:
            all_questions.extend(generate_questions_for_file(p, language="arabic", count=30))

    return all_questions

def create_initial_golden_dataset(data_dir: str) -> List[Dict]:
    if not os.path.exists(data_dir):
        return []

    files = [
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
        and not is_internal_system_file(f)
    ]

    if not files:
        return []

    file_paths = [os.path.join(data_dir, f) for f in files]
    return extract_questions_from_files(file_paths, multilingual=True)

# 1. Initialize FastAPI and define security scheme
app = FastAPI(title="CityFoam Customer Support API", version="2.1.0")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != Config.CITYFOAM_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Access Denied: Invalid API Key")
    return api_key

# 2. Setup Global Clients (Collection initialized in startup to prevent crashes)
chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_DIR)
collection = None 

llm_client = AzureOpenAI(
    azure_endpoint=str(Config.AZURE_ENDPOINT),
    api_key=str(Config.AZURE_API_KEY),
    api_version=str(Config.AZURE_API_VERSION)
)

# 3. Models and Helper functions
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = []
    language: str = "auto"

def normalize_arabic(text: str) -> str:
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'[\u064B-\u065F]', '', text)
    return text

# ========== حدث التشغيل التلقائي لأول مرة ==========
@app.on_event("startup")
def startup_initialize():
    global collection, chroma_client
    initialize_log_file(LOG_FILE)
    
    # Safely load or trigger creation of DB
    try:
        collection = chroma_client.get_collection(name=Config.COLLECTION_NAME)
    except ValueError:
        print("[STARTUP] Collection not found. It will be created during knowledge base build.")
        
    if not os.path.exists(GOLDEN_PATH):
        if os.path.exists(Config.DATA_DIR):
            files = [f for f in os.listdir(Config.DATA_DIR) if os.path.isfile(os.path.join(Config.DATA_DIR, f))]
            if files:
                print("[STARTUP] Golden dataset not found. Generating from existing files...")
                initial_questions = create_initial_golden_dataset(Config.DATA_DIR)
                if initial_questions:
                    df_initial = pd.DataFrame(initial_questions)
                    df_initial.to_excel(GOLDEN_PATH, index=False)
                    print(f"[STARTUP] Created golden dataset with {len(initial_questions)} questions.")

                    # تشغيل ingest لبناء قاعدة المعرفة
                    import importlib
                    import ingest
                    try:
                        importlib.reload(ingest)
                        ingest.main()
                        chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_DIR)
                        collection = chroma_client.get_collection(name=Config.COLLECTION_NAME)
                        print("[STARTUP] Knowledge base built successfully.")
                    except Exception as e:
                        print(f"[STARTUP] Error building knowledge base: {e}")
                else:
                    print("[STARTUP] No questions extracted from data files.")


@app.post("/api/admin/login")
async def admin_login(username: str = Form(...), password: str = Form(...)):
    # Docker Secure Authentication: Use Environment Variables
    admin_user = os.getenv("ADMIN_USERNAME", "admin")
    admin_pass = os.getenv("ADMIN_PASSWORD", "admin123")
    
    if username == admin_user and password == admin_pass:
        return {"status": "success", "token": Config.CITYFOAM_SECRET_KEY}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
# 4. RAG Pipeline Function
def get_rag_response(user_query: str, history: list) -> dict:
    clean_query = normalize_arabic(user_query)
    is_arabic = bool(re.search(r'[\u0600-\u06FF]', clean_query))
    target_language = "ARABIC" if is_arabic else "ENGLISH"

    router_prompt = f"""You are an advanced search router and query optimizer.
    Your job is to analyze the user's query and output a strict JSON object with two fields:
    
    1. "intent": Classify the query into EXACTLY one of these categories:
       - "branches" 
       - "policy" 
       - "catalog" 
       - "general" 
       
    2. "keywords": Clean, focused search keywords. 
       CRITICAL: If the user's query is in Arabic, you MUST translate the main entities and keywords to English and include BOTH the Arabic and English terms together in this field (e.g., if the user asks 'فرع بنها', output 'بنها Banha branch'). This is essential because the database documents are primarily in English.
    
    CRITICAL: Output ONLY valid raw JSON. No markdown wrappers."""

    try:
        router_output = llm_client.chat.completions.create(
            model=Config.AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": router_prompt},
                {"role": "user", "content": clean_query}
            ],
            temperature=0.0
        ).choices[0].message.content
        
        router_output = router_output or ""
        match = re.search(r'\{.*\}', router_output, re.DOTALL)
        
        if match:
            route_data = json.loads(match.group(0))
            intent = route_data.get("intent", "general").lower()
            
            # --- THE FIX IS HERE ---
            # Safely handle the keywords whether the AI returns a string or a list
            raw_keywords = route_data.get("keywords", clean_query)
            if isinstance(raw_keywords, list):
                optimized_query = " ".join(str(k) for k in raw_keywords)
            else:
                optimized_query = str(raw_keywords)
            # -----------------------
            
        else:
            raise ValueError("No JSON")
    except:
        intent = "general"
        optimized_query = clean_query
        
    query_vector = EMBEDDING_MODEL.encode([optimized_query]).tolist()
    
    if is_list_query(user_query):
        retrieval_limit = 25
    elif intent == "branches":
        retrieval_limit = 7
    elif intent == "catalog":
        retrieval_limit = 25
    else:
        retrieval_limit = 6

    # Safe collection query
    if collection is None:
        return {"answer": "النظام قيد التحديث، يرجى المحاولة لاحقاً.", "context_chunks": [], "intent": intent, "usage": None}

    results = collection.query(query_embeddings=query_vector, n_results=retrieval_limit)
    context_chunks = results['documents'][0] if results and results['documents'] else []

    if not context_chunks:
        results = collection.query(query_embeddings=query_vector, n_results=10)
        if results and results['documents']:
            context_chunks = results['documents'][0]

    context_string = "\n\n".join(context_chunks)

    system_prompt = f"""You are the official Customer Support AI Assistant for CityFoam. 
Your primary function is to provide accurate, helpful answers based STRICTLY on the official Knowledge Base provided below.

CRITICAL RULES:
1. ZERO HALLUCINATION: You must only use facts stated in the 'KNOWLEDGE BASE CONTEXT'. 
2. UNKNOWN ANSWERS: If the answer cannot be found, politely state that you do not have that information.
3. LANGUAGE LOCK: You MUST write your ENTIRE reply strictly in {target_language}.
4. CONTEXT AWARENESS: Use the previous conversation history to understand pronouns.
5. DO NOT output raw markdown tables. Instead, explain the information in natural language, summarizing when possible, and listing items in a readable format.

KNOWLEDGE BASE CONTEXT:
{context_string}
"""
    # Add the type hint here: list[ChatCompletionMessageParam]
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt}
    ]
    
    for msg in history:
        # The type checker now knows these dictionaries are valid message parameters
        messages.append({"role": msg.role, "content": msg.content})
        
    messages.append({"role": "user", "content": user_query})

    response = llm_client.chat.completions.create(
        model=Config.AZURE_DEPLOYMENT_NAME,
        messages=messages,
        temperature=0.2
    )
    answer = response.choices[0].message.content
    usage = response.usage

    return {
        "answer": answer,
        "context_chunks": context_chunks,
        "intent": intent,
        "usage": usage
    }

# 5. API Endpoints
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(os.path.join(STATIC_DIR, "favicon.ico"))

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    start = time.time()
    error = 0
    answer = ""
    chunks = []
    intent = "general"
    usage = None
    interaction_id = generate_interaction_id()
    try:
        result = get_rag_response(request.query, request.history)
        answer = result["answer"]
        chunks = result["context_chunks"]
        intent = result["intent"]
        usage = result["usage"]
    except Exception as e:
        error = 1
        answer = f"Error: {str(e)}"
    finally:
        latency_ms = (time.time() - start) * 1000
        retrieval_qual = compute_retrieval_quality(request.query, chunks, EMBEDDING_MODEL) if not error else 0.0
        hall_flag = 1 if detect_hallucination(answer, chunks) else 0
        response_len = len(answer)
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        log_interaction(
            log_path=LOG_FILE,
            interaction_id=interaction_id,
            timestamp=datetime.now(),
            query=request.query,
            response=answer,
            latency_ms=latency_ms,
            error=error,
            intent=intent,
            chunks_retrieved=len(chunks),
            retrieval_quality=retrieval_qual,
            hallucination_flag=hall_flag,
            response_length=response_len,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )

    if error:
        raise HTTPException(status_code=500, detail=answer)
    return {
        "status": "success",
        "query": request.query,
        "response": answer,
        "interaction_id": interaction_id
    }

@app.post("/api/rate")
def rate_interaction(interaction_id: str, rating: int):
    if rating not in (0, 1):
        raise HTTPException(status_code=400, detail="Rating must be 0 (negative) or 1 (positive)")
    success = update_user_satisfaction(LOG_FILE, interaction_id, rating)
    if not success:
        raise HTTPException(status_code=404, detail="Interaction ID not found")
    return {"status": "success", "message": "Rating recorded"}

@app.get("/api/dashboard/metrics")
def dashboard_metrics():
    metrics = get_dashboard_metrics(LOG_FILE)
    return JSONResponse(content=metrics)

# ========== غلاف التقييم ==========
def pipeline_evaluation_wrapper(case_dict):
    query = case_dict["query"]
    intent = case_dict.get("intent", "general")

    clean_query = normalize_arabic(query)
    is_arabic = bool(re.search(r'[\u0600-\u06FF]', clean_query))
    target_language = "ARABIC" if is_arabic else "ENGLISH"

    query_vector = EMBEDDING_MODEL.encode([clean_query]).tolist()

    if is_list_query(query):
        retrieval_limit = 25
    elif intent == "branches":
        retrieval_limit = 7
    elif intent == "catalog":
        retrieval_limit = 25
    else:
        retrieval_limit = 5

    if collection is None:
        return "", [], 0.0

    results = collection.query(query_embeddings=query_vector, n_results=retrieval_limit)
    context_chunks = results['documents'][0] if results and results['documents'] else []

    if not context_chunks:
        results = collection.query(query_embeddings=query_vector, n_results=5)
        if results and results['documents']:
            context_chunks = results['documents'][0]

    context_string = "\n\n".join(context_chunks)

    system_prompt = f"""You are the official Customer Support AI Assistant for CityFoam. 
Answer the user's question accurately based on the Knowledge Base.
- Answer ONLY in {target_language}.
- Be concise and directly address the question. Do not include tables.
- Only use facts from the context.

KNOWLEDGE BASE CONTEXT:
{context_string}"""

    try:
        response = llm_client.chat.completions.create(
            model=Config.AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0
        )
        generated_answer = response.choices[0].message.content
    except:
        generated_answer = ""

    vec_expected = EMBEDDING_MODEL.encode(case_dict["expected_answer"])
    vec_generated = EMBEDDING_MODEL.encode(generated_answer)
    sim = evaluation.calculate_cosine_similarity(vec_expected, vec_generated)
    return generated_answer, context_chunks, sim

# ========== رفع الملفات ==========#added the verify api key
@app.post("/api/upload-knowledge")
async def upload_knowledge_file(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    allowed_extensions = {".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt"}
    
    # 1. Safely guarantee the filename is a string, with a fallback
    safe_filename = str(file.filename) if file.filename else "unknown_file"
    
    file_ext = os.path.splitext(safe_filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="امتداد الملف غير مدعوم!")

    # 2. Safely guarantee the directory path is a string
    safe_data_dir = str(Config.DATA_DIR)
    
    os.makedirs(safe_data_dir, exist_ok=True)
    file_path = os.path.join(safe_data_dir, safe_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        add_pending_file(safe_filename) # Use the safe string here too
        return {"status": "success", "message": f"تم رفع '{safe_filename}' بنجاح."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== ml flow and dashboard ===============
@app.get("/api/rag-experiments")
def get_rag_experiments():
    return load_rag_experiment_results()

@app.get("/rag-experiments-dashboard")
def serve_rag_experiments_dashboard():
    return FileResponse("static/rag_experiments_dashboard.html")
# ========== إعادة التدريب اللاحقة ==========
@app.post("/api/trigger-retraining")#added security lock
async def trigger_retraining_pipeline(api_key: str = Depends(verify_api_key)):
    try:
        pending_files = load_pending_files()

        if not os.path.exists(GOLDEN_PATH):
            return JSONResponse(status_code=400, content={
                "status": "error",
                "message": "الملف الذهبي غير موجود. أعد تشغيل الخادم أو تأكد من وجود ملفات في مجلد data."
            })

        if not pending_files:
            return JSONResponse(status_code=200, content={
                "status": "idle",
                "message": "لا توجد ملفات جديدة لإعادة التدريب."
            })

        new_file_paths = [os.path.join(Config.DATA_DIR, f) for f in pending_files]
        new_questions = extract_questions_from_files(new_file_paths, multilingual=True)
        if not new_questions:
            print("[RETRAIN] No new questions extracted.")

        previous_metrics = evaluation.run_system_evaluation(pipeline_evaluation_wrapper)

        if os.path.exists(BACKUP_DB_DIR):
            shutil.rmtree(BACKUP_DB_DIR)
        shutil.copytree(Config.CHROMA_DB_DIR, BACKUP_DB_DIR)

        import importlib
        import ingest
        try:
            importlib.reload(ingest)
            ingest.main()
        except Exception as e:
            shutil.rmtree(Config.CHROMA_DB_DIR)
            shutil.copytree(BACKUP_DB_DIR, Config.CHROMA_DB_DIR)
            for f in pending_files:
                try:
                    os.remove(os.path.join(Config.DATA_DIR, f))
                except:
                    pass
            clear_pending_files()
            raise HTTPException(status_code=500, detail=f"فشل أثناء تحديث قاعدة البيانات: {str(e)}")

        global collection, chroma_client
        chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_DIR)
        collection = chroma_client.get_collection(name=Config.COLLECTION_NAME)

        post_metrics_original = evaluation.run_system_evaluation(pipeline_evaluation_wrapper)

        metrics_comparison = []
        for key in previous_metrics:
            prev = previous_metrics[key]
            curr = post_metrics_original[key]
            change = round(curr - prev, 2)
            metrics_comparison.append({
                "metric": key.upper().replace("_", " "),
                "previous": prev,
                "current": curr,
                "change": change
            })

        severely_degraded = any(
            post_metrics_original[m] < previous_metrics[m] - 15.0
            for m in ["hit_rate", "semantic_similarity"]
        )

        new_metrics = None
        if new_questions:
            new_metrics = evaluation.run_system_evaluation(pipeline_evaluation_wrapper, custom_dataset=new_questions)

        if severely_degraded:
            shutil.rmtree(Config.CHROMA_DB_DIR)
            shutil.copytree(BACKUP_DB_DIR, Config.CHROMA_DB_DIR)
            for f in pending_files:
                try:
                    os.remove(os.path.join(Config.DATA_DIR, f))
                except:
                    pass
            clear_pending_files()
            chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_DIR)
            collection = chroma_client.get_collection(name=Config.COLLECTION_NAME)

            return JSONResponse(status_code=200, content={
                "status": "failed",
                "message": "فشل التدريب: تدهور كبير في المؤشرات.",
                "metrics": metrics_comparison,
                "new_questions_metrics": new_metrics
            })
        else:
            shutil.rmtree(BACKUP_DB_DIR)

            if new_questions:
                try:
                    df_existing = pd.read_excel(GOLDEN_PATH)
                    df_new = pd.DataFrame(new_questions)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_excel(GOLDEN_PATH, index=False)
                    print(f"[RETRAIN] Added {len(new_questions)} new multilingual questions.")
                except Exception as e:
                    print(f"[ERROR] Failed to update golden dataset: {e}")

            clear_pending_files()

            return JSONResponse(status_code=200, content={
                "status": "success",
                "message": "تم اعتماد التدريب وإضافة الأسئلة الجديدة للملف الذهبي." if new_questions else "تم اعتماد التدريب.",
                "metrics": metrics_comparison,
                "new_questions_count": len(new_questions) if new_questions else 0,
                "new_questions_metrics": new_metrics
            })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ داخلي: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Changed host from 127.0.0.1 to 0.0.0.0 for Docker routing on Azure
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)