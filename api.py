# Adaptive-Tutor/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import re
import openai
from dotenv import load_dotenv

# Load .env (for OPENAI_API_KEY)
load_dotenv()

# Local imports
from memory import episodic_memory, biographical_memory, procedural_memory
from policy import select_next_question
from train_model import featurize_for_classifier, normalize_text, extract_numbers, compare_answers

# FastAPI app
app = FastAPI(title="Adaptive Math Tutor API")

# Config & Globals
DATA_FILE = "data/math_questions.csv"
MODEL_FILE = "data/answer_classifier.joblib"
USE_LLM_FOR_HINTS = True  # flip to False to disable LLM hints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if USE_LLM_FOR_HINTS and not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env to use LLM hints")
openai.api_key = OPENAI_API_KEY

questions_df = None
classifier_model = None

def load_resources():
    global questions_df, classifier_model
    if not os.path.exists(DATA_FILE):
        raise RuntimeError(f"Data file '{DATA_FILE}' not found.")
    questions_df = pd.read_csv(DATA_FILE).fillna("")
    if not os.path.exists(MODEL_FILE):
        raise RuntimeError(f"Model file '{MODEL_FILE}' not found.")
    classifier_model = joblib.load(MODEL_FILE)
    print("API Resources loaded successfully.")

load_resources()

# Pydantic models
class UserAnswerInput(BaseModel):
    user_id: str
    question_id: str
    user_answer: str
    session_id: str = "default_session"

class HintInput(BaseModel):
    user_id: str
    question_id: str

# Helper to fetch question
def get_question_details(qid: str):
    row = questions_df[questions_df['question_id'] == qid]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

# --- Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Adaptive Math Tutor API!"}

@app.get("/next_question")
async def get_next_question(user_id: str):
    q = select_next_question(user_id, questions_df)
    if not q:
        # fallback
        q = questions_df.sample(1).iloc[0].to_dict()
    return {
        "question_id": q["question_id"],
        "topic": q["topic"],
        "question_text": q["question_text"],
        "difficulty": q["difficulty"]
    }

@app.post("/classify")
async def classify_user_answer(data: UserAnswerInput):
    # 1. Fetch correct answer
    details = get_question_details(data.question_id)
    if not details:
        raise HTTPException(404, f"Question {data.question_id} not found.")
    correct_text = str(details["correct_answer"])

    # 2. Featurize and predict
    row = pd.Series({
        'user_answer': data.user_answer,
        'correct_answer_for_q': correct_text
    })
    features = featurize_for_classifier(row)
    label = classifier_model.predict([features])[0]

    # 3. Assign reward
    reward = {"correct":1.0, "partial":0.25, "incorrect":-0.5}[label]

    # 4. Log to episodic memory
    episodic_memory.add_event(
        user_id=data.user_id,
        question_id=data.question_id,
        user_answer=data.user_answer,
        classified_label=label,
        reward=reward,
        session_id=data.session_id
    )

    # 5. Update topic mastery
    biographical_memory.update_topic_mastery(
        user_id=data.user_id,
        topic=details["topic"],
        is_correct=(label == "correct")
    )

    # 6. Record procedural mistakes
    if label in ("incorrect","partial"):
        procedural_memory.record_mistake(
            data.user_id,
            data.question_id,
            mistake_type=f"classified_as_{label}"
        )

    # 7. Track hint success if applicable
    issued = procedural_memory.get_hints_issued(data.user_id)
    if label == "correct" and data.question_id in issued:
        procedural_memory.record_hint_success(data.user_id, data.question_id)
        # Remove to avoid double‐counting
        issued.remove(data.question_id)

    # 8. Return feedback
    return {
        "question_id": data.question_id,
        "user_answer": data.user_answer,
        "classified_label": label,
        "correct_answer": correct_text,
        "solution_explanation": details.get("solution_explanation",""),
        "reward_assigned": reward
    }

@app.post("/hint")
async def get_hint(data: HintInput):
    details = get_question_details(data.question_id)
    if not details:
        raise HTTPException(404, f"Question {data.question_id} not found.")

    # 1. Record that we issued a hint
    procedural_memory.record_hint_issued(data.user_id, data.question_id)

    # 2. Gather context for the hint
    #    a) last student answer
    attempts = episodic_memory.get_attempts_for_question(data.user_id, data.question_id)
    last_ans = attempts[-1]["user_answer"] if attempts else ""
    #    b) top mistake
    mistakes = procedural_memory.get_common_mistakes(data.user_id)
    top_mistake = max(mistakes, key=mistakes.get) if mistakes else "none"

    # 3. Build LLM prompt
    system_msg = (
        "You are a patient math tutor. "
        "Given a question, a student's last answer, and their common mistake type, "
        "provide a single‐step hint pointing them in the right direction without giving away the full solution."
    )
    user_msg = (
        f"Question: {details['question_text']}\n"
        f"Student's last answer: {last_ans}\n"
        f"Detected mistake: {top_mistake}\n"
        "Please give a concise, next-step hint."
    )

    # 4. Call LLM (OpenAI)
    hint_text = ""
    try:
        if USE_LLM_FOR_HINTS:
            resp = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role":"system","content":system_msg},
                    {"role":"user","content":user_msg}
                ],
                temperature=0.5,
                max_tokens=50
            )
            hint_text = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Hint] LLM call failed: {e}")

    # 5. Fallback if LLM unavailable or failed
    if not hint_text:
        expl = details.get("solution_explanation","Review the problem statement carefully.")
        m = re.match(r"^([^.]+\.)", expl)
        hint_text = m.group(1) if m else expl
        if len(hint_text) > 150:
            hint_text = hint_text[:150] + "..."

    return {"question_id": data.question_id, "hint": hint_text}


if __name__ == "__main__":
    import uvicorn
    print("Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
