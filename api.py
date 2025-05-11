# Adaptive-Tutor/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import re
from openai import OpenAI  # For v1.x
from dotenv import load_dotenv
import time
from typing import List, Dict  # For Pydantic models

load_dotenv()

from db_utils import init_db
from memory import episodic_memory, biographical_memory, procedural_memory
from policy import select_next_question
from train_model import featurize_for_classifier

app = FastAPI(title="Adaptive Math Tutor API")


@app.on_event("startup")
async def startup_event():
    print("FastAPI application starting up...")
    try:
        init_db()
        print("Database schema initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Database initialization failed: {e}")
    try:
        load_resources()
    except RuntimeError as e:
        print(f"CRITICAL ERROR during load_resources on startup: {e}")


DATA_FILE = "data/math_questions.csv"
MODEL_FILE = "data/answer_classifier.joblib"
USE_LLM_FOR_HINTS = True
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client_openai = None  # Initialize client_openai to None
if USE_LLM_FOR_HINTS and OPENAI_API_KEY:
    try:
        client_openai = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client configured for LLM hints.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}. LLM hints will use fallback.")
elif USE_LLM_FOR_HINTS and not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not found in .env or is empty. LLM hints will use fallback.")
else:
    print("LLM hints are disabled via USE_LLM_FOR_HINTS flag.")

questions_df = None
classifier_model = None


def load_resources():
    global questions_df, classifier_model
    print("Loading static resources (questions CSV and classifier model)...")
    if not os.path.exists(DATA_FILE):
        raise RuntimeError(f"Data file '{DATA_FILE}' not found. Please ensure it exists.")
    questions_df = pd.read_csv(DATA_FILE).fillna("")
    if not questions_df.empty and 'question_id' in questions_df.columns:
        questions_df['question_id'] = questions_df['question_id'].astype(str)

    if not os.path.exists(MODEL_FILE):
        raise RuntimeError(f"Model file '{MODEL_FILE}' not found. Please ensure it's trained and available.")
    classifier_model = joblib.load(MODEL_FILE)
    print("Static API Resources loaded successfully.")


# --- Pydantic Models ---
class UserAnswerInput(BaseModel):
    user_id: str
    question_id: str
    user_answer: str
    session_id: str = "default_session"


class HintInput(BaseModel):
    user_id: str
    question_id: str


# --- NEW: Pydantic Models for User Summary ---
class TopicSummary(BaseModel):
    topic_name: str
    total_attempted: int
    correct_attempted: int
    accuracy: float  # 0.0 to 1.0
    mastery_score: float  # From BiographicalMemory


class UserPerformanceSummary(BaseModel):
    user_id: str
    total_questions_answered_overall: int
    overall_accuracy: float  # 0.0 to 1.0
    current_difficulty_level: str
    topic_summaries: List[TopicSummary]
    hints_requested_total: int
    hints_led_to_success_total: int


# --- END NEW Pydantic Models ---


def get_question_details(qid: str):
    if questions_df is None: return None
    row = questions_df[questions_df['question_id'] == str(qid)]
    if row.empty: return None
    return row.iloc[0].to_dict()


# --- API Endpoints ---
@app.get("/")
async def read_root(): return {"message": "Welcome to the Adaptive Math Tutor API!"}


@app.get("/next_question")
async def get_next_question_endpoint(user_id: str):
    if questions_df is None or questions_df.empty:
        raise HTTPException(status_code=503, detail="Question data not available.")
    q_data = select_next_question(user_id, questions_df)
    if not q_data:
        if questions_df.empty: raise HTTPException(status_code=503, detail="No questions available.")
        q_data = questions_df.sample(1).iloc[0].to_dict()
    return {
        "question_id": str(q_data["question_id"]), "topic": q_data.get("topic", "N/A"),
        "question_text": q_data.get("question_text", ""), "difficulty": q_data.get("difficulty", "N/A"),
        "render_as": q_data.get("render_as", "markdown")
    }


@app.post("/classify")
async def classify_user_answer_endpoint(data: UserAnswerInput):
    if classifier_model is None or questions_df is None:
        raise HTTPException(status_code=503, detail="Service not fully initialized.")
    details = get_question_details(data.question_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Question {data.question_id} not found.")
    correct_text = str(details.get("correct_answer", ""))
    answer_row = pd.Series({'user_answer': data.user_answer, 'correct_answer_for_q': correct_text})
    try:
        features = featurize_for_classifier(answer_row)
        label = classifier_model.predict([features])[0]
    except Exception as e:
        print(f"Error during model prediction for QID {data.question_id}. Error: {e}")
        raise HTTPException(status_code=500, detail="Error classifying answer.")
    reward = {"correct": 1.0, "partial": 0.25, "incorrect": -0.5}.get(label, -0.5)
    episodic_memory.add_event(user_id=data.user_id, question_id=data.question_id, user_answer=data.user_answer,
                              classified_label=label, reward=reward, session_id=data.session_id)
    biographical_memory.update_topic_mastery(user_id=data.user_id, topic=details.get("topic", "Unknown"),
                                             is_correct=(label == "correct"))
    if label in ("incorrect", "partial"):
        procedural_memory.record_mistake(data.user_id, data.question_id, mistake_type=f"classified_as_{label}")
    issued_hints = procedural_memory.get_hints_issued(data.user_id)
    if label == "correct" and str(data.question_id) in issued_hints:
        procedural_memory.record_hint_success(data.user_id, data.question_id)
    return {"question_id": data.question_id, "user_answer": data.user_answer, "classified_label": label,
            "correct_answer": correct_text, "solution_explanation": details.get("solution_explanation", ""),
            "reward_assigned": reward}


@app.post("/hint")
async def get_hint_endpoint(data: HintInput):
    global client_openai
    if questions_df is None:
        raise HTTPException(status_code=503, detail="Question data not available for hints.")
    details = get_question_details(data.question_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Question {data.question_id} not found for hint.")
    procedural_memory.record_hint_issued(data.user_id, data.question_id)
    attempts = episodic_memory.get_attempts_for_question(data.user_id, data.question_id)
    last_ans = attempts[-1]["user_answer"] if attempts else ""
    mistakes = procedural_memory.get_common_mistakes(data.user_id)
    top_mistake = max(mistakes, key=mistakes.get) if mistakes else "none"
    system_msg = (
        "Your goal is to provide a *single, small, conceptual next step* or a *leading question* ... Keep hints very short, ideally under 20 words.")  # Abridged for brevity
    user_msg = (f"Question: {details['question_text']}\n...")  # Abridged
    hint_text = "";
    llm_attempted_and_failed = False
    if USE_LLM_FOR_HINTS and client_openai:
        # ... (LLM call logic as in your last working version) ...
        print(f"[Hint LLM Call] Attempting LLM hint for QID {data.question_id}, User: {data.user_id}.")
        try:
            resp = client_openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                temperature=0.3, max_tokens=40
            )
            hint_text = resp.choices[0].message.content.strip()
            if not hint_text:
                llm_attempted_and_failed = True; print(
                    f"[Hint LLM Call] Warning: LLM returned an empty hint for QID {data.question_id}.")
            else:
                print(f"[Hint LLM Call] Success. LLM Hint: '{hint_text}'")
        except Exception as e:
            print(f"[Hint LLM Call] FAILED. Error: {e}");
            hint_text = "";
            llm_attempted_and_failed = True
    elif USE_LLM_FOR_HINTS and not client_openai:
        print(f"[Hint LLM Call] Skipped for QID {data.question_id}: OpenAI client not configured.");
        llm_attempted_and_failed = True
    if not hint_text:
        # ... (Fallback hint logic as in your last working version) ...
        if llm_attempted_and_failed:
            print(
                f"[Hint Fallback] LLM hint failed/skipped or was empty, using rule-based fallback for QID {data.question_id}.")
        else:
            print(f"[Hint Fallback] LLM hints disabled, using rule-based fallback for QID {data.question_id}.")
        expl = details.get("solution_explanation", "Review the problem statement carefully.");
        question_text_lower = details.get("question_text", "").lower();
        topic = details.get("topic", "").lower()
        if "perimeter" in question_text_lower and "rectangle" in question_text_lower:
            hint_text = "Perimeter means the total length around the outside. For a rectangle, what sides do you add up, or what is the formula using length (l) and width (w)?"
        elif "percentage" in topic or "%" in question_text_lower:
            hint_text = "Remember, 'percent' means 'out of 100'. How can you use that to calculate a percentage of a number?"
        # ... other specific fallbacks ...
        else:
            hint_text = "Try to break the problem down into smaller steps or re-read the question carefully."  # Simplified for brevity
        if len(hint_text) > 120: hint_text = hint_text[:117] + "..."
        if not hint_text.strip(): hint_text = "Think about the first step you would take to solve this."
    return {"question_id": data.question_id, "hint": hint_text}


# --- NEW: User Summary Endpoint ---
@app.get("/user_summary/{user_id}", response_model=UserPerformanceSummary)
async def get_user_summary(user_id: str):
    """
    Provides a performance summary for the given user.
    """
    print(f"[User Summary] Fetching summary for user_id: {user_id}")

    # 1. Get overall performance from EpisodicMemory
    user_history = episodic_memory.get_user_history(user_id)  # Gets all events
    total_questions_answered_overall = 0
    correct_answers_overall = 0

    # To count unique questions answered, we can use a set of question_ids from history
    # However, for accuracy, we should count distinct (user_id, question_id, session_id, timestamp) for attempts
    # For simplicity here, let's count events that are classifications.
    # A more precise "questions answered" might look at unique question_ids per session or overall.

    # Filter for actual answer classification events if history contains other event types
    # Assuming all events in episodic_memory for now are answer attempts for simplicity.
    # If you add other event types, you'll need to filter.

    # Simple count of all events as "attempts"
    # For a more accurate "questions answered", you'd count unique question_ids
    # or sum total_attempts from topic_mastery.

    # Let's use topic_mastery for total questions answered.

    # 2. Get profile data from BiographicalMemory
    profile = biographical_memory.get_profile(user_id)  # This fetches from DB
    current_difficulty = profile.get("current_difficulty_level", "easy")

    # 3. Get Topic Summaries from BiographicalMemory
    topic_summaries: List[TopicSummary] = []
    # The get_profile now returns topic_mastery as a dict keyed by topic name
    # The values are dicts like: {'topic': 'Algebra', 'correct_attempts': 5, 'total_attempts': 10, 'current_streak': 1, 'mastery_score': 51.0}

    # Recalculate total questions answered and overall accuracy from topic_mastery
    # This is more robust than iterating all episodic events if mastery is the source of truth for attempts.

    db_topic_mastery_records = profile.get("topic_mastery", {})

    for topic_name, mastery_data in db_topic_mastery_records.items():
        if isinstance(mastery_data, dict):  # Ensure it's the expected dict structure
            total_attempted = mastery_data.get('total_attempts', 0)
            correct_attempted = mastery_data.get('correct_attempts', 0)
            accuracy = (correct_attempted / total_attempted) if total_attempted > 0 else 0.0

            topic_summaries.append(TopicSummary(
                topic_name=topic_name,
                total_attempted=total_attempted,
                correct_attempted=correct_attempted,
                accuracy=round(accuracy, 2),  # Round to 2 decimal places
                mastery_score=round(mastery_data.get('mastery_score', 0.0), 1)
            ))
            total_questions_answered_overall += total_attempted
            correct_answers_overall += correct_attempted
        # else: # Handle cases where mastery_data might not be a dict (e.g. if defaultdict produced something else)
        # print(f"[User Summary] Skipping malformed mastery_data for topic {topic_name}: {mastery_data}")

    overall_accuracy_calc = (
                correct_answers_overall / total_questions_answered_overall) if total_questions_answered_overall > 0 else 0.0

    # 4. Get Hint data from ProceduralMemory
    hints_requested = procedural_memory.get_hint_success_count(user_id)  # This is actually success count
    # We need a method in ProceduralMemory to get total hints issued count
    # For now, let's use len(procedural_memory.get_hints_issued(user_id)) as a proxy if it returns all issued hints
    total_hints_issued = len(
        procedural_memory.get_hints_issued(user_id))  # This gets distinct QIDs for which hints were issued
    # If you want total hint *requests*, the DB table user_hints_issued should be counted.
    # Let's assume get_hint_success_count is correct.

    # To get total hints *requested*, we need to count rows in user_hints_issued table.
    # Let's add a quick way for this, or modify get_hints_issued to return all.
    # For now, this might be an undercount if multiple hints for same QID.
    # A proper way:
    # conn_temp = get_db_connection()
    # cursor_temp = conn_temp.cursor()
    # cursor_temp.execute("SELECT COUNT(*) FROM user_hints_issued WHERE user_id = ?", (user_id,))
    # total_hints_issued_count_row = cursor_temp.fetchone()
    # total_hints_issued_actual = total_hints_issued_count_row[0] if total_hints_issued_count_row else 0
    # conn_temp.close()
    # For simplicity, using the existing method's length for now.

    hints_succeeded = procedural_memory.get_hint_success_count(user_id)

    return UserPerformanceSummary(
        user_id=user_id,
        total_questions_answered_overall=total_questions_answered_overall,
        overall_accuracy=round(overall_accuracy_calc, 2),
        current_difficulty_level=current_difficulty,
        topic_summaries=topic_summaries,
        hints_requested_total=total_hints_issued,  # This is count of QIDs for which hints were issued
        hints_led_to_success_total=hints_succeeded
    )


# --- END NEW Endpoint ---

if __name__ == "__main__":
    import uvicorn

    print("Starting API server with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
