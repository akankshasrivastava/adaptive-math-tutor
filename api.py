# Adaptive-Tutor/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import re
import openai
from dotenv import load_dotenv
import time  # Good to have for any direct timestamp needs

# Load .env (for OPENAI_API_KEY)
load_dotenv()

# --- NEW: Import for database initialization ---
from db_utils import init_db  # <--- ADDED THIS

# Local imports
from memory import episodic_memory, biographical_memory, procedural_memory
from policy import select_next_question
# featurize_for_classifier is the main function needed from train_model at runtime
from train_model import featurize_for_classifier

# normalize_text, extract_numbers, compare_answers are dependencies of featurize_for_classifier

# FastAPI app
app = FastAPI(title="Adaptive Math Tutor API")


# --- NEW: Startup event to initialize the database ---
@app.on_event("startup")
async def startup_event():
    """
    Event handler that runs when the FastAPI application starts.
    Initializes the database schema.
    """
    print("FastAPI application starting up...")
    try:
        init_db()  # This will create tables if they don't exist
        print("Database schema initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Database initialization failed: {e}")
        # Depending on the severity, you might want to prevent the app from fully starting
        # or handle this more gracefully. For now, it prints an error.

    # load_resources() is called at the module level below.
    # This ensures static files are loaded after the app object is created but before serving requests.
    # DB init should happen early, so this ordering is fine.


# --- END NEW ---

# Config & Globals
DATA_FILE = "data/math_questions.csv"
MODEL_FILE = "data/answer_classifier.joblib"
USE_LLM_FOR_HINTS = True  # flip to False to disable LLM hints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup OpenAI API key
if USE_LLM_FOR_HINTS:
    if not OPENAI_API_KEY:
        # Changed from RuntimeError to a print warning, as the /hint endpoint
        # will also check and use fallback if key is missing.
        print("WARNING: OPENAI_API_KEY not set in .env. LLM hints will use fallback if this key is required.")
    # Setting openai.api_key to None if OPENAI_API_KEY is None is fine;
    # the library calls will fail if a key is needed and not provided.
    openai.api_key = OPENAI_API_KEY
else:
    print("LLM hints are disabled via USE_LLM_FOR_HINTS flag.")

questions_df = None
classifier_model = None


def load_resources():
    """Loads static resources like the questions CSV and the trained classifier model."""
    global questions_df, classifier_model
    print("Loading static resources (questions CSV and classifier model)...")
    if not os.path.exists(DATA_FILE):
        raise RuntimeError(f"Data file '{DATA_FILE}' not found. Please ensure it exists.")
    questions_df = pd.read_csv(DATA_FILE).fillna("")

    if not os.path.exists(MODEL_FILE):
        raise RuntimeError(f"Model file '{MODEL_FILE}' not found. Please ensure it's trained and available.")
    classifier_model = joblib.load(MODEL_FILE)
    print("Static API Resources loaded successfully.")


# Load static resources once when the module is imported.
# This happens after app object creation but typically before the "startup" event fully completes for Uvicorn.
# The DB should be initialized by the time these resources might hypothetically need it (they don't currently).
try:
    load_resources()
except RuntimeError as e:
    print(f"CRITICAL ERROR during load_resources: {e}")
    # Exit if critical resources can't be loaded, as the app won't function.
    # This is a simple way to handle it; a more complex app might try to run in a degraded state.
    raise


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
    """Fetches question details from the global questions_df."""
    if questions_df is None:
        # This should ideally be caught by load_resources failing earlier
        print("Error in get_question_details: questions_df is not loaded.")
        return None  # Or raise an internal server error

    # Ensure qid is compared as string, as CSV IDs can be mixed type upon read
    row = questions_df[questions_df['question_id'].astype(str) == str(qid)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


# --- Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Adaptive Math Tutor API!"}


@app.get("/next_question")
async def get_next_question(user_id: str):  # Kept original endpoint name
    """Selects and returns the next question for the user based on policy."""
    if questions_df is None or questions_df.empty:
        # Service might be still starting or data file is problematic
        raise HTTPException(status_code=503, detail="Question data not available. Please try again shortly.")

    q_data = select_next_question(user_id, questions_df)
    if not q_data:
        print(f"Warning: Policy returned no question for user {user_id}. Sampling randomly as fallback.")
        if questions_df.empty:  # Should not happen if initial check passed
            raise HTTPException(status_code=503, detail="No questions available in the dataset for fallback.")
        q_data = questions_df.sample(1).iloc[0].to_dict()

    return {
        "question_id": str(q_data["question_id"]),  # Ensure ID is consistently string
        "topic": q_data["topic"],
        "question_text": q_data["question_text"],
        "difficulty": q_data["difficulty"],
        "render_as": q_data.get("render_as", "markdown")  # Include render_as if available
    }


@app.post("/classify")
async def classify_user_answer(data: UserAnswerInput):  # Kept original endpoint name
    """Classifies the user's answer, records interaction, and updates mastery."""
    if classifier_model is None or questions_df is None:
        raise HTTPException(status_code=503,
                            detail="Service not fully initialized (model or data missing). Please try again shortly.")

    details = get_question_details(data.question_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Question {data.question_id} not found.")
    correct_text = str(details["correct_answer"])

    # Prepare a Series for the featurizer
    answer_row_for_featurizer = pd.Series({
        'user_answer': data.user_answer,
        'correct_answer_for_q': correct_text
    })

    try:
        features = featurize_for_classifier(answer_row_for_featurizer)
        label = classifier_model.predict([features])[0]
    except Exception as e:
        print(f"Error during model prediction for QID {data.question_id}, User: {data.user_id}. Error: {e}")
        # Optionally, log the 'features' variable here for debugging
        raise HTTPException(status_code=500, detail="Error classifying answer.")

    reward_map = {"correct": 1.0, "partial": 0.25, "incorrect": -0.5}
    reward = reward_map.get(label, -0.5)  # Default reward for unexpected labels

    # These memory calls will be updated to use the database in the next phase
    current_timestamp = time.time()  # Generate timestamp once for all related memory events

    episodic_memory.add_event(
        user_id=data.user_id,
        question_id=data.question_id,
        user_answer=data.user_answer,
        classified_label=label,
        reward=reward,
        session_id=data.session_id,
        # timestamp=current_timestamp # The add_event method will handle its own timestamping
    )

    biographical_memory.update_topic_mastery(
        user_id=data.user_id,
        topic=details["topic"],
        is_correct=(label == "correct")
        # timestamp=current_timestamp # update_topic_mastery might also want a timestamp
    )

    if label in ("incorrect", "partial"):
        procedural_memory.record_mistake(
            user_id=data.user_id,
            question_id=data.question_id,
            mistake_type=f"classified_as_{label}"
            # timestamp=current_timestamp
        )

    # The get_hints_issued will query DB. The list `issued` is local to this call.
    issued = procedural_memory.get_hints_issued(data.user_id)
    if label == "correct" and data.question_id in issued:
        procedural_memory.record_hint_success(data.user_id, data.question_id)
        # The original `issued.remove(data.question_id)` was problematic for in-memory lists
        # if `get_hints_issued` returned a direct reference.
        # With DB, `record_hint_success` should handle idempotency or `get_hints_issued` should
        # perhaps return hints that haven't yet led to success for this question.
        # For now, the logic is kept as is, assuming `record_hint_success` logs an event.

    return {
        "question_id": data.question_id,
        "user_answer": data.user_answer,
        "classified_label": label,
        "correct_answer": correct_text,
        "solution_explanation": details.get("solution_explanation", "No explanation available."),
        "reward_assigned": reward
    }


@app.post("/hint")
async def get_hint(data: HintInput):  # Kept original endpoint name
    """Provides a hint for a given question."""
    if questions_df is None:
        raise HTTPException(status_code=503, detail="Question data not available for hints. Please try again shortly.")

    details = get_question_details(data.question_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Question {data.question_id} not found for hint.")

    procedural_memory.record_hint_issued(data.user_id, data.question_id)

    attempts = episodic_memory.get_attempts_for_question(data.user_id, data.question_id)
    last_ans = attempts[-1]["user_answer"] if attempts else ""

    mistakes = procedural_memory.get_common_mistakes(data.user_id)
    top_mistake = max(mistakes, key=mistakes.get) if mistakes else "none"

    # Using the refined system message from previous discussions
    system_msg = (
        "You are an expert Socratic math tutor for middle school students. "
        "Your hints should guide the student to discover the next step themselves. "
        "Do NOT give direct answers or formulas unless the formula itself IS the hint. "
        "Focus on a single conceptual or procedural step. "
        "If the student's last answer is close, acknowledge the correct part before guiding. "
        "If a common mistake is provided, tailor the hint to address that specific "
        "misconception without explicitly stating 'you made X mistake'. "
        "Keep hints under 30 words."  # Slightly increased for flexibility
    )
    user_msg = (
        f"Question: {details['question_text']}\n"
        f"Student's last answer (if any): {last_ans}\n"
        f"Student's common mistake type (if relevant): {top_mistake}\n"
        "Please provide a concise, Socratic, next-step hint based on these details."
    )

    hint_text = ""
    if USE_LLM_FOR_HINTS and openai.api_key:  # Explicitly check if API key is available
        try:
            resp = openai.ChatCompletion.create(  # Assuming older openai lib version syntax
                model="gpt-4o",  # Or your preferred model like "gpt-3.5-turbo"
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.5,
                max_tokens=60  # Increased max_tokens for hint
            )
            hint_text = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Hint LLM Call] Error: {e}")
            # Fall through to rule-based hint
    elif USE_LLM_FOR_HINTS and not openai.api_key:
        print("[Hint LLM Call] Skipped: OpenAI API key not configured.")

    if not hint_text:  # Fallback if LLM disabled, failed, or no API key
        print(f"[Hint Fallback] Using rule-based hint for QID {data.question_id} for user {data.user_id}.")
        expl = details.get("solution_explanation",
                           "Review the problem statement carefully and identify the first step.")

        match = re.match(r"^([^.]+\.)", expl)  # Get first sentence
        if match:
            hint_text = match.group(1)
        else:  # If no period, or explanation is short
            # Take up to a certain number of words or characters if no sentence structure
            words = expl.split()
            if len(words) > 15:  # Arbitrary word limit for a hint
                hint_text = " ".join(words[:15]) + "..."
            else:
                hint_text = expl

        # Ensure fallback hint is not overly long
        if len(hint_text) > 150:  # Max length for fallback
            hint_text = hint_text[:147] + "..."
        if not hint_text.strip():  # Final safety net
            hint_text = "Try to break the problem down into smaller steps or re-read the question carefully."

    return {"question_id": data.question_id, "hint": hint_text}


if __name__ == "__main__":
    import uvicorn

    # init_db() is called via the app startup event.
    # load_resources() is called at module scope.
    print("Starting API server with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)