# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
import os
import pandas as pd
import joblib
import re
import sys
import time
import numpy as np  # Make sure numpy is imported for np.pi and other functions

# Add the project root to sys.path to allow importing api and its dependencies
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api import app, load_resources, DATA_FILE, MODEL_FILE, USE_LLM_FOR_HINTS
from memory import episodic_memory, biographical_memory, procedural_memory
from policy import DIFFICULTY_ORDER, STREAK_TO_ADVANCE_DIFFICULTY

# ---- ADDED/CORRECTED IMPORTS for debugging within tests ----
from train_model import normalize_text as tm_normalize_text
from train_model import extract_numbers as tm_extract_numbers
from train_model import compare_answers as tm_compare_answers

# ---- END OF ADDED/CORRECTED IMPORTS ----

# --- Test Configuration & Constants ---
TEST_USER_ID = "test_user_comprehensive"
TEST_SESSION_ID = "test_session_comprehensive"


# TEST_DATA_DIR = "data" # Not directly used in this script's logic path


# --- Helper Functions for Tests ---
def get_question_detail_from_df(question_id, df):
    """Helper to get question details directly from the DataFrame for assertions."""
    row = df[df['question_id'] == question_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


# --- Pytest Fixtures ---
@pytest.fixture(scope="module")
def client():
    """
    Test client fixture that ensures resources are loaded once per module.
    Handles potential errors during resource loading.
    """
    try:
        # Ensure the working directory is the project root when tests run
        # This helps `api.py` find `data/math_questions.csv` correctly
        # This chdir might be problematic if tests are run from different subdirs or with certain test runners.
        # Consider making file paths in `api.py` absolute or relative to `api.py`'s location.
        # For now, assuming tests run from project root or this chdir works.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if os.getcwd() != project_root:
            os.chdir(project_root)
        print(f"Current working directory for tests: {os.getcwd()}")

        # Check for essential files before attempting to load resources
        if not os.path.exists(DATA_FILE):
            pytest.exit(f"Critical: Test data file '{DATA_FILE}' not found. Run generate_synthetic_data.py.")
        if not os.path.exists(MODEL_FILE):
            pytest.exit(f"Critical: Model file '{MODEL_FILE}' not found. Run train_model.py.")

        load_resources()  # Load actual resources for the API
        print("Test setup: API resources loaded successfully for the module.")
    except RuntimeError as e:
        pytest.exit(f"Failed to initialize test module: {e}")

    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_memory_state():
    """Auto-applied fixture to reset in-memory stores before each test for isolation."""
    episodic_memory.log = []
    procedural_memory.user_mistakes.clear()
    procedural_memory.user_hints_issued.clear()
    procedural_memory.user_hint_success_count.clear()
    biographical_memory.user_profiles.clear()


@pytest.fixture
def questions_df_loaded():
    """Provides the globally loaded questions_df from the API module."""
    from api import questions_df  # Import here to get the instance loaded by the app
    if questions_df is None:
        # This might happen if `load_resources` failed silently or was mocked out
        # Or if the test runner has a strange module caching issue.
        # For now, let's assume load_resources in the client fixture handles it.
        try:
            global_questions_df_check = pd.read_csv(DATA_FILE)  # Try to load it directly for the fixture
            if global_questions_df_check.empty:
                pytest.skip("questions_df seems empty even after direct load, skipping test needing it.")
            return global_questions_df_check
        except Exception as e:
            pytest.skip(f"questions_df not loaded in API module and direct load failed: {e}, skipping test.")

    return questions_df


# --- Mocking LLM Calls ---
class MockChoice:
    def __init__(self, text):
        self.message = MockMessage(text)


class MockMessage:
    def __init__(self, text):
        self.content = text


class MockOpenAIResponse:
    def __init__(self, text="Mocked LLM Hint."):
        self.choices = [MockChoice(text)]


def mock_openai_chat_completion_success(*args, **kwargs):
    return MockOpenAIResponse()


def mock_openai_chat_completion_failure(*args, **kwargs):
    raise Exception("Mocked LLM API Call Failed")


# --- Test Cases ---

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Adaptive Math Tutor API!"}


def test_get_next_question_new_user(client):
    response = client.get(f"/next_question?user_id={TEST_USER_ID}")
    assert response.status_code == 200
    data = response.json()
    assert "question_id" in data
    assert "topic" in data
    profile = biographical_memory.get_profile(TEST_USER_ID)
    assert profile["current_difficulty_level"] == "easy"


def test_get_next_question_existing_user(client, questions_df_loaded):
    response1 = client.get(f"/next_question?user_id={TEST_USER_ID}")
    assert response1.status_code == 200
    qid1 = response1.json()["question_id"]
    q_detail = get_question_detail_from_df(qid1, questions_df_loaded)
    if q_detail is None: pytest.skip(f"QID {qid1} not found in loaded df for existing user test.")

    client.post("/classify", json={
        "user_id": TEST_USER_ID, "question_id": qid1, "user_answer": q_detail["correct_answer"],
        "session_id": TEST_SESSION_ID
    })
    response2 = client.get(f"/next_question?user_id={TEST_USER_ID}")
    assert response2.status_code == 200
    qid2 = response2.json()["question_id"]
    if len(questions_df_loaded) > 1:
        assert qid1 != qid2


@pytest.mark.parametrize("answer_type, get_user_answer_func, expected_label, expected_reward, mistake_key_suffix", [
    ("correct", lambda ca, cq: ca, "correct", 1.0, None),
    ("incorrect_generic", lambda ca, cq: "this is completely wrong", "incorrect", -0.5, "incorrect"),
    ("incorrect_numerical",
     lambda ca, cq: str(float(re.findall(r'-?\d+\.?\d*', ca)[0]) + 100) if re.findall(r'-?\d+\.?\d*', ca) else "99999",
     "incorrect", -0.5, "incorrect"),
])
def test_classify_various_answers(client, questions_df_loaded, answer_type, get_user_answer_func, expected_label,
                                  expected_reward, mistake_key_suffix):
    q_resp = client.get(f"/next_question?user_id={TEST_USER_ID}")
    assert q_resp.status_code == 200, "Failed to get question for classification test"
    q_data = q_resp.json()
    question_id = q_data["question_id"]
    q_detail = get_question_detail_from_df(question_id, questions_df_loaded)
    if q_detail is None: pytest.skip(f"QID {question_id} not found in loaded df for classify_various test.")
    correct_answer_text = q_detail["correct_answer"]
    user_answer = get_user_answer_func(correct_answer_text, q_detail["question_text"])
    payload = {
        "user_id": TEST_USER_ID, "question_id": question_id,
        "user_answer": user_answer, "session_id": TEST_SESSION_ID
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["classified_label"] == expected_label
    assert data["reward_assigned"] == expected_reward
    history = episodic_memory.get_user_history(TEST_USER_ID)
    assert len(history) > 0
    assert history[0]["classified_label"] == expected_label
    if mistake_key_suffix:
        mistake_type = f"classified_as_{mistake_key_suffix}"
        assert procedural_memory.user_mistakes[TEST_USER_ID][mistake_type] > 0


def test_classify_partial_answer(client, questions_df_loaded):
    quadratic_q = questions_df_loaded[
        questions_df_loaded['question_text'].str.contains(r"x\^2", regex=True) &
        questions_df_loaded['correct_answer'].str.contains("or|and|,", case=False, regex=True)
        ]
    if quadratic_q.empty:
        pytest.skip("No suitable quadratic question found for partial answer test.")
    q_detail = quadratic_q.iloc[0].to_dict()
    question_id = q_detail["question_id"]
    correct_answer_text = q_detail["correct_answer"]
    first_root_match = re.search(r'(-?\d+\.?\d*)', correct_answer_text)
    if not first_root_match:
        pytest.skip("Could not parse a root from quadratic question's correct answer.")
    partial_user_answer = f"x = {first_root_match.group(1)}"
    payload = {
        "user_id": TEST_USER_ID, "question_id": question_id,
        "user_answer": partial_user_answer, "session_id": TEST_SESSION_ID
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["classified_label"] in ["partial", "incorrect"]
    if data["classified_label"] == "partial":
        assert data["reward_assigned"] == 0.25


def test_classify_fraction_answer(client, questions_df_loaded):
    fraction_q = questions_df_loaded[questions_df_loaded['correct_answer'].str.contains("/", na=False)]
    if fraction_q.empty:
        pytest.skip("No suitable question with a fractional answer found.")
    q_detail = fraction_q.iloc[0].to_dict()
    question_id = q_detail["question_id"]
    correct_answer_text = q_detail["correct_answer"]
    user_answers_to_test = [correct_answer_text]
    if "/" in correct_answer_text:
        try:
            answer_part = correct_answer_text.split("=")[-1].strip()
            if '/' in answer_part:
                num_str, den_str = answer_part.split('/')
                if den_str.strip() and float(den_str.strip()) != 0:
                    numerical_equivalent = str(float(num_str.strip()) / float(den_str.strip()))
                    user_answers_to_test.append(numerical_equivalent)
        except Exception:
            pass

    for user_answer in user_answers_to_test:
        payload = {
            "user_id": TEST_USER_ID, "question_id": question_id, "user_answer": user_answer,
            "session_id": TEST_SESSION_ID
        }
        response = client.post("/classify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data[
                   "classified_label"] == "correct", f"Fraction test failed for user_answer='{user_answer}', expected 'correct', got '{data['classified_label']}'"
        assert data["reward_assigned"] == 1.0


def test_classify_answer_with_pi(client, questions_df_loaded):
    """Test answers involving 'pi' (symbolic vs. numeric)."""
    pi_q = questions_df_loaded[questions_df_loaded['correct_answer'].str.lower().str.contains("pi", na=False)]
    if pi_q.empty:
        pytest.skip("No suitable question with 'pi' in the answer found in test data.")

    q_detail = pi_q.iloc[0].to_dict()
    question_id = q_detail["question_id"]
    correct_answer_text_orig = q_detail["correct_answer"]

    print(f"\n\n[DEBUG PI TEST START] QID: {question_id}")
    print(f"[DEBUG PI TEST] Original Correct Answer Text: '{correct_answer_text_orig}'")

    user_answer_symbolic = correct_answer_text_orig

    normalized_correct_for_approx = tm_normalize_text(correct_answer_text_orig)
    extracted_nums_from_correct = tm_extract_numbers(normalized_correct_for_approx)

    user_answer_numeric_approx = user_answer_symbolic
    unit_suffix = ""

    unit_match_correct = re.search(r'\s*([a-zA-Z²³]+)\s*$', correct_answer_text_orig)
    if not unit_match_correct:
        unit_match_correct = re.search(r'pi\s*([a-zA-Z²³]+)\s*$', correct_answer_text_orig.lower())

    if unit_match_correct:
        unit_suffix = " " + unit_match_correct.group(1).strip()

    if extracted_nums_from_correct:
        main_correct_num = extracted_nums_from_correct[0]
        user_answer_numeric_approx = f"{main_correct_num:.2f}{unit_suffix}".strip()
        if correct_answer_text_orig.lower().replace(unit_suffix.lower().strip(), "").strip() == "pi":
            user_answer_numeric_approx = f"{np.pi:.2f}{unit_suffix}".strip()

    print(f"[DEBUG PI TEST] User Answer Symbolic to test: '{user_answer_symbolic}'")
    print(f"[DEBUG PI TEST] User Answer Numeric Approx to test: '{user_answer_numeric_approx}'")

    answers_to_test_in_loop = [user_answer_symbolic]
    if user_answer_numeric_approx != user_answer_symbolic:
        answers_to_test_in_loop.append(user_answer_numeric_approx)

    for i, user_answer_to_test in enumerate(answers_to_test_in_loop):
        print(f"\n[DEBUG PI TEST] --- Loop Iteration {i + 1} ---")
        print(f"[DEBUG PI TEST] Testing User Answer: '{user_answer_to_test}'")

        ua_norm = tm_normalize_text(user_answer_to_test)
        ca_norm = tm_normalize_text(correct_answer_text_orig)
        print(f"[DEBUG PI TEST] User Normalized: '{ua_norm}'")
        print(f"[DEBUG PI TEST] Correct Original Normalized: '{ca_norm}'")

        ua_nums = tm_extract_numbers(ua_norm)
        ca_nums = tm_extract_numbers(ca_norm)
        print(f"[DEBUG PI TEST] User Numbers Extracted: {ua_nums}")
        print(f"[DEBUG PI TEST] Correct Original Numbers Extracted: {ca_nums}")

        features_dict_debug = tm_compare_answers(ua_norm, ca_norm)
        print(f"[DEBUG PI TEST] Generated Features (Debug): {features_dict_debug}")

        payload = {"user_id": TEST_USER_ID, "question_id": question_id, "user_answer": user_answer_to_test,
                   "session_id": TEST_SESSION_ID}
        response = client.post("/classify", json=payload)
        assert response.status_code == 200
        data = response.json()
        print(f"[DEBUG PI TEST] API Response Label for '{user_answer_to_test}': {data['classified_label']}")

        assert data["classified_label"] == "correct", \
            f"Failed for user_answer='{user_answer_to_test}'. Expected 'correct', got '{data['classified_label']}'"
        assert data["reward_assigned"] == 1.0
    print(f"[DEBUG PI TEST END] QID: {question_id}")


def test_classify_empty_or_malformed_answer(client, questions_df_loaded):
    q_resp = client.get(f"/next_question?user_id={TEST_USER_ID}")
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    question_id = q_data["question_id"]
    malformed_answers = ["", "   ", "!@#$%^", "very long string" * 10]  # Reduced length
    for user_answer in malformed_answers:
        payload = {
            "user_id": TEST_USER_ID, "question_id": question_id, "user_answer": user_answer,
            "session_id": TEST_SESSION_ID
        }
        response = client.post("/classify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["classified_label"] == "incorrect"
        assert data["reward_assigned"] == -0.5


def test_classify_invalid_question_id(client):
    payload = {
        "user_id": TEST_USER_ID, "question_id": "INVALID_QID_NONEXISTENT",
        "user_answer": "any answer", "session_id": TEST_SESSION_ID
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 404


# --- /hint Endpoint Tests ---
def test_get_hint_llm_success(client, questions_df_loaded, monkeypatch):
    monkeypatch.setattr("api.USE_LLM_FOR_HINTS", True)
    if not os.getenv("OPENAI_API_KEY"):  # Simpler check for dummy key
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_test_llm_success")
        monkeypatch.setattr("api.OPENAI_API_KEY", "dummy_key_for_test_llm_success")  # Ensure API module sees it

    monkeypatch.setattr("openai.ChatCompletion.create", mock_openai_chat_completion_success)
    q_resp = client.get(f"/next_question?user_id={TEST_USER_ID}")
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    question_id = q_data["question_id"]
    client.post("/classify", json={  # Simulate prior attempt
        "user_id": TEST_USER_ID, "question_id": question_id, "user_answer": "wrong", "session_id": TEST_SESSION_ID
    })
    payload = {"user_id": TEST_USER_ID, "question_id": question_id}
    response = client.post("/hint", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["hint"] == "Mocked LLM Hint."
    assert question_id in procedural_memory.get_hints_issued(TEST_USER_ID)


def test_get_hint_llm_failure_fallback(client, questions_df_loaded, monkeypatch):
    monkeypatch.setattr("api.USE_LLM_FOR_HINTS", True)
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_test_llm_fail")
        monkeypatch.setattr("api.OPENAI_API_KEY", "dummy_key_for_test_llm_fail")

    monkeypatch.setattr("openai.ChatCompletion.create", mock_openai_chat_completion_failure)
    q_resp = client.get(f"/next_question?user_id={TEST_USER_ID}")
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    question_id = q_data["question_id"]
    q_detail = get_question_detail_from_df(question_id, questions_df_loaded)
    if q_detail is None: pytest.skip(f"QID {question_id} not found for LLM fallback test.")

    payload = {"user_id": TEST_USER_ID, "question_id": question_id}
    response = client.post("/hint", json=payload)
    assert response.status_code == 200
    data = response.json()
    expl = q_detail.get("solution_explanation", "Review the problem statement carefully.")
    m = re.match(r"^([^.]+\.)", expl)
    expected_fallback_hint = m.group(1) if m else expl
    if len(expected_fallback_hint) > 150: expected_fallback_hint = expected_fallback_hint[:150] + "..."
    assert data["hint"] == expected_fallback_hint


def test_get_hint_disabled_llm(client, questions_df_loaded, monkeypatch):
    monkeypatch.setattr("api.USE_LLM_FOR_HINTS", False)
    q_resp = client.get(f"/next_question?user_id={TEST_USER_ID}")
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    question_id = q_data["question_id"]
    q_detail = get_question_detail_from_df(question_id, questions_df_loaded)
    if q_detail is None: pytest.skip(f"QID {question_id} not found for disabled LLM test.")

    payload = {"user_id": TEST_USER_ID, "question_id": question_id}
    response = client.post("/hint", json=payload)
    assert response.status_code == 200
    data = response.json()
    expl = q_detail.get("solution_explanation", "Review the problem statement carefully.")
    m = re.match(r"^([^.]+\.)", expl)
    expected_fallback_hint = m.group(1) if m else expl
    if len(expected_fallback_hint) > 150: expected_fallback_hint = expected_fallback_hint[:150] + "..."
    assert data["hint"] == expected_fallback_hint


def test_get_hint_invalid_question_id(client):
    payload = {"user_id": TEST_USER_ID, "question_id": "INVALID_QID_XYZ"}
    response = client.post("/hint", json=payload)
    assert response.status_code == 404


# --- Memory and Policy Interaction Tests ---
def test_hint_success_recording(client, questions_df_loaded, monkeypatch):
    monkeypatch.setattr("api.USE_LLM_FOR_HINTS", False)
    q_resp = client.get(f"/next_question?user_id={TEST_USER_ID}")
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    question_id = q_data["question_id"]
    q_detail = get_question_detail_from_df(question_id, questions_df_loaded)
    if q_detail is None: pytest.skip(f"QID {question_id} not found for hint success test.")
    correct_answer = q_detail["correct_answer"]

    client.post("/hint", json={"user_id": TEST_USER_ID, "question_id": question_id})
    initial_hint_success_count = procedural_memory.get_hint_success_count(TEST_USER_ID)
    client.post("/classify", json={
        "user_id": TEST_USER_ID, "question_id": question_id,
        "user_answer": correct_answer, "session_id": TEST_SESSION_ID
    })
    assert procedural_memory.get_hint_success_count(TEST_USER_ID) == initial_hint_success_count + 1


def test_difficulty_progression(client, questions_df_loaded):
    user_id_streak = f"{TEST_USER_ID}_streak"
    # biographical_memory.user_profiles[user_id_streak]['current_difficulty_level'] = 'easy' # Explicitly set for test

    current_difficulty_in_profile = biographical_memory.get_profile(user_id_streak)["current_difficulty_level"]
    assert current_difficulty_in_profile == "easy"

    for i in range(STREAK_TO_ADVANCE_DIFFICULTY):  # Loop exactly STREAK_TO_ADVANCE_DIFFICULTY times for correct answers
        q_resp = client.get(f"/next_question?user_id={user_id_streak}")
        assert q_resp.status_code == 200
        q_data = q_resp.json()
        question_id = q_data["question_id"]
        q_detail = get_question_detail_from_df(question_id, questions_df_loaded)
        if q_detail is None: pytest.skip(f"QID {question_id} not found for difficulty progression test.")

        print(
            f"\nAttempt {i + 1} for user {user_id_streak}, QID: {question_id}, Current Profile Diff (before classify): {biographical_memory.get_profile(user_id_streak)['current_difficulty_level']}")
        print(
            f"Attempt {i + 1} for user {user_id_streak}, QID: {question_id}, Question Actual Diff: {q_data['difficulty']}")

        client.post("/classify", json={
            "user_id": user_id_streak, "question_id": question_id,
            "user_answer": q_detail["correct_answer"], "session_id": TEST_SESSION_ID
        })
        print(
            f"After classify {i + 1}, Profile Diff: {biographical_memory.get_profile(user_id_streak)['current_difficulty_level']}")

    # After STREAK_TO_ADVANCE_DIFFICULTY correct answers, the *next* call to /next_question should promote difficulty
    print(
        f"\nCalling /next_question AFTER streak for {user_id_streak}. Profile diff before call: {biographical_memory.get_profile(user_id_streak)['current_difficulty_level']}")
    final_q_resp = client.get(f"/next_question?user_id={user_id_streak}")  # This call's policy run should promote
    assert final_q_resp.status_code == 200

    final_difficulty_in_profile = biographical_memory.get_profile(user_id_streak)["current_difficulty_level"]
    print(f"After final /next_question, Profile Diff: {final_difficulty_in_profile}")

    current_diff_idx_from_easy = DIFFICULTY_ORDER.index('easy')
    if current_diff_idx_from_easy < len(DIFFICULTY_ORDER) - 1:
        expected_new_difficulty = DIFFICULTY_ORDER[current_diff_idx_from_easy + 1]
        assert final_difficulty_in_profile == expected_new_difficulty
    else:  # If 'easy' was already 'hard' or last in list somehow
        assert final_difficulty_in_profile == DIFFICULTY_ORDER[-1]


def test_topic_focus_on_struggle(client, questions_df_loaded):
    user_id_struggle = f"{TEST_USER_ID}_struggle"
    topics = questions_df_loaded['topic'].unique()
    if len(topics) < 2: pytest.skip("Not enough unique topics for topic focus test.")

    topic_to_struggle_on = topics[0]
    other_topic = topics[1]

    for _ in range(3):  # Correct on other_topic
        q_other_list = questions_df_loaded[questions_df_loaded['topic'] == other_topic]
        if q_other_list.empty: pytest.skip(f"No questions for other_topic: {other_topic}")
        q_other = q_other_list.sample(1).iloc[0]
        client.post("/classify", json={
            "user_id": user_id_struggle, "question_id": q_other["question_id"],
            "user_answer": q_other["correct_answer"], "session_id": TEST_SESSION_ID
        })

    for i in range(3):  # Incorrect on topic_to_struggle_on
        q_struggle_list = questions_df_loaded[questions_df_loaded['topic'] == topic_to_struggle_on]
        if q_struggle_list.empty: pytest.skip(f"No questions for struggle_topic: {topic_to_struggle_on}")
        q_struggle = q_struggle_list.sample(1).iloc[0]
        client.post("/classify", json={
            "user_id": user_id_struggle, "question_id": q_struggle["question_id"],
            "user_answer": f"wrong_ans_{i}", "session_id": TEST_SESSION_ID
        })

    focused_topic_count = 0
    for _ in range(3):  # Check next 3 questions
        next_q_resp = client.get(f"/next_question?user_id={user_id_struggle}")
        assert next_q_resp.status_code == 200
        next_q_topic = next_q_resp.json()["topic"]
        if next_q_topic == topic_to_struggle_on:
            focused_topic_count += 1
    assert focused_topic_count > 0