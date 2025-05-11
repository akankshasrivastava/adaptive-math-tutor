# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
import os
import pandas as pd
import joblib
import re
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api import app, DATA_FILE, MODEL_FILE  # Removed load_resources, it's called by api.py
from memory import episodic_memory, biographical_memory, procedural_memory
from policy import DIFFICULTY_ORDER, STREAK_TO_ADVANCE_DIFFICULTY
from train_model import normalize_text as tm_normalize_text
from train_model import extract_numbers as tm_extract_numbers
from train_model import compare_answers as tm_compare_answers
from db_utils import get_db_connection

TEST_USER_ID = "test_user_comprehensive"
TEST_SESSION_ID = "test_session_comprehensive"


def get_question_detail_from_df(question_id, df):
    row = df[df['question_id'].astype(str) == str(question_id)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


@pytest.fixture(scope="module")
def client():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if os.getcwd() != project_root:
        os.chdir(project_root)
    if not os.path.exists(DATA_FILE):
        pytest.exit(f"Critical: Test data file '{DATA_FILE}' not found.")
    if not os.path.exists(MODEL_FILE):
        pytest.exit(f"Critical: Model file '{MODEL_FILE}' not found.")
    # api.py calls load_resources() at module level when app is created.
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_memory_state():
    """Clears ALL relevant DB tables for test isolation."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM episodic_events;")
        cursor.execute("DELETE FROM user_mistakes;")
        cursor.execute("DELETE FROM user_reflections;")
        cursor.execute("DELETE FROM user_hints_issued;")
        cursor.execute("DELETE FROM user_hint_successes;")
        cursor.execute("DELETE FROM topic_mastery;")
        cursor.execute("DELETE FROM user_profiles;")
        conn.commit()
    except Exception as e:
        print(f"Error clearing database for tests: {e}")
    finally:
        if conn:
            conn.close()


@pytest.fixture
def questions_df_loaded(client):
    from api import questions_df
    if questions_df is None or questions_df.empty:
        pytest.skip("questions_df not loaded in API module, skipping test.")
    return questions_df


class MockChoice:
    def __init__(self, text): self.message = MockMessage(text)


class MockMessage:
    def __init__(self, text): self.content = text


class MockOpenAIResponse:
    def __init__(self, text="Mocked LLM Hint."): self.choices = [MockChoice(text)]


def mock_openai_chat_completion_success(*args, **kwargs): return MockOpenAIResponse()


def mock_openai_chat_completion_failure(*args, **kwargs): raise Exception("Mocked LLM API Call Failed")


# --- Test Cases ---

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Adaptive Math Tutor API!"}


def test_get_next_question_new_user(client):
    user_id = f"{TEST_USER_ID}_new_q_user"
    response = client.get(f"/next_question?user_id={user_id}")
    assert response.status_code == 200
    data = response.json()
    assert "question_id" in data
    profile = biographical_memory.get_profile(user_id)
    assert profile["current_difficulty_level"] == "easy"


def test_get_next_question_existing_user(client, questions_df_loaded):
    user_id = f"{TEST_USER_ID}_existing_q_user"
    response1 = client.get(f"/next_question?user_id={user_id}")
    qid1 = response1.json()["question_id"]
    q_detail = get_question_detail_from_df(qid1, questions_df_loaded)
    if not q_detail: pytest.skip(f"QID {qid1} not found in test_get_next_question_existing_user")

    client.post("/classify", json={
        "user_id": user_id, "question_id": qid1, "user_answer": q_detail["correct_answer"],
        "session_id": TEST_SESSION_ID})

    response2 = client.get(f"/next_question?user_id={user_id}")
    qid2 = response2.json()["question_id"]
    if len(questions_df_loaded) > 1: assert qid1 != qid2


@pytest.mark.parametrize("answer_type, get_user_answer_func, expected_label, expected_reward, mistake_key_suffix", [
    ("correct", lambda ca, cq: ca, "correct", 1.0, None),
    ("incorrect_generic", lambda ca, cq: "this is completely wrong", "incorrect", -0.5, "incorrect"),
    ("incorrect_numerical",
     lambda ca, cq: str(float(re.findall(r'-?\d+\.?\d*', ca)[0]) + 100) if re.findall(r'-?\d+\.?\d*', ca) else "99999",
     "incorrect", -0.5, "incorrect"),
])
def test_classify_various_answers(client, questions_df_loaded, answer_type, get_user_answer_func, expected_label,
                                  expected_reward, mistake_key_suffix):
    user_id = f"{TEST_USER_ID}_classify_various"
    q_resp = client.get(f"/next_question?user_id={user_id}")
    q_data = q_resp.json()
    question_id = q_data["question_id"]
    q_detail = get_question_detail_from_df(question_id, questions_df_loaded)
    if not q_detail: pytest.skip(f"QID {question_id} not found in test_classify_various_answers")
    correct_answer_text = q_detail["correct_answer"]
    user_answer = get_user_answer_func(correct_answer_text, q_detail["question_text"])

    payload = {"user_id": user_id, "question_id": question_id, "user_answer": user_answer,
               "session_id": TEST_SESSION_ID}
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["classified_label"] == expected_label
    assert data["reward_assigned"] == expected_reward

    if mistake_key_suffix:
        common_mistakes = procedural_memory.get_common_mistakes(user_id)
        assert f"classified_as_{mistake_key_suffix}" in common_mistakes
        assert common_mistakes[f"classified_as_{mistake_key_suffix}"] > 0


def test_classify_partial_answer(client, questions_df_loaded):
    user_id = f"{TEST_USER_ID}_partial"
    quadratic_q = questions_df_loaded[
        questions_df_loaded['question_text'].str.contains(r"x\^2", regex=True) &
        questions_df_loaded['correct_answer'].str.contains("or|and|,", case=False, regex=True)]
    if quadratic_q.empty: pytest.skip("No suitable quadratic question for partial answer test.")
    q_detail = quadratic_q.iloc[0].to_dict()
    question_id = q_detail["question_id"]
    correct_answer_text = q_detail["correct_answer"]
    first_root_match = re.search(r'(-?\d+\.?\d*)', correct_answer_text)
    if not first_root_match: pytest.skip("Could not parse root from quadratic answer.")
    partial_user_answer = f"x = {first_root_match.group(1)}"

    payload = {"user_id": user_id, "question_id": question_id, "user_answer": partial_user_answer,
               "session_id": TEST_SESSION_ID}
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["classified_label"] in ["partial", "incorrect"]
    if data["classified_label"] == "partial":
        common_mistakes = procedural_memory.get_common_mistakes(user_id)
        assert "classified_as_partial" in common_mistakes
        assert common_mistakes["classified_as_partial"] > 0


def test_classify_fraction_answer(client, questions_df_loaded):
    user_id = f"{TEST_USER_ID}_fraction"
    fraction_q = questions_df_loaded[questions_df_loaded['correct_answer'].str.contains("/", na=False)]
    if fraction_q.empty: pytest.skip("No fractional answer questions found.")
    q_detail = fraction_q.iloc[0].to_dict()
    question_id = q_detail["question_id"]
    correct_answer_text = q_detail["correct_answer"]
    user_answers_to_test = [correct_answer_text]
    if "/" in correct_answer_text:
        try:
            answer_part = correct_answer_text.split("=")[-1].strip()
            if '/' in answer_part:
                num_str, den_str = map(str.strip, answer_part.split('/'))
                if float(den_str) != 0:
                    user_answers_to_test.append(str(float(num_str) / float(den_str)))
        except Exception:
            pass
    for user_answer in user_answers_to_test:
        payload = {"user_id": user_id, "question_id": question_id, "user_answer": user_answer,
                   "session_id": TEST_SESSION_ID}
        response = client.post("/classify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data[
                   "classified_label"] == "correct", f"Fraction test: User Ans: '{user_answer}', Correct: '{correct_answer_text}', API Label: {data['classified_label']}"


def test_classify_answer_with_pi(client, questions_df_loaded):
    user_id = f"{TEST_USER_ID}_pi"
    pi_q = questions_df_loaded[questions_df_loaded['correct_answer'].str.lower().str.contains("pi", na=False)]
    if pi_q.empty: pytest.skip("No 'pi' questions found.")
    q_detail = pi_q.iloc[0].to_dict()
    question_id = q_detail["question_id"]
    correct_answer_text_orig = q_detail["correct_answer"]
    user_answer_symbolic = correct_answer_text_orig
    normalized_correct = tm_normalize_text(correct_answer_text_orig)
    extracted_nums_correct = tm_extract_numbers(normalized_correct)
    user_answer_numeric_approx = user_answer_symbolic
    unit_suffix = ""
    unit_match = re.search(r'\s*([a-zA-Z²³]+)\s*$', correct_answer_text_orig) or \
                 re.search(r'pi\s*([a-zA-Z²³]+)\s*$', correct_answer_text_orig.lower())
    if unit_match: unit_suffix = " " + unit_match.group(1).strip()
    if extracted_nums_correct:
        main_num = extracted_nums_correct[0]
        user_answer_numeric_approx = f"{main_num:.2f}{unit_suffix}".strip()
        if correct_answer_text_orig.lower().replace(unit_suffix.lower().strip(), "").strip() == "pi":
            user_answer_numeric_approx = f"{np.pi:.2f}{unit_suffix}".strip()
    answers_to_test = [user_answer_symbolic]
    if user_answer_numeric_approx != user_answer_symbolic: answers_to_test.append(user_answer_numeric_approx)
    for user_answer in answers_to_test:
        payload = {"user_id": user_id, "question_id": question_id, "user_answer": user_answer,
                   "session_id": TEST_SESSION_ID}
        response = client.post("/classify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data[
                   "classified_label"] == "correct", f"Pi test failed for user_answer='{user_answer}', correct_ans='{correct_answer_text_orig}'"


def test_classify_empty_or_malformed_answer(client, questions_df_loaded):
    user_id = f"{TEST_USER_ID}_malformed"
    q_resp = client.get(f"/next_question?user_id={user_id}")
    q_data = q_resp.json()
    question_id = q_data["question_id"]
    for ans in ["", "   ", "!@#$%^", "long string" * 5]:
        payload = {"user_id": user_id, "question_id": question_id, "user_answer": ans, "session_id": TEST_SESSION_ID}
        response = client.post("/classify", json=payload)
        assert response.status_code == 200
        data = response.json();
        assert data["classified_label"] == "incorrect"


def test_classify_invalid_question_id(client):
    user_id = f"{TEST_USER_ID}_invalid_qid"
    payload = {"user_id": user_id, "question_id": "INVALID_QID_XYZ123", "user_answer": "any",
               "session_id": TEST_SESSION_ID}
    response = client.post("/classify", json=payload)
    assert response.status_code == 404


def test_get_hint_llm_success(client, questions_df_loaded, monkeypatch):
    user_id = f"{TEST_USER_ID}_hint_llm_ok"
    from api import USE_LLM_FOR_HINTS as api_use_llm_flag
    monkeypatch.setattr("api.USE_LLM_FOR_HINTS", True)
    if not os.getenv("OPENAI_API_KEY"): monkeypatch.setenv("OPENAI_API_KEY", "dummy_for_test_ok")
    monkeypatch.setattr("api.client_openai",
                        OpenAI(api_key=os.getenv("OPENAI_API_KEY")))  # Ensure client is re-init with key
    monkeypatch.setattr("openai.OpenAI.chat.completions.create",
                        mock_openai_chat_completion_success)  # Mock the method on the instance

    q_resp = client.get(f"/next_question?user_id={user_id}")
    q_data = q_resp.json();
    question_id = q_data["question_id"]
    client.post("/classify", json={"user_id": user_id, "question_id": question_id, "user_answer": "wrong",
                                   "session_id": TEST_SESSION_ID})
    response = client.post("/hint", json={"user_id": user_id, "question_id": question_id})
    assert response.status_code == 200;
    assert response.json()["hint"] == "Mocked LLM Hint."
    assert question_id in procedural_memory.get_hints_issued(user_id)


def test_get_hint_llm_failure_fallback(client, questions_df_loaded, monkeypatch):
    user_id = f"{TEST_USER_ID}_hint_llm_fail"
    monkeypatch.setattr("api.USE_LLM_FOR_HINTS", True)
    if not os.getenv("OPENAI_API_KEY"): monkeypatch.setenv("OPENAI_API_KEY", "dummy_for_test_fail")
    monkeypatch.setattr("api.client_openai", OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    monkeypatch.setattr("openai.OpenAI.chat.completions.create", mock_openai_chat_completion_failure)

    q_resp = client.get(f"/next_question?user_id={user_id}")
    q_data = q_resp.json();
    question_id = q_data["question_id"]
    q_detail = get_question_detail_from_df(question_id, questions_df_loaded)
    if not q_detail: pytest.skip("QID not found for hint fallback.")
    response = client.post("/hint", json={"user_id": user_id, "question_id": question_id})
    assert response.status_code == 200
    expl = q_detail.get("solution_explanation",
                        "Review the problem statement carefully and try to identify the first step.")
    m = re.match(r"^([^.]+\.)", expl);
    expected_hint_text = m.group(1) if m else expl.split('.')[0]
    if len(expected_hint_text) > 120 or (m is None and len(expl.split('.')[0]) < 10 and len(expl) > 10):
        expected_hint_text = (expl[:100] + "...") if len(expl) > 100 else expl
    if len(expected_hint_text) > 120: expected_hint_text = expected_hint_text[:117] + "..."  # Adjusted length
    if not expected_hint_text.strip(): expected_hint_text = "Think about the first step you would take to solve this."
    assert response.json()["hint"] == expected_hint_text


def test_get_hint_disabled_llm(client, questions_df_loaded, monkeypatch):
    user_id = f"{TEST_USER_ID}_hint_disabled"
    monkeypatch.setattr("api.USE_LLM_FOR_HINTS", False)
    monkeypatch.setattr("api.client_openai", None)  # Also ensure client is None
    q_resp = client.get(f"/next_question?user_id={user_id}")
    q_data = q_resp.json();
    question_id = q_data["question_id"]
    q_detail = get_question_detail_from_df(question_id, questions_df_loaded)
    if not q_detail: pytest.skip("QID not found for disabled LLM hint.")
    response = client.post("/hint", json={"user_id": user_id, "question_id": question_id})
    assert response.status_code == 200
    expl = q_detail.get("solution_explanation",
                        "Review the problem statement carefully and try to identify the first step.")
    m = re.match(r"^([^.]+\.)", expl);
    expected_hint_text = m.group(1) if m else expl.split('.')[0]
    if len(expected_hint_text) > 120 or (m is None and len(expl.split('.')[0]) < 10 and len(expl) > 10):
        expected_hint_text = (expl[:100] + "...") if len(expl) > 100 else expl
    if len(expected_hint_text) > 120: expected_hint_text = expected_hint_text[:117] + "..."
    if not expected_hint_text.strip(): expected_hint_text = "Think about the first step you would take to solve this."
    assert response.json()["hint"] == expected_hint_text


def test_get_hint_invalid_question_id(client):
    user_id = f"{TEST_USER_ID}_hint_invalid_qid"
    response = client.post("/hint", json={"user_id": user_id, "question_id": "INVALID_QID_FOR_HINT"})
    assert response.status_code == 404


def test_hint_success_recording(client, questions_df_loaded, monkeypatch):
    user_id = f"{TEST_USER_ID}_hint_success"
    monkeypatch.setattr("api.USE_LLM_FOR_HINTS", False)
    q_resp = client.get(f"/next_question?user_id={user_id}")
    q_data = q_resp.json();
    question_id = q_data["question_id"]
    q_detail = get_question_detail_from_df(question_id, questions_df_loaded)
    if not q_detail: pytest.skip("QID not found for hint success.")
    correct_answer = q_detail["correct_answer"]

    client.post("/hint", json={"user_id": user_id, "question_id": question_id})
    assert str(question_id) in procedural_memory.get_hints_issued(user_id)

    initial_count = procedural_memory.get_hint_success_count(user_id)

    client.post("/classify", json={"user_id": user_id, "question_id": question_id, "user_answer": correct_answer,
                                   "session_id": TEST_SESSION_ID})

    assert procedural_memory.get_hint_success_count(user_id) == initial_count + 1


def test_difficulty_progression(client, questions_df_loaded):
    user_id_streak = f"{TEST_USER_ID}_streak_final_reset"
    biographical_memory.set_overall_difficulty_preference(user_id_streak, 'easy')
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM topic_mastery WHERE user_id = ?", (user_id_streak,))
    cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id_streak,))
    conn.commit()
    conn.close()
    biographical_memory.set_overall_difficulty_preference(user_id_streak, 'easy')

    profile_before_loop = biographical_memory.get_profile(user_id_streak)
    assert profile_before_loop["current_difficulty_level"] == "easy"

    for i in range(STREAK_TO_ADVANCE_DIFFICULTY):
        q_resp = client.get(f"/next_question?user_id={user_id_streak}")
        assert q_resp.status_code == 200
        q_data = q_resp.json()
        question_id = q_data["question_id"]
        q_detail = get_question_detail_from_df(question_id, questions_df_loaded)
        if not q_detail: pytest.skip(f"QID {question_id} not found in difficulty_progression loop.")
        client.post("/classify", json={
            "user_id": user_id_streak, "question_id": question_id,
            "user_answer": q_detail["correct_answer"], "session_id": TEST_SESSION_ID})

    final_q_resp = client.get(f"/next_question?user_id={user_id_streak}")
    assert final_q_resp.status_code == 200

    final_difficulty_in_profile = biographical_memory.get_profile(user_id_streak)["current_difficulty_level"]
    expected_new_difficulty = DIFFICULTY_ORDER[DIFFICULTY_ORDER.index('easy') + 1]
    assert final_difficulty_in_profile == expected_new_difficulty


def test_topic_focus_on_struggle(client, questions_df_loaded):
    user_id_struggle = f"{TEST_USER_ID}_struggle_final_reset"
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM topic_mastery WHERE user_id = ?", (user_id_struggle,))
    cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id_struggle,))
    conn.commit()
    conn.close()
    biographical_memory.set_overall_difficulty_preference(user_id_struggle, 'easy')

    topics = questions_df_loaded['topic'].unique()
    if len(topics) < 2: pytest.skip("Not enough unique topics for topic focus.")
    topic_to_struggle_on = topics[0];
    other_topic = topics[1]

    for i in range(3):
        q_list = questions_df_loaded[questions_df_loaded['topic'] == other_topic]
        if q_list.empty: pytest.skip(f"No questions for other_topic {other_topic}")
        q_detail = q_list.sample(1).iloc[0].to_dict()
        client.post("/classify", json={"user_id": user_id_struggle, "question_id": q_detail["question_id"],
                                       "user_answer": q_detail["correct_answer"], "session_id": TEST_SESSION_ID})

    for i in range(3):
        q_list = questions_df_loaded[questions_df_loaded['topic'] == topic_to_struggle_on]
        if q_list.empty: pytest.skip(f"No questions for struggle_topic {topic_to_struggle_on}")
        q_detail = q_list.sample(1).iloc[0].to_dict()
        client.post("/classify", json={"user_id": user_id_struggle, "question_id": q_detail["question_id"],
                                       "user_answer": f"wrong_ans_{i}", "session_id": TEST_SESSION_ID})

    focused_topic_count = 0
    for i in range(5):
        next_q_resp = client.get(f"/next_question?user_id={user_id_struggle}")
        assert next_q_resp.status_code == 200
        next_q_data = next_q_resp.json()
        if next_q_data['topic'] == topic_to_struggle_on:
            focused_topic_count += 1
    assert focused_topic_count > 0, f"Policy did not focus on struggling topic {topic_to_struggle_on}"


# --- NEW Test for User Summary Endpoint ---
def test_get_user_summary(client, questions_df_loaded):
    user_id_summary = f"{TEST_USER_ID}_summary_test"
    session_id_summary = f"{TEST_SESSION_ID}_summary"

    # Ensure clean state for this user
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM episodic_events WHERE user_id = ?", (user_id_summary,))
    cursor.execute("DELETE FROM user_mistakes WHERE user_id = ?", (user_id_summary,))
    cursor.execute("DELETE FROM user_hints_issued WHERE user_id = ?", (user_id_summary,))
    cursor.execute("DELETE FROM user_hint_successes WHERE user_id = ?", (user_id_summary,))
    cursor.execute("DELETE FROM topic_mastery WHERE user_id = ?", (user_id_summary,))
    cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id_summary,))
    conn.commit()
    conn.close()

    # Set initial difficulty
    biographical_memory.set_overall_difficulty_preference(user_id_summary, "easy")

    # Simulate some activity
    # Attempt 1: Algebra, Correct
    q1_details = questions_df_loaded[questions_df_loaded['topic'] == 'Algebra'].iloc[0].to_dict()
    client.post("/classify", json={
        "user_id": user_id_summary, "question_id": q1_details['question_id'],
        "user_answer": q1_details['correct_answer'], "session_id": session_id_summary
    })

    # Attempt 2: Algebra, Incorrect
    q2_details = questions_df_loaded[questions_df_loaded['topic'] == 'Algebra'].iloc[1].to_dict()
    client.post("/classify", json={
        "user_id": user_id_summary, "question_id": q2_details['question_id'],
        "user_answer": "wrong_answer_algebra", "session_id": session_id_summary
    })

    # Attempt 3: Geometry, Correct, with Hint issued and success
    q3_details = questions_df_loaded[questions_df_loaded['topic'] == 'Geometry'].iloc[0].to_dict()
    client.post("/hint", json={"user_id": user_id_summary, "question_id": q3_details['question_id']})  # Request hint
    client.post("/classify", json={
        "user_id": user_id_summary, "question_id": q3_details['question_id'],
        "user_answer": q3_details['correct_answer'], "session_id": session_id_summary
    })

    # Attempt 4: Geometry, Incorrect, with Hint issued but no success
    q4_details = questions_df_loaded[questions_df_loaded['topic'] == 'Geometry'].iloc[1].to_dict()
    client.post("/hint", json={"user_id": user_id_summary, "question_id": q4_details['question_id']})  # Request hint
    client.post("/classify", json={
        "user_id": user_id_summary, "question_id": q4_details['question_id'],
        "user_answer": "wrong_answer_geometry", "session_id": session_id_summary
    })

    # Call the summary endpoint
    summary_response = client.get(f"/user_summary/{user_id_summary}")
    assert summary_response.status_code == 200
    summary_data = summary_response.json()

    # Assertions
    assert summary_data["user_id"] == user_id_summary
    assert summary_data["total_questions_answered_overall"] == 4  # 4 classification events

    # Overall accuracy: 2 correct out of 4 = 0.5
    assert summary_data["overall_accuracy"] == 0.5

    # Difficulty might have changed, check it's a valid one
    assert summary_data["current_difficulty_level"] in DIFFICULTY_ORDER

    assert len(summary_data["topic_summaries"]) == 2  # Algebra and Geometry

    algebra_summary = next((ts for ts in summary_data["topic_summaries"] if ts["topic_name"] == "Algebra"), None)
    assert algebra_summary is not None
    assert algebra_summary["total_attempted"] == 2
    assert algebra_summary["correct_attempted"] == 1
    assert algebra_summary["accuracy"] == 0.5
    # Mastery score depends on streak, check if it's roughly 50 + streak_bonus (0 for last incorrect)
    assert 50.0 <= algebra_summary[
        "mastery_score"] < 55.0  # 1 correct, 1 incorrect -> streak 0 for last one. Score = 50.

    geometry_summary = next((ts for ts in summary_data["topic_summaries"] if ts["topic_name"] == "Geometry"), None)
    assert geometry_summary is not None
    assert geometry_summary["total_attempted"] == 2
    assert geometry_summary["correct_attempted"] == 1
    assert geometry_summary["accuracy"] == 0.5
    # Mastery score: 1 correct, 1 incorrect. If correct was first, streak = 1. Score = 50 + 1 = 51.
    # If incorrect was first, streak = 0. Score = 50.
    # The order of q3 and q4 matters for streak. q3 was correct, q4 incorrect. So streak for geometry ends at 0 for q4.
    # The mastery score for geometry should be 50.0 based on the last event being incorrect.
    assert 50.0 <= geometry_summary["mastery_score"] < 55.0

    # Hints: 2 distinct questions had hints issued (Q3, Q4)
    assert summary_data["hints_requested_total"] == 2
    assert summary_data["hints_led_to_success_total"] == 1  # Only Q3 was a success after hint
