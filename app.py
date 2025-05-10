# Adaptive-Tutor/app.py

import streamlit as st
import requests
import uuid  # For generating user_id

# API Gateway
API_URL = "http://127.0.0.1:8000"

# --- Helper Functions to Call API ---
def get_next_question_from_api(user_id):
    try:
        response = requests.get(f"{API_URL}/next_question?user_id={user_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching next question: {e}")
        return None

def classify_answer_from_api(user_id, q_id, user_ans, session_id):
    payload = {
        "user_id": user_id,
        "question_id": q_id,
        "user_answer": user_ans,
        "session_id": session_id
    }
    try:
        response = requests.post(f"{API_URL}/classify", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error classifying answer: {e}")
        return None

def get_hint_from_api(user_id, q_id):
    payload = {"user_id": user_id, "question_id": q_id}
    try:
        response = requests.post(f"{API_URL}/hint", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching hint: {e}")
        return None

# --- Initialize Streamlit Session State ---
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())  # Generate a unique user ID
    st.session_state.session_id = str(uuid.uuid4())  # For this interaction session

if 'current_question' not in st.session_state:
    st.session_state.current_question = None

if 'feedback' not in st.session_state:
    st.session_state.feedback = None

if 'show_hint' not in st.session_state:
    st.session_state.show_hint = False

if 'hint_text' not in st.session_state:
    st.session_state.hint_text = ""

if 'answer_submitted' not in st.session_state:
    st.session_state.answer_submitted = False

# --- Main App Logic ---
st.set_page_config(page_title="Adaptive Math Tutor", layout="wide")
st.title("📚 Adaptive Math Tutor")
st.markdown(f"**User ID:** `{st.session_state.user_id}` (This is unique to your session)")

# Sidebar for controls / info
st.sidebar.header("Tutor Controls")
if st.sidebar.button("Start Over (New User ID)"):
    # Wipe all session state except the new IDs
    for key in list(st.session_state.keys()):
        if key not in ['user_id', 'session_id']:
            del st.session_state[key]
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.current_question = None
    st.session_state.feedback = None
    st.session_state.show_hint = False
    st.session_state.hint_text = ""
    st.session_state.answer_submitted = False
    st.rerun()

def load_new_question():
    """Loads a new question and resets all related state."""
    st.session_state.current_question = get_next_question_from_api(st.session_state.user_id)
    st.session_state.feedback = None
    st.session_state.show_hint = False
    st.session_state.hint_text = ""
    st.session_state.answer_submitted = False
    # Remove the previous input widget so it reappears empty
    st.session_state.pop("user_answer_input", None)

# Load the initial question
if st.session_state.current_question is None:
    load_new_question()

# --- Display Question and Handle Answer ---
if st.session_state.current_question:
    q_data = st.session_state.current_question

    # Header
    st.subheader(f"Question (ID: {q_data['question_id']})")
    st.markdown(f"**Topic:** {q_data['topic']} | **Difficulty:** {q_data['difficulty']}")

    # Render the question text: LaTeX for pure math, plain markdown otherwise
    qt = q_data['question_text']
    if all(c.isdigit() or c.isspace() or c in "+-*/^=()." for c in qt):
        st.latex(qt)
    else:
        st.markdown(qt)

    # Answer input
    user_answer = st.text_input(
        "Your Answer:",
        key="user_answer_input",
        disabled=st.session_state.answer_submitted
    )

    col1, col2, _ = st.columns([1, 1, 1])

    # Submit Answer button
    with col1:
        if st.button(
            "Submit Answer",
            disabled=st.session_state.answer_submitted or not user_answer
        ):
            st.session_state.feedback = classify_answer_from_api(
                st.session_state.user_id,
                q_data['question_id'],
                user_answer,
                st.session_state.session_id
            )
            st.session_state.answer_submitted = True
            st.rerun()  # Trigger re-render to show feedback

    # Ask for Hint button
    with col2:
        if st.button("💡 Ask for Hint", disabled=st.session_state.answer_submitted):
            hint_data = get_hint_from_api(st.session_state.user_id, q_data['question_id'])
            st.session_state.hint_text = hint_data.get('hint', "Sorry, no hint available.")
            st.session_state.show_hint = True
            st.rerun()

    # Show hint if requested
    if st.session_state.show_hint and st.session_state.hint_text:
        st.info(f"**Hint:** {st.session_state.hint_text}")

    # Show feedback after submission
    if st.session_state.feedback:
        fb = st.session_state.feedback
        label = fb['classified_label']

        if label == "correct":
            st.success("🎉 Correct!")
        elif label == "partial":
            st.warning("🔎 Partially Correct. Keep trying or view the explanation.")
            st.markdown(f"**Your Answer:** `{fb['user_answer']}`")
            st.markdown(f"**Correct Answer:** `{fb['correct_answer']}`")
            with st.expander("View Solution Explanation"):
                st.markdown(fb['solution_explanation'])
        else:
            st.error("❌ Incorrect.")
            st.markdown(f"**Your Answer:** `{fb['user_answer']}`")
            st.markdown(f"**Correct Answer:** `{fb['correct_answer']}`")
            with st.expander("View Solution Explanation"):
                st.markdown(fb['solution_explanation'])

        # Next Question button now clears input and loads a fresh one
        st.button(
            "Next Question",
            key="explicit_next_q_button",
            on_click=load_new_question
        )

else:
    st.warning("Could not load a question. Ensure the API is running.")
    if st.button("Try Reloading Question"):
        load_new_question()
        st.rerun()

# Optional styling to make buttons full-width
st.markdown("""
    <style>
        .stButton>button { width: 100%; }
    </style>
""", unsafe_allow_html=True)
