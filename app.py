# Adaptive-Tutor/app.py

import streamlit as st
import requests
import uuid

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
    payload = {"user_id": user_id, "question_id": q_id, "user_answer": user_ans, "session_id": session_id}
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


def get_user_summary_from_api(user_id):
    try:
        response = requests.get(f"{API_URL}/user_summary/{user_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching user summary: {e}")
        return None


# --- End Helper Functions ---

# --- Initialize Session State ---
default_session_state = {
    'user_id': str(uuid.uuid4()), 'session_id': str(uuid.uuid4()),
    'current_question': None, 'feedback': None, 'hint_text': None,
    'show_hint': False, 'answer_submitted': False,
    'text_input_key_counter': 0, 'questions_answered_session': 0,
    'user_answer_value': "",
    'user_summary_data': None
}
for key, default_value in default_session_state.items():
    if key not in st.session_state: st.session_state[key] = default_value

# --- Main App UI ---
st.set_page_config(page_title="Adaptive Math Tutor", layout="wide")
st.title("📚 Adaptive Math Tutor")
st.markdown(f"**User ID:** `{st.session_state.user_id}`")
st.markdown(f"**Questions Answered (This Session):** {st.session_state.questions_answered_session}")

# --- Sidebar ---
st.sidebar.header("Tutor Controls")
if st.sidebar.button("Start Over (New User ID)"):
    new_user_id = str(uuid.uuid4());
    new_session_id = str(uuid.uuid4())
    for key in list(st.session_state.keys()): del st.session_state[key]
    for key, default_value in default_session_state.items(): st.session_state[key] = default_value
    st.session_state.user_id = new_user_id;
    st.session_state.session_id = new_session_id
    st.rerun()

# --- Performance Summary Section (in Sidebar) ---
with st.sidebar.expander("My Progress Summary", expanded=False):
    if st.button("Show/Refresh My Progress"):
        st.session_state.user_summary_data = get_user_summary_from_api(st.session_state.user_id)
        # No st.rerun() needed here, Streamlit will rerun due to state change if data is fetched

    if st.session_state.user_summary_data:
        summary = st.session_state.user_summary_data
        st.metric("Overall Accuracy", f"{summary.get('overall_accuracy', 0) * 100:.1f}%")
        st.metric("Total Questions Answered (Overall)", summary.get('total_questions_answered_overall', 0))
        st.metric("Current Difficulty Level", summary.get('current_difficulty_level', 'N/A').capitalize())

        st.markdown("---")
        st.subheader("Topic Performance:")
        if summary.get("topic_summaries"):
            for topic_data in summary["topic_summaries"]:
                st.markdown(f"**{topic_data.get('topic_name', 'Unknown Topic')}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{topic_data.get('accuracy', 0) * 100:.1f}%", delta_color="off")
                    st.metric("Mastery Score", f"{topic_data.get('mastery_score', 0):.1f}", delta_color="off")
                with col2:
                    st.metric("Attempted", topic_data.get('total_attempted', 0), delta_color="off")
                    st.metric("Correct", topic_data.get('correct_attempted', 0), delta_color="off")

                # Ensure progress value is between 0.0 and 1.0
                progress_val = min(max(topic_data.get('mastery_score', 0) / 100.0, 0.0), 1.0)
                st.progress(progress_val)
                st.markdown("---")
        else:
            st.write("No topic performance data yet.")

        st.subheader("Hint Usage:")
        st.metric("Hints Requested (for Qs)", summary.get('hints_requested_total', 0))
        st.metric("Hints Led to Success", summary.get('hints_led_to_success_total', 0))
    elif 'user_id' in st.session_state and st.session_state.user_summary_data is None:
        pass  # Button not clicked or API returned None


# --- END Performance Summary ---


def load_new_question():
    st.session_state.current_question = get_next_question_from_api(st.session_state.user_id)
    st.session_state.feedback = None
    st.session_state.show_hint = False;
    st.session_state.hint_text = None
    st.session_state.answer_submitted = False
    st.session_state.text_input_key_counter += 1
    st.session_state.user_answer_value = ""


if st.session_state.current_question is None and 'user_id' in st.session_state:
    load_new_question()

# --- Main Question Area ---
if st.session_state.current_question:
    q_data = st.session_state.current_question

    main_col, _ = st.columns([2, 1])

    with main_col:
        st.subheader(f"Question (ID: {q_data.get('question_id', 'N/A')})")
        st.markdown(f"**Topic:** {q_data.get('topic', 'N/A')} | **Difficulty:** {q_data.get('difficulty', 'N/A')}")

        qt = q_data.get('question_text', 'Question text not available.')
        render_as = q_data.get('render_as', 'markdown')

        if render_as == 'latex':
            if qt.count(' ') < 7 and any(c in qt for c in ['=', '\\', '^', '_', '+', '*', '/']):
                st.latex(qt)
            else:
                st.markdown(qt, unsafe_allow_html=True)
        else:
            st.markdown(qt, unsafe_allow_html=True)


        def text_input_on_change():
            widget_key = f"answer_input_key_{st.session_state.text_input_key_counter}"
            if widget_key in st.session_state:
                st.session_state.user_answer_value = st.session_state[widget_key]


        st.text_input(
            "Your Answer:", value=st.session_state.user_answer_value,
            key=f"answer_input_key_{st.session_state.text_input_key_counter}",
            on_change=text_input_on_change,
            disabled=st.session_state.answer_submitted,
            placeholder="Type your answer here..."
        )

        current_typed_answer = st.session_state.user_answer_value

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            submit_is_disabled = st.session_state.answer_submitted or not current_typed_answer.strip()
            button_visual_type = "primary" if not submit_is_disabled else "secondary"

            if st.button("Submit Answer",
                         disabled=submit_is_disabled,
                         use_container_width=True,
                         type=button_visual_type):
                st.session_state.feedback = classify_answer_from_api(
                    st.session_state.user_id, q_data['question_id'], current_typed_answer, st.session_state.session_id
                )
                if st.session_state.feedback: st.session_state.questions_answered_session += 1
                st.session_state.answer_submitted = True
                st.rerun()

        with btn_col2:
            if st.button("🤔 Ask for Hint", disabled=st.session_state.answer_submitted, use_container_width=True):
                hint_data = get_hint_from_api(st.session_state.user_id, q_data['question_id'])
                st.session_state.hint_text = hint_data.get('hint',
                                                           "Sorry, no hint available.") if hint_data else "Could not retrieve hint."
                st.session_state.show_hint = True
                st.rerun()

        if st.session_state.show_hint and st.session_state.hint_text:
            st.info(f"**Hint:** {st.session_state.hint_text}")

        if st.session_state.feedback:
            fb = st.session_state.feedback;
            label = fb.get('classified_label', 'unknown')
            if label == "correct":
                st.success("🎉 Correct!")
            elif label == "partial":
                st.warning("👍 Partially Correct. Almost there!")
            else:
                st.error("🤔 Incorrect. Let's review.")
            if label != "correct":
                st.markdown(f"**Your Answer:** `{fb.get('user_answer', '')}`")
                st.markdown(f"**Correct Answer:** `{fb.get('correct_answer', 'N/A')}`")
            if fb.get('solution_explanation'):
                with st.expander("View Solution Explanation"):
                    st.markdown(fb['solution_explanation'], unsafe_allow_html=True)

            if st.button("Next Question ➡️", on_click=load_new_question, use_container_width=True, type="primary"):
                pass
else:
    st.warning("Could not load a question. Ensure API is running and refresh.")
    if st.button("Try Reloading Question", use_container_width=True):
        st.session_state.current_question = None
        st.session_state.text_input_key_counter += 1
        st.session_state.user_answer_value = ""
        st.rerun()
