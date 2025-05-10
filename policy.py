# Adaptive-Tutor/policy.py

import random
import pandas as pd
from memory import biographical_memory, episodic_memory, procedural_memory

# Difficulty tiers
DIFFICULTY_ORDER = ['easy', 'medium', 'hard']
STREAK_TO_ADVANCE_DIFFICULTY = 3  # Number of consecutive correct answers to advance
RECENT_WINDOW = 5  # Consider last N attempts for streak calculation


def select_next_question(user_id, all_questions_df):
    """
    Heuristic-based policy for choosing the next question.
    Updates user's difficulty preference based on recent performance.
    """
    profile = biographical_memory.get_profile(user_id)  # This now fetches from DB
    history = episodic_memory.get_user_history(user_id, last_n=RECENT_WINDOW * 2)  # Fetches from DB

    attempted_qids_in_recent_history = {e['question_id'] for e in history}

    streak = 0
    for ev in history[:RECENT_WINDOW]:  # Check only the most recent relevant window
        if ev['classified_label'] == 'correct':
            streak += 1
        else:
            break

    current_difficulty_in_profile = profile.get('current_difficulty_level', 'easy')  # Use .get for safety
    new_difficulty_for_this_turn = current_difficulty_in_profile

    if streak >= STREAK_TO_ADVANCE_DIFFICULTY:
        if current_difficulty_in_profile != 'hard':
            try:
                current_diff_idx = DIFFICULTY_ORDER.index(current_difficulty_in_profile)
                new_difficulty_for_this_turn = DIFFICULTY_ORDER[current_diff_idx + 1]
                biographical_memory.set_overall_difficulty_preference(user_id,
                                                                      new_difficulty_for_this_turn)  # Updates DB
            except ValueError:
                # print(f"[Policy Warning] User {user_id} has invalid difficulty '{current_difficulty_in_profile}'. Resetting to 'easy'.")
                new_difficulty_for_this_turn = 'easy'
                biographical_memory.set_overall_difficulty_preference(user_id, new_difficulty_for_this_turn)

    mastery = profile.get('topic_mastery', {})  # Use .get for safety, mastery is a dict of dicts

    # CORRECTED KEYS: 'total_attempts' and 'mastery_score'
    # Also use .get() for safety when accessing keys within m_data
    struggling_topics = [
        t for t, m_data in mastery.items()
        if isinstance(m_data, dict) and m_data.get('total_attempts', 0) > 2 and m_data.get('mastery_score', 0.0) < 50
    ]

    chosen_topic_filter = None
    if struggling_topics:
        # Optional: sort struggling_topics by score to pick the lowest
        # struggling_topics.sort(key=lambda t: mastery.get(t, {}).get('mastery_score', 101))
        chosen_topic_filter = struggling_topics[0]

    pool = all_questions_df.copy()

    if chosen_topic_filter:
        topic_pool = pool[pool['topic'] == chosen_topic_filter]
        if not topic_pool.empty:
            pool = topic_pool

    difficulty_pool = pool[pool['difficulty'] == new_difficulty_for_this_turn]
    if not difficulty_pool.empty:
        pool = difficulty_pool

    # Fallbacks
    if pool.empty and chosen_topic_filter:
        pool = all_questions_df[all_questions_df['topic'] == chosen_topic_filter]

    if pool.empty:
        pool = all_questions_df[all_questions_df['difficulty'] == new_difficulty_for_this_turn]

    if not pool.empty:
        pool_after_attempt_filter = pool[~pool['question_id'].isin(attempted_qids_in_recent_history)]
        if not pool_after_attempt_filter.empty:
            pool = pool_after_attempt_filter

    if pool.empty:
        pool = all_questions_df[~all_questions_df['question_id'].isin(attempted_qids_in_recent_history)]
        if pool.empty:
            pool = all_questions_df

    if pool.empty:
        if all_questions_df.empty:
            raise ValueError("Policy Error: all_questions_df is empty, cannot select a question.")
        return all_questions_df.sample(1).iloc[0].to_dict()

    return pool.sample(1).iloc[0].to_dict()
