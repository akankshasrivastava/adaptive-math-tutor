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
    # ---- DEBUG PRINTS (Uncomment to trace policy execution during tests) ----
    # print(f"\n[Policy] User: {user_id}")
    # ---- END DEBUG PRINTS ----

    profile = biographical_memory.get_profile(user_id)
    # History is sorted: most recent first
    history = episodic_memory.get_user_history(user_id, last_n=RECENT_WINDOW * 2)  # Get a bit more history just in case

    # ---- DEBUG PRINTS ----
    # print(f"[Policy] Current profile difficulty: {profile['current_difficulty_level']}")
    # print(f"[Policy] Recent history (up to {RECENT_WINDOW} for streak):")
    # for i, ev in enumerate(history[:RECENT_WINDOW]):
    #     print(f"[Policy]  - Hist {i}: QID={ev['question_id']}, Label={ev['classified_label']}, TS={ev['timestamp']}")
    # ---- END DEBUG PRINTS ----

    attempted_qids_in_recent_history = {e['question_id'] for e in history}

    streak = 0
    for ev in history[:RECENT_WINDOW]:
        if ev['classified_label'] == 'correct':
            streak += 1
        else:
            break

    # ---- DEBUG PRINTS ----
    # print(f"[Policy] Calculated streak: {streak}")
    # ---- END DEBUG PRINTS ----

    current_difficulty_in_profile = profile['current_difficulty_level']
    new_difficulty_for_this_turn = current_difficulty_in_profile

    if streak >= STREAK_TO_ADVANCE_DIFFICULTY:
        if current_difficulty_in_profile != 'hard':
            current_diff_idx = DIFFICULTY_ORDER.index(current_difficulty_in_profile)
            new_difficulty_for_this_turn = DIFFICULTY_ORDER[current_diff_idx + 1]
            biographical_memory.set_overall_difficulty_preference(user_id, new_difficulty_for_this_turn)
            # ---- DEBUG PRINTS ----
            # print(f"[Policy] Streak condition met! Promoting difficulty from {current_difficulty_in_profile} to {new_difficulty_for_this_turn}")
            # ---- END DEBUG PRINTS ----
        # else:
        # ---- DEBUG PRINTS ----
        # print(f"[Policy] Streak condition met, but already at 'hard'. Staying at 'hard'.")
        # ---- END DEBUG PRINTS ----
    # else:
    # ---- DEBUG PRINTS ----
    # print(f"[Policy] Streak condition NOT met (streak {streak} < {STREAK_TO_ADVANCE_DIFFICULTY}). Difficulty remains {current_difficulty_in_profile}.")
    # ---- END DEBUG PRINTS ----

    mastery = profile['topic_mastery']
    struggling_topics = [t for t, m_data in mastery.items() if m_data['total'] > 2 and m_data['score'] < 50]

    chosen_topic_filter = None
    if struggling_topics:
        chosen_topic_filter = struggling_topics[0]
        # ---- DEBUG PRINTS ----
        # print(f"[Policy] Focusing on struggling topic: {chosen_topic_filter}")
        # ---- END DEBUG PRINTS ----

    pool = all_questions_df.copy()

    if chosen_topic_filter:
        topic_pool = pool[pool['topic'] == chosen_topic_filter]
        if not topic_pool.empty:  # CHANGED
            pool = topic_pool
        # else:
        # ---- DEBUG PRINTS ----
        # print(f"[Policy] Warning: Struggling topic {chosen_topic_filter} has no questions in the main pool.")
        # ---- END DEBUG PRINTS ----

    difficulty_pool = pool[pool['difficulty'] == new_difficulty_for_this_turn]
    if not difficulty_pool.empty:  # CHANGED
        pool = difficulty_pool
    # else:
    # ---- DEBUG PRINTS ----
    # print(f"[Policy] No questions found for difficulty '{new_difficulty_for_this_turn}' (topic: {chosen_topic_filter}). Broadening difficulty search.")
    # ---- END DEBUG PRINTS ----

    # Fallbacks
    if pool.empty and chosen_topic_filter:  # CHANGED
        pool = all_questions_df[all_questions_df['topic'] == chosen_topic_filter]
        # ---- DEBUG PRINTS ----
        # print(f"[Policy] Fallback 1: No Qs at diff/topic. Relaxed difficulty, keeping topic {chosen_topic_filter}. Pool size: {len(pool)}")
        # ---- END DEBUG PRINTS ----

    if pool.empty:  # CHANGED
        pool = all_questions_df[all_questions_df['difficulty'] == new_difficulty_for_this_turn]
        # ---- DEBUG PRINTS ----
        # print(f"[Policy] Fallback 2: No Qs at diff/topic or just topic. Relaxed topic, keeping difficulty {new_difficulty_for_this_turn}. Pool size: {len(pool)}")
        # ---- END DEBUG PRINTS ----

    if not pool.empty:  # CHANGED
        pool_after_attempt_filter = pool[~pool['question_id'].isin(attempted_qids_in_recent_history)]
        if not pool_after_attempt_filter.empty:  # CHANGED
            pool = pool_after_attempt_filter
        # else:
        # ---- DEBUG PRINTS ----
        # print(f"[Policy] Warning: All questions in the current filtered pool were recently attempted. Allowing repeats from this pool (size {len(pool)}).")
        # ---- END DEBUG PRINTS ----

    if pool.empty:  # CHANGED
        pool = all_questions_df[~all_questions_df['question_id'].isin(attempted_qids_in_recent_history)]
        # ---- DEBUG PRINTS ----
        # print(f"[Policy] Fallback 3: Pool was empty after filters. Using all questions not recently attempted. Pool size: {len(pool)}")
        # ---- END DEBUG PRINTS ----
        if pool.empty:  # CHANGED
            pool = all_questions_df
            # ---- DEBUG PRINTS ----
            # print(f"[Policy] Fallback 4: All questions recently attempted. Picking from entire dataset. Pool size: {len(pool)}")
            # ---- END DEBUG PRINTS ----

    if pool.empty:  # CHANGED
        if all_questions_df.empty:
            raise ValueError("Policy Error: all_questions_df is empty, cannot select a question.")
        # print("[Policy] CRITICAL FALLBACK: No questions available in any pool after all fallbacks, returning a random sample from all_questions_df.")
        chosen = all_questions_df.sample(1).iloc[0]
    else:
        chosen = pool.sample(1).iloc[0]

    # ---- DEBUG PRINTS ----
    # print(f"[Policy] Final chosen question ID: {chosen['question_id']}, Topic: {chosen['topic']}, Difficulty: {chosen['difficulty']}")
    # print(f"[Policy] User {user_id} profile difficulty after this selection: {biographical_memory.get_profile(user_id)['current_difficulty_level']}")
    # ---- END DEBUG PRINTS ----

    return chosen.to_dict()