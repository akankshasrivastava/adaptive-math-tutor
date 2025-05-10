# Adaptive-Tutor/memory.py

import time
from collections import defaultdict
import sqlite3
import json  # For storing lists (like preferred_topics) as JSON strings in DB
from db_utils import get_db_connection


# --- ValuesMemory (Remains In-Memory) ---
class ValuesMemory:
    """M1: Stores core principles and policies of the tutor."""

    def __init__(self):
        self.policies = [
            "Be encouraging and supportive.",
            "Aim for clarity in explanations.",
            "Adapt to the user's learning pace.",
            "Focus on understanding, not just memorization."
        ]

    def get_policy(self, keyword=None):
        if keyword: return [p for p in self.policies if keyword.lower() in p.lower()]
        return self.policies

    def add_policy(self, text):
        self.policies.append(text)
        # print(f"[ValuesMemory] Added policy: {text}")


# --- EpisodicMemory (DB-Backed) ---
class EpisodicMemory:
    """M2: Logs user's question attempts and interactions using SQLite."""

    def __init__(self):
        pass  # DB initialized at API startup

    def add_event(self, user_id, question_id, user_answer, classified_label, reward, session_id="default_session"):
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            current_timestamp = time.time()
            cursor.execute("""
                           INSERT INTO episodic_events
                           (user_id, session_id, question_id, user_answer, classified_label, reward, timestamp)
                           VALUES (?, ?, ?, ?, ?, ?, ?)
                           """, (user_id, session_id, str(question_id), user_answer, classified_label, reward,
                                 current_timestamp))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error in EpisodicMemory.add_event: {e}")
        finally:
            if conn: conn.close()

    def get_user_history(self, user_id, last_n=None):
        history = []
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            # Ensure all columns as defined in the table are selected
            query = "SELECT event_id, user_id, session_id, question_id, user_answer, classified_label, reward, timestamp FROM episodic_events WHERE user_id = ? ORDER BY timestamp DESC"
            params = (user_id,)
            if last_n is not None and isinstance(last_n, int) and last_n > 0:
                query += " LIMIT ?"
                params += (last_n,)
            cursor.execute(query, params)
            history = [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Database error in EpisodicMemory.get_user_history: {e}")
        finally:
            if conn: conn.close()
        return history

    def get_attempts_for_question(self, user_id, question_id):
        attempts = []
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            # Ensure all columns as defined in the table are selected
            query = "SELECT event_id, user_id, session_id, question_id, user_answer, classified_label, reward, timestamp FROM episodic_events WHERE user_id = ? AND question_id = ? ORDER BY timestamp ASC"
            params = (user_id, str(question_id))
            cursor.execute(query, params)
            attempts = [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Database error in EpisodicMemory.get_attempts_for_question: {e}")
        finally:
            if conn: conn.close()
        return attempts


# --- ProceduralMemory (DB-Backed) ---
class ProceduralMemory:
    """MProc: Tracks mistake types, reflections, and hint efficacy per user using SQLite."""

    def __init__(self):
        pass

    def record_mistake(self, user_id, question_id, mistake_type="general_error"):
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            current_timestamp = time.time()
            cursor.execute(
                "INSERT INTO user_mistakes (user_id, question_id, mistake_type, timestamp) VALUES (?, ?, ?, ?)",
                (user_id, str(question_id), mistake_type, current_timestamp))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error in ProceduralMemory.record_mistake: {e}")
        finally:
            if conn: conn.close()

    def get_common_mistakes(self, user_id):
        mistakes = defaultdict(int)
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT mistake_type, COUNT(*) as count FROM user_mistakes WHERE user_id = ? GROUP BY mistake_type ORDER BY count DESC",
                (user_id,))
            for row in cursor.fetchall(): mistakes[row['mistake_type']] = row['count']
        except sqlite3.Error as e:
            print(f"Database error in ProceduralMemory.get_common_mistakes: {e}")
        finally:
            if conn: conn.close()
        return mistakes

    def add_reflection(self, user_id, reflection_text, triggering_context=None):
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            current_timestamp = time.time()
            cursor.execute(
                "INSERT INTO user_reflections (user_id, reflection_text, triggering_context, timestamp) VALUES (?, ?, ?, ?)",
                (user_id, reflection_text, triggering_context, current_timestamp))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error in ProceduralMemory.add_reflection: {e}")
        finally:
            if conn: conn.close()

    def get_reflections(self, user_id, last_n=None):
        reflections = []
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            # Ensure all columns as defined in the table are selected
            query = "SELECT reflection_id, user_id, reflection_text, triggering_context, timestamp FROM user_reflections WHERE user_id = ? ORDER BY timestamp DESC"
            params = (user_id,)
            if last_n is not None and isinstance(last_n, int) and last_n > 0:
                query += " LIMIT ?"
                params += (last_n,)
            cursor.execute(query, params)
            reflections = [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Database error in ProceduralMemory.get_reflections: {e}")
        finally:
            if conn: conn.close()
        return reflections

    def record_hint_issued(self, user_id, question_id):
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            current_timestamp = time.time()
            cursor.execute("INSERT INTO user_hints_issued (user_id, question_id, timestamp) VALUES (?, ?, ?)",
                           (user_id, str(question_id), current_timestamp))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error in ProceduralMemory.record_hint_issued: {e}")
        finally:
            if conn: conn.close()

    def record_hint_success(self, user_id, question_id):
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            current_timestamp = time.time()
            cursor.execute("INSERT INTO user_hint_successes (user_id, question_id, timestamp) VALUES (?, ?, ?)",
                           (user_id, str(question_id), current_timestamp))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error in ProceduralMemory.record_hint_success: {e}")
        finally:
            if conn: conn.close()

    def get_hints_issued(self, user_id):
        hinted_qids = []
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT question_id FROM user_hints_issued WHERE user_id = ?", (user_id,))
            hinted_qids = [row['question_id'] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Database error in ProceduralMemory.get_hints_issued: {e}")
        finally:
            if conn: conn.close()
        return hinted_qids

    def get_hint_success_count(self, user_id):
        count = 0
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as success_count FROM user_hint_successes WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            if result: count = result['success_count']
        except sqlite3.Error as e:
            print(f"Database error in ProceduralMemory.get_hint_success_count: {e}")
        finally:
            if conn: conn.close()
        return count


# --- BiographicalMemory (DB-Backed) ---
class BiographicalMemory:
    """MBio: Stores user-specific preferences and long-term data using SQLite."""

    def __init__(self):
        pass  # DB initialized at API startup

    def _ensure_user_profile_exists(self, conn, user_id):
        """Helper to ensure a user profile row exists, creating with defaults if not."""
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO user_profiles (user_id) VALUES (?)", (user_id,))
        # No commit here; calling function manages transaction.

    def update_preference(self, user_id, key, value):
        """Updates a specific preference in the user_profiles table."""
        conn = None
        try:
            conn = get_db_connection()
            self._ensure_user_profile_exists(conn, user_id)
            column_name = None
            db_value = value
            if key == "preferred_topics":
                db_value = json.dumps(value if isinstance(value, list) else [])
                column_name = "preferred_topics"
            elif key == "learning_goal":
                column_name = "learning_goal"
            else:
                # print(f"[BiographicalMemory DB] Warning: update_preference with unhandled key '{key}' for user {user_id}")
                return
            if column_name:
                cursor = conn.cursor()
                cursor.execute(f"UPDATE user_profiles SET {column_name} = ? WHERE user_id = ?", (db_value, user_id))
            conn.commit()
        except sqlite3.Error as e:
            print(f"DB error in BiographicalMemory.update_preference ({key}): {e}")
            if conn: conn.rollback()
        except json.JSONDecodeError as e:
            print(f"JSON error in BiographicalMemory.update_preference ({key}): {e}")
            if conn: conn.rollback()
        finally:
            if conn: conn.close()

    def get_profile(self, user_id):
        """
        Retrieves a user's profile from the database.
        Returns a dictionary with default values if the user or parts of their profile don't exist.
        Topic mastery is fetched separately and included.
        """
        profile_data = {
            "user_id": user_id, "preferred_topics": [],
            "learning_goal": "general improvement", "current_difficulty_level": "easy",
            "topic_mastery": defaultdict(
                lambda: {"topic": "", "correct_attempts": 0, "total_attempts": 0, "current_streak": 0,
                         "mastery_score": 0.0})
        }
        conn = None
        try:
            conn = get_db_connection()
            self._ensure_user_profile_exists(conn, user_id)
            conn.commit()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_id, preferred_topics, learning_goal, current_difficulty_level FROM user_profiles WHERE user_id = ?",
                (user_id,))
            user_row = cursor.fetchone()
            if user_row:
                user_dict = dict(user_row)
                # Update only if DB value is not None, otherwise keep default from profile_data
                profile_data["learning_goal"] = user_dict.get("learning_goal") if user_dict.get(
                    "learning_goal") is not None else profile_data["learning_goal"]
                profile_data["current_difficulty_level"] = user_dict.get("current_difficulty_level") if user_dict.get(
                    "current_difficulty_level") is not None else profile_data["current_difficulty_level"]

                stored_preferred_topics = user_dict.get("preferred_topics")
                if stored_preferred_topics:
                    try:
                        loaded_topics = json.loads(stored_preferred_topics)
                        if isinstance(loaded_topics, list): profile_data["preferred_topics"] = loaded_topics
                    except:
                        profile_data["preferred_topics"] = []
                else:
                    profile_data["preferred_topics"] = []

            cursor.execute(
                "SELECT topic, correct_attempts, total_attempts, current_streak, mastery_score FROM topic_mastery WHERE user_id = ?",
                (user_id,))
            for row in cursor.fetchall():
                topic_name = row['topic']
                profile_data["topic_mastery"][topic_name] = dict(row)
                profile_data["topic_mastery"][topic_name]['topic'] = topic_name
        except sqlite3.Error as e:
            print(f"DB error in BiographicalMemory.get_profile for {user_id}: {e}")
        finally:
            if conn: conn.close()
        return profile_data

    def update_topic_mastery(self, user_id, topic, is_correct):
        conn = None
        try:
            conn = get_db_connection()
            self._ensure_user_profile_exists(conn, user_id)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT correct_attempts, total_attempts, current_streak FROM topic_mastery WHERE user_id = ? AND topic = ?",
                (user_id, topic))
            row = cursor.fetchone()
            correct, total, streak = (row['correct_attempts'], row['total_attempts'],
                                      row['current_streak']) if row else (0, 0, 0)
            total += 1
            if is_correct:
                correct += 1; streak += 1
            else:
                streak = 0
            score = (correct / total) * 100 + min(streak, 5) if total > 0 else 0.0
            cursor.execute("""
                INSERT OR REPLACE INTO topic_mastery 
                    (user_id, topic, correct_attempts, total_attempts, current_streak, mastery_score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, topic, correct, total, streak, score))
            conn.commit()
        except sqlite3.Error as e:
            print(f"DB error in BiographicalMemory.update_topic_mastery ({user_id}, {topic}): {e}")
            if conn: conn.rollback()
        finally:
            if conn: conn.close()

    def get_topic_mastery(self, user_id, topic):
        mastery_data = {"topic": topic, "correct_attempts": 0, "total_attempts": 0, "current_streak": 0,
                        "mastery_score": 0.0}
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT topic, correct_attempts, total_attempts, current_streak, mastery_score FROM topic_mastery WHERE user_id = ? AND topic = ?",
                (user_id, topic))
            row = cursor.fetchone()
            if row: mastery_data.update(dict(row))
        except sqlite3.Error as e:
            print(f"DB error in BiographicalMemory.get_topic_mastery ({user_id}, {topic}): {e}")
        finally:
            if conn: conn.close()
        return mastery_data

    def get_overall_difficulty_preference(self, user_id):
        difficulty = "easy"
        conn = None
        try:
            conn = get_db_connection()
            self._ensure_user_profile_exists(conn, user_id)
            conn.commit()
            cursor = conn.cursor()
            cursor.execute("SELECT current_difficulty_level FROM user_profiles WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row and row['current_difficulty_level']: difficulty = row['current_difficulty_level']
        except sqlite3.Error as e:
            print(f"DB error in BiographicalMemory.get_overall_difficulty_preference for {user_id}: {e}")
        finally:
            if conn: conn.close()
        return difficulty

    def set_overall_difficulty_preference(self, user_id, difficulty):
        conn = None
        try:
            conn = get_db_connection()
            self._ensure_user_profile_exists(conn, user_id)
            cursor = conn.cursor()
            cursor.execute("UPDATE user_profiles SET current_difficulty_level = ? WHERE user_id = ?",
                           (difficulty, user_id))
            conn.commit()
        except sqlite3.Error as e:
            print(f"DB error in BiographicalMemory.set_overall_difficulty_preference for {user_id}: {e}")
            if conn: conn.rollback()
        finally:
            if conn: conn.close()


# Global instances
values_memory = ValuesMemory()
episodic_memory = EpisodicMemory()
procedural_memory = ProceduralMemory()
biographical_memory = BiographicalMemory()  # Now fully DB-backed
