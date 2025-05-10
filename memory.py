# Adaptive-Tutor/memory.py

import time
from collections import defaultdict

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
        if keyword:
            return [p for p in self.policies if keyword.lower() in p.lower()]
        return self.policies

    def add_policy(self, text):
        self.policies.append(text)
        print(f"[ValuesMemory] Added policy: {text}")

class EpisodicMemory:
    """M2: Logs user's question attempts and interactions."""
    def __init__(self):
        self.log = []  # Stores dicts of each event

    def add_event(self, user_id, question_id, user_answer, classified_label, reward, session_id="default_session"):
        event = {
            "user_id": user_id,
            "question_id": question_id,
            "user_answer": user_answer,
            "classified_label": classified_label,
            "reward": reward,
            "timestamp": time.time(),
            "session_id": session_id
        }
        self.log.append(event)

    def get_user_history(self, user_id, last_n=None):
        history = [e for e in self.log if e["user_id"] == user_id]
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        return history[:last_n] if last_n else history

    def get_attempts_for_question(self, user_id, question_id):
        return [e for e in self.log if e["user_id"] == user_id and e["question_id"] == question_id]

class ProceduralMemory:
    """MProc: Tracks mistake types and hint efficacy per user."""
    def __init__(self):
        # user_id -> {mistake_type: count}
        self.user_mistakes = defaultdict(lambda: defaultdict(int))
        # user_id -> list of question_ids for which a hint was issued
        self.user_hints_issued = defaultdict(list)
        # user_id -> count of hint successes
        self.user_hint_success_count = defaultdict(int)
        # user_id -> [reflection_strings]
        self.user_reflections = defaultdict(list)

    def record_mistake(self, user_id, question_id, mistake_type="general_error"):
        """Increment the count for a particular mistake type."""
        self.user_mistakes[user_id][mistake_type] += 1

    def get_common_mistakes(self, user_id):
        """Return the dict of mistake_type -> count for this user."""
        return self.user_mistakes[user_id]

    def add_reflection(self, user_id, reflection_text):
        """Store a free-form reflection for the user."""
        self.user_reflections[user_id].append(reflection_text)

    def get_reflections(self, user_id):
        return self.user_reflections[user_id]

    # ─── New methods for hint tracking ──────────────────────────────────────────────

    def record_hint_issued(self, user_id, question_id):
        """Note that we gave a hint for this user/question."""
        self.user_hints_issued[user_id].append(question_id)

    def record_hint_success(self, user_id, question_id):
        """Called when a hint leads to a correct answer."""
        self.user_hint_success_count[user_id] += 1

    def get_hints_issued(self, user_id):
        """Return the list of question_ids for which hints were given."""
        return self.user_hints_issued[user_id]

    def get_hint_success_count(self, user_id):
        """Return how many times hints have led to success."""
        return self.user_hint_success_count[user_id]

class BiographicalMemory:
    """MBio: Stores user-specific preferences and long-term data."""
    def __init__(self):
        self.user_profiles = defaultdict(lambda: {
            "preferred_topics": [],
            "learning_goal": "general improvement",
            "current_difficulty_level": "easy",
            "topic_mastery": defaultdict(lambda: {"correct": 0, "total": 0, "streak": 0, "score": 0.0})
        })

    def update_preference(self, user_id, key, value):
        self.user_profiles[user_id][key] = value

    def get_profile(self, user_id):
        return self.user_profiles[user_id]

    def update_topic_mastery(self, user_id, topic, is_correct):
        mastery = self.user_profiles[user_id]["topic_mastery"][topic]
        mastery["total"] += 1
        if is_correct:
            mastery["correct"] += 1
            mastery["streak"] += 1
        else:
            mastery["streak"] = 0
        if mastery["total"] > 0:
            mastery["score"] = (mastery["correct"] / mastery["total"]) * 100 + min(mastery["streak"], 5)
        else:
            mastery["score"] = 0

    def get_topic_mastery(self, user_id, topic):
        return self.user_profiles[user_id]["topic_mastery"][topic]

    def get_overall_difficulty_preference(self, user_id):
        return self.user_profiles[user_id]["current_difficulty_level"]

    def set_overall_difficulty_preference(self, user_id, difficulty):
        self.user_profiles[user_id]["current_difficulty_level"] = difficulty

# Global instances
values_memory = ValuesMemory()
episodic_memory = EpisodicMemory()
procedural_memory = ProceduralMemory()
biographical_memory = BiographicalMemory()
