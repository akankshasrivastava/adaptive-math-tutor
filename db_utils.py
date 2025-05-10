# Adaptive-Tutor/db_utils.py
import sqlite3
import os

# Define the database file path
# It's good practice to place data files within a 'data' subdirectory.
DATA_DIR = "data"
DB_FILE = os.path.join(DATA_DIR, "tutor_memory.db")


def get_db_connection():
    """
    Establishes a connection to the SQLite database.
    Ensures the 'data' directory exists.
    Returns a sqlite3.Connection object with Row factory enabled.
    """
    # Ensure the data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn


def init_db():
    """
    Initializes the database by creating all necessary tables if they don't already exist.
    This function should be called once at application startup.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # --- EpisodicMemory Table ---
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS episodic_events
                   (
                       event_id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       user_id
                       TEXT
                       NOT
                       NULL,
                       session_id
                       TEXT,
                       question_id
                       TEXT
                       NOT
                       NULL,
                       user_answer
                       TEXT,
                       classified_label
                       TEXT,
                       reward
                       REAL,
                       timestamp
                       REAL
                       NOT
                       NULL
                   );
                   """)
    # Indexes for faster queries on episodic_events
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_user_id ON episodic_events (user_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_user_question ON episodic_events (user_id, question_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_timestamp ON episodic_events (timestamp);")

    # --- ProceduralMemory Tables ---
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS user_mistakes
                   (
                       mistake_id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       user_id
                       TEXT
                       NOT
                       NULL,
                       question_id
                       TEXT,
                       mistake_type
                       TEXT
                       NOT
                       NULL,
                       timestamp
                       REAL
                       NOT
                       NULL
                   );
                   """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mistakes_user_id ON user_mistakes (user_id);")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_mistakes_user_mistake_type ON user_mistakes (user_id, mistake_type);")

    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS user_reflections
                   (
                       reflection_id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       user_id
                       TEXT
                       NOT
                       NULL,
                       reflection_text
                       TEXT
                       NOT
                       NULL,
                       triggering_context
                       TEXT,
                       timestamp
                       REAL
                       NOT
                       NULL
                   );
                   """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reflections_user_id ON user_reflections (user_id);")

    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS user_hints_issued
                   (
                       hint_issue_id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       user_id
                       TEXT
                       NOT
                       NULL,
                       question_id
                       TEXT
                       NOT
                       NULL,
                       timestamp
                       REAL
                       NOT
                       NULL
                   );
                   """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_hints_issued_user_question ON user_hints_issued (user_id, question_id);")

    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS user_hint_successes
                   (
                       hint_success_id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       user_id
                       TEXT
                       NOT
                       NULL,
                       question_id
                       TEXT
                       NOT
                       NULL,
                       timestamp
                       REAL
                       NOT
                       NULL
                   );
                   """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_hint_successes_user_question ON user_hint_successes (user_id, question_id);")

    # --- BiographicalMemory Tables ---
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS user_profiles
                   (
                       user_id
                       TEXT
                       PRIMARY
                       KEY,
                       preferred_topics
                       TEXT, -- Store as JSON string e.g., '["Algebra", "Geometry"]'
                       learning_goal
                       TEXT
                       DEFAULT
                       'general improvement',
                       current_difficulty_level
                       TEXT
                       DEFAULT
                       'easy'
                   );
                   """)
    # No separate index needed for user_id as it's PRIMARY KEY.

    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS topic_mastery
                   (
                       mastery_id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       user_id
                       TEXT
                       NOT
                       NULL,
                       topic
                       TEXT
                       NOT
                       NULL,
                       correct_attempts
                       INTEGER
                       DEFAULT
                       0,
                       total_attempts
                       INTEGER
                       DEFAULT
                       0,
                       current_streak
                       INTEGER
                       DEFAULT
                       0,
                       mastery_score
                       REAL
                       DEFAULT
                       0.0,
                       FOREIGN
                       KEY
                   (
                       user_id
                   ) REFERENCES user_profiles
                   (
                       user_id
                   ) ON DELETE CASCADE,
                       UNIQUE
                   (
                       user_id,
                       topic
                   )
                       );
                   """)
    # ON DELETE CASCADE for user_id FK means if a user_profile is deleted, their topic_mastery records are also deleted.
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic_mastery_user_topic ON topic_mastery (user_id, topic);")

    conn.commit()
    conn.close()
    print(f"Database initialized/checked at {DB_FILE}")


if __name__ == '__main__':
    # This allows you to initialize the DB by running: python db_utils.py
    print("Initializing database schema...")
    init_db()
    print("Database schema initialization complete.")

    # Example of how to use the connection:
    # conn_test = get_db_connection()
    # # Do something with conn_test, e.g., conn_test.execute(...)
    # print("Test connection successful.")
    # conn_test.close()
