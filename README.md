# adaptive-math-tutor/README.md

# Adaptive Math Tutor

This project is a prototype of an end-to-end Adaptive Math Tutor web application. It features synthetic math question generation, NLP-based answer classification, a minimal reinforcement learning (RL) approach for question selection, a MIRO-inspired memory system, hint generation, a Python backend API (FastAPI), and a Streamlit frontend.

## Features

1.  **Synthetic Data Generation**: Creates a CSV dataset of short-answer math questions covering topics like Arithmetic, Algebra, Geometry, and Word Problems with varying difficulty.
2.  **NLP Classification**: Classifies user's short answers as 'correct', 'partial', or 'incorrect' using a Logistic Regression model trained on custom features.
3.  **Adaptive Question Selection**: A heuristic-based policy (minimal RL) selects the next question based on user performance, topic mastery, and question difficulty.
4.  **MIRO Multi-Part Memory**:
    * **ValuesMemory (M1)**: Core principles (not actively used by logic yet).
    * **EpisodicMemory (M2)**: Logs user interactions (question attempts, answers, outcomes).
    * **ProceduralMemory (Mproc)**: Tracks common mistake types (basic implementation).
    * **BiographicalMemory (Mbio)**: Stores user preferences, topic mastery, and current difficulty level.
5.  **Hint Generation**: Provides hints for questions, either via a mocked LLM call (placeholder) or by falling back to the question's solution explanation (or its first sentence).
6.  **Backend**: FastAPI server providing endpoints for question selection, answer classification, and hint requests.
7.  **Frontend**: Streamlit web interface for users to interact with the tutor.

## Project Structure