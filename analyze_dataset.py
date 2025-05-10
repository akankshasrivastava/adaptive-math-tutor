# analyze_dataset.py
import pandas as pd
import numpy as np
import re
import os
from collections import Counter

DATA_FILE = "data/math_questions.csv"
EXPECTED_COLUMNS = ["question_id", "topic", "question_text", "correct_answer", "label", "solution_explanation",
                    "difficulty"]


def analyze_math_questions_csv(file_path):
    """
    Analyzes the math_questions.csv file for quality, quantity, and specific features.
    """
    print(f"--- Comprehensive Analysis of {file_path} ---\n")

    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"ERROR: Could not read CSV file. Reason: {e}")
        return

    # --- I. Basic File and Column Checks ---
    print("\n--- I. Basic File and Column Checks ---")
    print(f"Total number of questions (rows): {len(df)}")

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing expected columns: {missing_cols}")
    else:
        print(f"All expected columns are present: {list(df.columns)}")

    print("\nColumn Data Types:")
    print(df.dtypes)

    print(f"\nNumber of unique question_id values: {df['question_id'].nunique()}")
    if df['question_id'].nunique() != len(df):
        print(
            f"WARNING: Number of unique question_ids ({df['question_id'].nunique()}) does not match total rows ({len(df)}). Possible duplicate IDs.")
        duplicate_ids = df[df.duplicated(subset=['question_id'], keep=False)]['question_id']
        if not duplicate_ids.empty:
            print(f"Duplicate question_ids found: {list(duplicate_ids.unique())}")

    # --- II. Content Analysis - Per Column ---
    print("\n\n--- II. Content Analysis - Per Column ---")

    # `topic`
    print("\n-- Column: 'topic' --")
    if 'topic' in df.columns:
        print(f"Distribution of questions per topic:\n{df['topic'].value_counts().to_string()}")
        print(f"Number of unique topics: {df['topic'].nunique()}")
    else:
        print("Column 'topic' not found.")

    # `difficulty`
    print("\n-- Column: 'difficulty' --")
    if 'difficulty' in df.columns:
        print(f"Distribution of questions per difficulty level:\n{df['difficulty'].value_counts().to_string()}")
        expected_difficulties = ['easy', 'medium', 'hard']
        actual_difficulties = df['difficulty'].unique()
        for diff in actual_difficulties:
            if diff not in expected_difficulties:
                print(f"WARNING: Unexpected difficulty value found: '{diff}'")
    else:
        print("Column 'difficulty' not found.")

    # `question_text`
    print("\n-- Column: 'question_text' --")
    if 'question_text' in df.columns:
        df['question_text_len'] = df['question_text'].astype(str).apply(len)
        print(f"Question text length stats:\n{df['question_text_len'].describe().to_string()}")
        empty_questions = df[df['question_text_len'] == 0]
        if not empty_questions.empty:
            print(f"WARNING: Found {len(empty_questions)} questions with empty text.")

        # LaTeX-like syntax check (simple check)
        latex_chars = ['^', '\\', '_', '$', '{', '}']
        df['contains_latex_like'] = df['question_text'].astype(str).apply(
            lambda x: any(char in x for char in latex_chars))
        print(
            f"Number of questions containing common LaTeX-like characters: {df['contains_latex_like'].sum()} / {len(df)}")
    else:
        print("Column 'question_text' not found.")

    # `correct_answer`
    print("\n-- Column: 'correct_answer' --")
    if 'correct_answer' in df.columns:
        df['correct_answer_str'] = df['correct_answer'].astype(str)  # Ensure string type for analysis
        df['correct_answer_len'] = df['correct_answer_str'].apply(len)
        print(f"Correct answer length stats:\n{df['correct_answer_len'].describe().to_string()}")
        empty_answers = df[df['correct_answer_len'] == 0]
        if not empty_answers.empty:
            print(f"WARNING: Found {len(empty_answers)} questions with empty correct answers.")

        # Analyze answer types
        def get_answer_type(ans_str):
            types = []
            ans_lower = ans_str.lower()
            if not ans_str.strip():
                types.append("empty")
                return types

            # Check for multi-part (heuristic)
            if any(sep in ans_lower for sep in [' or ', ' and ', ',']):
                # More specific check for quadratics: two numbers separated by 'or'/'and' or solutions like "x = val1 or x = val2"
                if len(re.findall(r'-?\d+\.?\d*', ans_str)) >= 2 and any(sep in ans_lower for sep in [' or ', ' and ']):
                    types.append("multi-part (likely quadratic)")
                elif len(ans_str.split(',')) > 1 and all(
                        re.fullmatch(r'\s*-?\d+\.?\d*\s*', part) for part in ans_str.split(',')):
                    types.append("multi-part (numeric list)")
                else:
                    types.append("multi-part (generic)")

            if re.search(r'\d/\d', ans_str) or re.search(r'/\d', ans_str) or re.search(r'\d/', ans_str):
                types.append("fractional")
            if 'pi' in ans_lower:
                types.append("contains_pi")

            # Check if it's purely numerical (integer or decimal, possibly with 'x = ')
            # Strip "x =", "y =", etc.
            cleaned_ans_for_num_check = re.sub(r'^[a-zA-Z]\s*=\s*', '', ans_str.strip()).strip()
            if re.fullmatch(r'-?\d+\.?\d*', cleaned_ans_for_num_check):  # Matches numbers like 5, -5, 5.0, -5.2
                types.append("numerical (single)")
            elif re.fullmatch(r'[a-zA-Z]+\s*=\s*-?\d+\.?\d*', ans_str.strip()):  # Matches x = 5
                types.append("numerical (single with var assign)")

            if not types:  # If no specific type matched above
                if re.match(r'^[a-zA-Z\s]+$', ans_str):  # Only letters and spaces
                    types.append("textual")
                else:
                    types.append("other/mixed")
            return types

        df['answer_types'] = df['correct_answer_str'].apply(get_answer_type)
        answer_type_counts = Counter(t for types_list in df['answer_types'] for t in types_list)
        print("\nAnalysis of correct_answer content types (a single answer can have multiple types):")
        for type_name, count in answer_type_counts.most_common():
            print(f"  - {type_name}: {count} instances")

    else:
        print("Column 'correct_answer' not found.")

    # `solution_explanation`
    print("\n-- Column: 'solution_explanation' --")
    if 'solution_explanation' in df.columns:
        df['solution_explanation_str'] = df['solution_explanation'].astype(str)
        df['solution_explanation_len'] = df['solution_explanation_str'].apply(len)
        print(f"Solution explanation length stats:\n{df['solution_explanation_len'].describe().to_string()}")
        empty_explanations = df[df['solution_explanation_len'] < 10]  # Arbitrary short length
        if not empty_explanations.empty:
            print(f"WARNING: Found {len(empty_explanations)} questions with very short (<10 chars) explanations.")

        # Check if correct answer is in explanation (simple check)
        def explanation_contains_answer(row):
            try:
                # Extract numbers from correct answer
                correct_nums = sorted(re.findall(r'-?\d+\.?\d*', str(row['correct_answer_str'])))
                # Extract numbers from explanation
                explanation_nums = sorted(re.findall(r'-?\d+\.?\d*', str(row['solution_explanation_str'])))

                if not correct_nums:  # If correct answer has no numbers (e.g. "pi"), use string search
                    return str(row['correct_answer_str']).lower() in str(row['solution_explanation_str']).lower()

                # Check if all numbers from correct answer are present in explanation numbers
                return all(num in explanation_nums for num in correct_nums)
            except:
                return False  # Error during processing

        df['expl_contains_ans_val'] = df.apply(explanation_contains_answer, axis=1)
        print(
            f"Number of explanations appearing to contain the correct answer's numerical value(s) or text: {df['expl_contains_ans_val'].sum()} / {len(df)}")

    else:
        print("Column 'solution_explanation' not found.")

    # `label`
    print("\n-- Column: 'label' --")
    if 'label' in df.columns:
        print(f"Value counts for 'label' column:\n{df['label'].value_counts().to_string()}")
        if not (df['label'] == 'correct').all():
            print(
                "WARNING: Some questions have labels other than 'correct'. This might be intentional for pre-set examples.")
    else:
        print("Column 'label' not found.")

    # --- III. Cross-Column Consistency & Specific Question Type Checks ---
    print("\n\n--- III. Cross-Column Consistency & Specific Question Type Checks ---")

    if 'topic' in df.columns and 'question_text' in df.columns and 'correct_answer_str' in df.columns:
        # Algebra Questions
        algebra_df = df[df['topic'] == 'Algebra']
        if not algebra_df.empty:
            print("\n-- Algebra Topic Analysis --")
            algebra_df['is_linear_heuristic'] = algebra_df['question_text'].astype(str).apply(
                lambda x: 'x^2' not in x.lower() and 'solve for x:' in x.lower())
            algebra_df['is_quadratic_heuristic'] = algebra_df['question_text'].astype(str).apply(
                lambda x: 'x^2' in x.lower())
            print(f"Number of Algebra questions likely linear (heuristic): {algebra_df['is_linear_heuristic'].sum()}")
            print(
                f"Number of Algebra questions likely quadratic (heuristic): {algebra_df['is_quadratic_heuristic'].sum()}")

            quadratic_answers_multi = algebra_df[algebra_df['is_quadratic_heuristic']]['answer_types'].apply(lambda
                                                                                                                 x: "multi-part (likely quadratic)" in x or "multi-part (generic)" in x or "multi-part (numeric list)" in x).sum()
            print(
                f"Quadratic questions with multi-part answers: {quadratic_answers_multi} / {algebra_df['is_quadratic_heuristic'].sum()}")
        else:
            print("\nNo questions found for topic 'Algebra'.")

        # Geometry Questions
        geometry_df = df[df['topic'] == 'Geometry']
        if not geometry_df.empty:
            print("\n-- Geometry Topic Analysis --")
            geometry_pi_answers = geometry_df['answer_types'].apply(lambda x: "contains_pi" in x).sum()
            print(f"Geometry questions with 'pi' in answer: {geometry_pi_answers} / {len(geometry_df)}")

            # Check for units (simple heuristic based on common units)
            units_pattern = r'\b(cm|m|mm|km|sq|square|cubic|units)\b'
            geometry_df['q_has_units'] = geometry_df['question_text'].astype(str).str.contains(units_pattern,
                                                                                               case=False, regex=True)
            geometry_df['a_has_units'] = geometry_df['correct_answer_str'].astype(str).str.contains(units_pattern,
                                                                                                    case=False,
                                                                                                    regex=True)
            print(
                f"Geometry questions mentioning units in text: {geometry_df['q_has_units'].sum()} / {len(geometry_df)}")
            print(
                f"Geometry questions mentioning units in answer: {geometry_df['a_has_units'].sum()} / {len(geometry_df)}")
        else:
            print("\nNo questions found for topic 'Geometry'.")

        # Check for specific pre-defined questions from generate_synthetic_data.py
        print("\n-- Specific Pre-defined Question Checks --")
        predefined_ids = ["Q_PARTIAL_001", "Q_INCORRECT_001", "Q_ERROR_001"]
        found_predefined = df[df['question_id'].isin(predefined_ids)]
        if not found_predefined.empty:
            print(f"Found {len(found_predefined)} predefined questions for classifier training:")
            print(found_predefined[['question_id', 'topic', 'question_text', 'correct_answer']].to_string())
        else:
            print(f"Did not find any of the specific predefined questions: {predefined_ids}")

    # --- IV. Potential Issues Summary (based on above analysis) ---
    print("\n\n--- IV. Potential Issues Summary ---")
    issues_found = False
    if 'question_id' in df.columns and df['question_id'].nunique() != len(df):
        print("- Duplicate question_ids detected.")
        issues_found = True
    if 'topic' in df.columns and df['topic'].nunique() < 3:  # Arbitrary threshold for variety
        print(f"- Low topic variety (only {df['topic'].nunique()} unique topics). Consider adding more.")
        issues_found = True
    if 'difficulty' in df.columns and len(df['difficulty'].unique()) < len(['easy', 'medium', 'hard']):
        print("- Not all difficulty levels ('easy', 'medium', 'hard') are present or well-represented.")
        issues_found = True
    if 'question_text_len' in df.columns and (df['question_text_len'] == 0).any():
        print("- Some questions have empty text.")
        issues_found = True
    if 'correct_answer_len' in df.columns and (df['correct_answer_len'] == 0).any():
        print("- Some questions have empty correct answers.")
        issues_found = True
    if 'solution_explanation_len' in df.columns and (df['solution_explanation_len'] < 10).any():
        print("- Some solution explanations are very short.")
        issues_found = True

    if 'answer_types' in df.columns:
        if answer_type_counts.get("fractional", 0) < 5:  # Need a few for robust testing/training
            print("- Low number of questions with fractional answers. Needed for robust testing/classifier training.")
            issues_found = True
        if answer_type_counts.get("contains_pi", 0) < 5:
            print("- Low number of questions with 'pi' in answers. Needed for robust testing/classifier training.")
            issues_found = True
        if answer_type_counts.get("multi-part (likely quadratic)", 0) < 5:
            print(
                "- Low number of questions with multi-part answers (e.g., for quadratics). Needed for 'partial' answer classification testing.")
            issues_found = True

    if not issues_found:
        print("No major structural or content variety issues flagged automatically. Manual review still recommended.")

    print("\n--- End of Analysis ---")


if __name__ == "__main__":
    analyze_math_questions_csv(DATA_FILE)