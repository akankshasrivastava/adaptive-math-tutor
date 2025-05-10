# Adaptive-Tutor/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random
import joblib
import re
import numpy as np
import os

# from math import gcd # Not directly used in this version of featurizer

os.makedirs("data", exist_ok=True)
DATA_FILE = "data/math_questions.csv"
MODEL_FILE = "data/answer_classifier.joblib"


def normalize_text(text):
    """Lowercase, remove punc, strip spaces, handle common math terms, evaluate 'pi'."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()

    # Attempt to pre-evaluate expressions involving pi and coefficients
    # This aims to convert "25pi", "25 * pi", "pi * 25", "pi / 2" into a single float string

    # Order of replacements can matter.
    # Handle multiplication/division with pi first.
    # Example: "25 * pi" or "25pi"
    text = re.sub(r'(-?\d+\.?\d*)\s*\*?\s*pi', lambda m: str(float(m.group(1)) * np.pi), text)
    # Example: "pi * 25"
    text = re.sub(r'pi\s*\*?\s*(-?\d+\.?\d*)', lambda m: str(np.pi * float(m.group(1))), text)
    # Example: "pi / 2"
    text = re.sub(r'pi\s*/\s*(-?\d+\.?\d*)',
                  lambda m: str(np.pi / float(m.group(1))) if float(m.group(1)) != 0 else " <pi_div_zero> ",
                  text)
    # Replace standalone "pi" if it wasn't part of an expression above
    text = text.replace('pi', str(np.pi))

    text = text.replace('or', ',').replace('and', ',')
    # Keep basic math chars, letters, numbers, dot, slash for fractions, equals, comma, parentheses
    text = re.sub(r'[^\w\s\.\-/=,\(\)\*\+]', '', text)  # Added * and + just in case they appear after pi eval
    text = text.strip()
    return text


def extract_numbers(text_input):
    if not isinstance(text_input, str):
        return []

    processed_text = text_input
    # Regex for "number / number" allowing for decimals in num/den
    fraction_pattern = r'(-?\s*\d+\.?\d*)\s*/\s*(-?\s*\d+\.?\d*)'

    def replace_fraction_match(match_obj):
        try:
            num = float(match_obj.group(1).replace(" ", ""))
            den = float(match_obj.group(2).replace(" ", ""))
            if den == 0: return " <fraction_div_zero> "
            return f" {num / den} "
        except ValueError:
            return " <fraction_error> "

    # Iteratively replace fractions.
    for _ in range(5):
        new_text = re.sub(fraction_pattern, replace_fraction_match, processed_text, count=1)
        if new_text == processed_text:
            break
        processed_text = new_text

    # Extract numbers (integers or floats, including those from evaluated fractions/pi)
    numbers = re.findall(r'-?\d+\.?\d*', processed_text)
    extracted_floats = []
    for n_str in numbers:
        try:
            extracted_floats.append(float(n_str))
        except ValueError:
            pass  # Ignore if something non-numeric was caught by regex somehow
    return sorted(extracted_floats)


def compare_answers(user_answer_norm, correct_answer_norm, tol=1e-3):
    features = {}
    features['exact_match'] = 1 if user_answer_norm == correct_answer_norm else 0

    user_numbers = extract_numbers(user_answer_norm)
    correct_numbers = extract_numbers(correct_answer_norm)

    features['same_number_count'] = 1 if len(user_numbers) == len(correct_numbers) else 0

    all_close = False
    some_close_count = 0
    is_subset_close_flag = True  # Assume true until proven false

    if correct_numbers:
        if len(user_numbers) == len(correct_numbers):
            temp_all_close = True
            sorted_user_nums = sorted(user_numbers)
            sorted_correct_nums = sorted(correct_numbers)
            for un, cn in zip(sorted_user_nums, sorted_correct_nums):
                if not np.isclose(un, cn, atol=tol, rtol=tol):  # Added rtol for relative tolerance
                    temp_all_close = False
                    break
            if temp_all_close:
                all_close = True

        # For 'some_close' and 'is_subset_close', we match each user number to a correct number
        # This avoids issues if the order is different but counts are the same.
        # A more robust approach for unordered sets of numbers:

        # Create frequency maps for robust multi-set comparison with tolerance
        user_num_counts = {}
        for un in user_numbers:
            matched_cn = None
            for cn_key in user_num_counts:  # Check if it matches an existing "bucket"
                if np.isclose(un, cn_key, atol=tol, rtol=tol):
                    matched_cn = cn_key
                    break
            if matched_cn is not None:
                user_num_counts[matched_cn] += 1
            else:
                user_num_counts[un] = 1

        correct_num_counts = {}
        for cn in correct_numbers:
            matched_cn_key = None
            for cn_key in correct_num_counts:
                if np.isclose(cn, cn_key, atol=tol, rtol=tol):
                    matched_cn_key = cn_key
                    break
            if matched_cn_key is not None:
                correct_num_counts[matched_cn_key] += 1
            else:
                correct_num_counts[cn] = 1

        # Calculate some_close_count and is_subset_close_flag based on these frequency maps
        temp_correct_counts_for_some = correct_num_counts.copy()
        for un_key, un_c in user_num_counts.items():
            found_match_for_un_key = False
            for cn_key, cn_c in temp_correct_counts_for_some.items():
                if np.isclose(un_key, cn_key, atol=tol, rtol=tol):
                    can_take = min(un_c, cn_c)
                    some_close_count += can_take
                    temp_correct_counts_for_some[cn_key] -= can_take  # Reduce available count
                    found_match_for_un_key = True  # Part of this user number found a match
                    break  # Move to next user number key after finding a bucket for this one

        # Check for subset property
        temp_correct_counts_for_subset = correct_num_counts.copy()
        if not user_numbers:  # Empty user_numbers is a subset
            is_subset_close_flag = True
        elif not correct_numbers and user_numbers:  # User has numbers, correct is empty
            is_subset_close_flag = False
        else:
            for un_key, un_c in user_num_counts.items():
                found_bucket_for_un_key = False
                for cn_key, cn_c in temp_correct_counts_for_subset.items():
                    if np.isclose(un_key, cn_key, atol=tol, rtol=tol):
                        if cn_c >= un_c:  # Correct bucket has enough instances
                            temp_correct_counts_for_subset[cn_key] -= un_c
                            found_bucket_for_un_key = True
                            break
                        else:  # Not enough in this specific bucket
                            is_subset_close_flag = False;
                            break
                if not found_bucket_for_un_key:
                    is_subset_close_flag = False;
                    break
                if not is_subset_close_flag: break

        features['all_correct_numbers_present_close'] = 1 if all_close else 0
        features['some_correct_numbers_present_close'] = 1 if some_close_count > 0 else 0
        features['ratio_correct_numbers_close'] = some_close_count / len(correct_numbers) if correct_numbers else (
            1.0 if not user_numbers else 0.0)
        features['is_subset_of_correct_close'] = 1 if is_subset_close_flag else 0

    else:  # No numbers in correct answer
        features['all_correct_numbers_present_close'] = 1 if not user_numbers else 0
        features['some_correct_numbers_present_close'] = 1 if not user_numbers else 0
        features['ratio_correct_numbers_close'] = 1.0 if not user_numbers else 0.0
        features['is_subset_of_correct_close'] = 1 if not user_numbers else 0

    len_user = len(user_answer_norm)
    len_correct = len(correct_answer_norm)
    features['len_diff_ratio'] = abs(len_user - len_correct) / max(1, len_correct, len_user)
    features['has_equals_sign'] = 1 if '=' in user_answer_norm else 0
    cleaned_user_ans_for_x = user_answer_norm.replace(" ", "")
    features['has_x_equals'] = 1 if 'x=' in cleaned_user_ans_for_x else 0
    return features


def generate_training_samples(df_questions):
    training_data = []
    for _, row in df_questions.iterrows():
        q_text = row['question_text']
        c_ans_text_orig = str(row['correct_answer'])
        topic = row['topic']

        # Using the improved normalize and extract for generating sensible variations
        c_ans_norm_for_variations = normalize_text(c_ans_text_orig)
        # correct_numbers_from_ans = extract_numbers(c_ans_norm_for_variations) # Not used directly below for now

        correct_variations = {c_ans_text_orig}  # Use a set to store variations to avoid duplicates

        if "or" in c_ans_text_orig.lower():
            parts = [p.strip() for p in re.split(r'\s+or\s+', c_ans_text_orig, flags=re.IGNORECASE)]
            if len(parts) == 2:
                correct_variations.add(f"{parts[1]} or {parts[0]}")

        # Handle 'pi' variations
        if 'pi' in c_ans_text_orig.lower():
            correct_variations.add(c_ans_text_orig.lower().replace(" ", ""))
            correct_variations.add(c_ans_text_orig.replace("pi", "PI"))

            # Generate numeric approximation based on *normalized* (evaluated pi) correct answer
            # This ensures we are approximating the actual value
            eval_pi_correct_ans_str = normalize_text(c_ans_text_orig)  # e.g. "25pi" -> "78.539..."
            eval_pi_correct_nums = extract_numbers(eval_pi_correct_ans_str)

            if eval_pi_correct_nums:
                # Add variations of the evaluated number(s)
                # Example: correct is "25pi", eval_pi_correct_nums = [78.539...]. User might type "78.54"
                num_to_approx = eval_pi_correct_nums[0]  # Assuming one main number for pi expressions
                correct_variations.add(f"{num_to_approx:.2f}")
                correct_variations.add(f"{num_to_approx:.3f}")

                # If original had units, try to append them to approximations
                unit_match = re.search(r'([a-zA-Z\^0-9]+)\s*$', c_ans_text_orig)  # Simple unit snatch from end
                if unit_match and not unit_match.group(1).isdigit():  # Ensure it's not just part of a number
                    unit = unit_match.group(1)
                    correct_variations.add(f"{num_to_approx:.2f} {unit}")
                    correct_variations.add(f"{num_to_approx:.3f} {unit}")

        # Handle fraction variations "A/B" vs "X.Y"
        if "/" in c_ans_text_orig:
            # Normalized version will already have the fraction evaluated if extract_numbers works
            eval_frac_correct_ans_str = normalize_text(c_ans_text_orig)
            eval_frac_correct_nums = extract_numbers(eval_frac_correct_ans_str)

            if eval_frac_correct_nums:
                # Add the evaluated decimal form as a correct answer
                # If c_ans_text_orig was "x = 1/2", eval_frac_correct_nums = [0.5]
                # User might type "0.5" or "x = 0.5"
                val_str_2dp = f"{eval_frac_correct_nums[0]:.2f}"
                val_str_3dp = f"{eval_frac_correct_nums[0]:.3f}"
                correct_variations.add(val_str_2dp)
                correct_variations.add(val_str_3dp)
                if "x =" in c_ans_text_orig.lower().split('/')[0]:  # If original was like "x = A/B"
                    correct_variations.add(f"x = {val_str_2dp}")
                    correct_variations.add(f"x = {val_str_3dp}")

        for ans_var in correct_variations:
            training_data.append({
                "question_text": q_text, "user_answer": ans_var,
                "correct_answer_for_q": c_ans_text_orig, "label": "correct"
            })

        # Partial/Incorrect logic (can be refined further based on model performance)
        # This is kept brief as the main focus was pi/fractions
        current_correct_numbers = extract_numbers(normalize_text(c_ans_text_orig))
        if len(current_correct_numbers) > 1 and ("or" in c_ans_text_orig.lower() or "," in c_ans_text_orig):
            training_data.append({"question_text": q_text, "user_answer": f"x = {current_correct_numbers[0]}",
                                  "correct_answer_for_q": c_ans_text_orig, "label": "partial"})
            training_data.append({"question_text": q_text, "user_answer": str(current_correct_numbers[0]),
                                  "correct_answer_for_q": c_ans_text_orig, "label": "partial"})

        if current_correct_numbers:
            incorrect_val = current_correct_numbers[0] + random.uniform(0.5, 2.5) * random.choice([-1, 1])
            if np.isclose(incorrect_val, current_correct_numbers[0]): incorrect_val += 1
            training_data.append(
                {"question_text": q_text, "user_answer": str(incorrect_val), "correct_answer_for_q": c_ans_text_orig,
                 "label": "incorrect"})

        training_data.append({"question_text": q_text, "user_answer": "idk", "correct_answer_for_q": c_ans_text_orig,
                              "label": "incorrect"})
        if topic == "Geometry" and any(u in c_ans_text_orig for u in ["cm", "m", "feet", "inches"]):
            no_unit_ans = re.sub(r'\s*[a-zA-Z\^²³]+$', '', c_ans_text_orig).strip()  # Basic unit strip
            if no_unit_ans and no_unit_ans != c_ans_text_orig:
                training_data.append(
                    {"question_text": q_text, "user_answer": no_unit_ans, "correct_answer_for_q": c_ans_text_orig,
                     "label": "partial"})

    return pd.DataFrame(training_data)


def featurize_for_classifier(df_row_series):
    user_ans_norm = normalize_text(df_row_series['user_answer'])
    correct_ans_norm = normalize_text(df_row_series['correct_answer_for_q'])
    features_dict = compare_answers(user_ans_norm, correct_ans_norm)
    feature_values = [
        features_dict['exact_match'],
        features_dict['same_number_count'],
        features_dict['all_correct_numbers_present_close'],
        features_dict['some_correct_numbers_present_close'],
        features_dict['ratio_correct_numbers_close'],
        features_dict['is_subset_of_correct_close'],
        features_dict['len_diff_ratio'],
        features_dict['has_equals_sign'],
        features_dict['has_x_equals']
    ]
    return feature_values


if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found. Please run `generate_synthetic_data.py` first.")
        exit()
    print("Loading dataset...")
    df_questions = pd.read_csv(DATA_FILE)
    df_questions.dropna(subset=['correct_answer', 'question_text', 'topic'], inplace=True)
    print("Generating training samples (synthetic user answers)...")
    df_train_raw = generate_training_samples(df_questions)
    if df_train_raw.empty:
        print("No training samples generated. Exiting.")
        exit()
    print(f"Generated {len(df_train_raw)} training samples.")
    print("Sample training data (first 5):")
    print(df_train_raw.head().to_string())
    # print("\nSample training data (last 5):") # Optional: print tail
    # print(df_train_raw.tail().to_string())
    print("\nFeaturizing data...")
    X_features_list = df_train_raw.apply(featurize_for_classifier, axis=1).tolist()
    X = np.array(X_features_list)
    y = df_train_raw['label']
    if X.shape[0] == 0 or X.shape[0] != len(y):
        print(f"Error: Feature matrix X (shape: {X.shape}) or target y (len: {len(y)}) is empty or mismatched.")
        exit()
    print(f"Feature matrix X shape: {X.shape}")
    print(f"Target vector y shape: {y.shape}")
    # print("Sample features (first 5 rows):") # Can be very verbose
    # print(X[:5])
    print("\nTarget labels distribution:")
    print(y.value_counts().to_string())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print("\nTraining classification model (Logistic Regression)...")
    model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42, max_iter=2000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")
    try:
        from sklearn.metrics import classification_report

        y_pred_test = model.predict(X_test)
        print("\nClassification Report on Test Set:")
        print(classification_report(y_test, y_pred_test, zero_division=0))  # Added zero_division
    except ImportError:
        print("sklearn.metrics.classification_report not available. Skipping detailed report.")
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to '{MODEL_FILE}'")
    print("Ensure feature engineering logic is consistent at prediction time.")