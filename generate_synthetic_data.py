# Adaptive-Tutor/generate_synthetic_data.py
import csv
import random
import os
import pandas as pd
from math import gcd

os.makedirs("data", exist_ok=True)
DATA_FILE = "data/math_questions.csv"
UNIT_CHOICES = ["cm", "m", "mm", "inches", "feet", "units"]


def generate_algebra_question():
    type_choice = random.choice(['integer_sol', 'fraction_sol'])
    a = random.randint(1, 15)
    b = random.randint(-25, 25)
    c = random.randint(-50, 50)
    render_as = "latex"
    if type_choice == 'integer_sol':
        while a == 0 or (c - b) % a != 0:
            a = random.randint(1, 15);
            b = random.randint(-25, 25);
            c = random.randint(-50, 50)
        solution_val = (c - b) // a
        solution = f"x = {solution_val}"
        explanation = f"Given: {a}x + {b} = {c}. Subtract {b} from both sides: {a}x = {c - b}. Then divide by {a}: x = {solution_val}."
        if abs(solution_val) > 15 or abs(a) > 10:
            difficulty = "hard"
        elif abs(solution_val) > 7 or abs(a) > 6:
            difficulty = "medium"
        else:
            difficulty = "easy"
    else:
        max_tries = 10;
        generated_fraction = False
        for _ in range(max_tries):
            a = random.randint(2, 15);
            b = random.randint(-25, 25);
            c = random.randint(-50, 50)
            if a != 0 and (c - b) != 0 and (c - b) % a != 0:
                generated_fraction = True;
                break
        if not generated_fraction:
            return generate_algebra_question()
        numerator = c - b;
        denominator = a
        common_divisor = gcd(numerator, denominator)
        s_numerator = numerator // common_divisor;
        s_denominator = denominator // common_divisor
        if s_denominator < 0: s_numerator = -s_numerator; s_denominator = -s_denominator
        if s_denominator == 1: return generate_algebra_question()
        solution = f"x = {s_numerator}/{s_denominator}"
        explanation = f"Given: {a}x + {b} = {c}. Subtract {b} from both sides: {a}x = {c - b}. So, x = ${numerator}/{denominator}$. Simplified, x = ${s_numerator}/{s_denominator}$."
        if abs(s_denominator) > 7 or abs(s_numerator) > 15:
            difficulty = "hard"
        elif abs(s_denominator) > 4 or abs(s_numerator) > 7:
            difficulty = "medium"
        else:
            difficulty = "easy"
    question = f"Solve for x: {a}x + {b} = {c}"
    return "Algebra", question, solution, explanation, difficulty, render_as


def generate_quadratic_question():
    r1 = random.randint(-8, 8);
    r2 = random.randint(-8, 8)
    while r1 == 0: r1 = random.randint(-8, 8)
    while r2 == 0 or r2 == r1: r2 = random.randint(-8, 8)
    B_coeff = -(r1 + r2);
    C_coeff = r1 * r2
    question = "Solve for x: x^2"
    if B_coeff > 0:
        question += f" + {B_coeff}x"
    elif B_coeff < 0:
        question += f" - {abs(B_coeff)}x"
    if C_coeff > 0:
        question += f" + {C_coeff} = 0"
    elif C_coeff < 0:
        question += f" - {abs(C_coeff)} = 0"
    else:
        question += " = 0"
    solutions = sorted([r1, r2]);
    correct_answer = f"x = {solutions[0]} or x = {solutions[1]}"
    explanation = f"The equation is $x^2 - ({r1 + r2})x + ({r1 * r2}) = 0$. Factor it as $(x - {r1})(x - {r2}) = 0$. So, the solutions are $x = {r1}$ or $x = {r2}$."
    if abs(C_coeff) > 25 or abs(B_coeff) > 10:
        difficulty = "hard"
    elif abs(C_coeff) > 10 or abs(B_coeff) > 6:
        difficulty = "medium"
    else:
        difficulty = "easy"
    return "Algebra", question, correct_answer, explanation, difficulty, "latex"


def generate_arithmetic_question():
    op_type = random.choice(['add', 'subtract', 'multiply', 'divide', 'percentage', 'mixed_easy', 'mixed_hard'])
    n1 = random.randint(1, 100);
    n2 = random.randint(1, 100)
    difficulty = "easy";
    render_as = "markdown"
    question, answer, explanation = "", "", ""  # Initialize
    if op_type == 'add':
        answer_val = n1 + n2;
        answer = str(answer_val);
        explanation = f"{n1} + {n2} = {answer_val}."
        if answer_val > 120: difficulty = "medium";
        if answer_val > 200: difficulty = "hard"
        question = f"What is {n1} + {n2}?"
    elif op_type == 'subtract':
        if n1 < n2 and random.random() < 0.7: n1, n2 = n2, n1
        answer_val = n1 - n2;
        answer = str(answer_val);
        explanation = f"{n1} - {n2} = {answer_val}."
        if abs(answer_val) > 75: difficulty = "medium"
        if abs(answer_val) > 150: difficulty = "hard"
        question = f"What is {n1} - {n2}?"
    elif op_type == 'multiply':
        n1 = random.randint(2, 30);
        n2 = random.randint(2, 20)
        answer_val = n1 * n2;
        answer = str(answer_val);
        explanation = f"{n1} * {n2} = {answer_val}."
        if answer_val > 150: difficulty = "medium"
        if answer_val > 300: difficulty = "hard"
        question = f"What is {n1} * {n2}?"
    elif op_type == 'divide':
        if random.random() < 0.5:
            answer_val = random.randint(2, 25);
            n2 = random.randint(2, 12);
            n1 = answer_val * n2
            answer = str(answer_val);
            explanation = f"{n1} / {n2} = {answer_val}."
            if n1 > 150: difficulty = "medium";
            if n1 > 300: difficulty = "hard"
            question = f"What is {n1} / {n2}?"
        else:
            n1_orig = random.randint(5, 70);
            n2_orig = random.randint(2, 25)
            while n2_orig == 0 or n1_orig % n2_orig == 0: n1_orig = random.randint(5, 70); n2_orig = random.randint(2,
                                                                                                                    25)
            common = gcd(n1_orig, n2_orig);
            s_n1 = n1_orig // common;
            s_n2 = n2_orig // common
            answer = f"{s_n1}/{s_n2}";
            explanation = f"${n1_orig} / {n2_orig}$ simplifies to the fraction ${s_n1}/{s_n2}$."
            if s_n2 > 10 or s_n1 > 20:
                difficulty = "hard"
            elif s_n2 > 5 or s_n1 > 10:
                difficulty = "medium"
            else:
                difficulty = "easy"
            render_as = "latex"
            question = f"What is ${n1_orig} \\div {n2_orig}$? Express as a simplified fraction."
    elif op_type == 'percentage':
        percent = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90]);
        num = random.randint(10, 300)
        if random.random() < 0.6:
            answer_val = (percent * num) // 100; answer = str(answer_val)
        else:
            answer_val = (percent * num) / 100.0; answer = f"{answer_val:.2f}".rstrip('0').rstrip('.')
        explanation = f"{percent}% of {num} is $({percent}/100) \\times {num} = {answer}$."
        if num > 150 or (isinstance(answer_val, float) and answer_val != int(answer_val)): difficulty = "medium"
        if num > 250 or (isinstance(answer_val, float) and abs(answer_val) > 100): difficulty = "hard"
        question = f"What is {percent}% of {num}?"
        render_as = "markdown"
    elif op_type == 'mixed_easy':
        n1 = random.randint(1, 20);
        n2 = random.randint(1, 20);
        n3 = random.randint(1, 10)
        ops = [random.choice(['+', '-']), random.choice(['*', '+'])]
        if ops[0] == '+':
            intermediate = n1 + n2
        else:
            intermediate = n1 - n2
        if ops[1] == '*':
            final_ans = intermediate * n3
        else:
            final_ans = intermediate + n3
        question = f"Calculate: {n1} {ops[0]} {n2} {ops[1]} {n3}"
        answer = str(final_ans)
        explanation = f"Calculating step-by-step: {n1} {ops[0]} {n2} = {intermediate}. Then {intermediate} {ops[1]} {n3} = {final_ans}."
        difficulty = "easy" if abs(final_ans) < 50 else "medium"
    elif op_type == 'mixed_hard':
        t1n1, t1n2 = random.randint(5, 15), random.randint(3, 10)
        t2d = random.randint(2, 5);
        t2n = t2d * random.randint(3, 12)
        question = f"Calculate: $({t1n1} \\times {t1n2}) + ({t2n} \\div {t2d})$"
        answer_val = (t1n1 * t1n2) + (t2n // t2d);
        answer = str(answer_val)
        explanation = f"First, $({t1n1} \\times {t1n2}) = {t1n1 * t1n2}$. Next, $({t2n} \\div {t2d}) = {t2n // t2d}$. Finally, ${t1n1 * t1n2} + {t2n // t2d} = {answer_val}$."
        difficulty = "hard";
        render_as = "latex"
    return "Arithmetic", question, answer, explanation, difficulty, render_as


def generate_geometry_question():
    shape = random.choice(
        ['square', 'rectangle', 'circle_area', 'circle_circumference', 'cube_volume', 'triangle_area'])
    difficulty = "easy";
    unit = random.choice(UNIT_CHOICES)
    render_as = "markdown"  # Default for descriptive geometry questions

    question, answer, explanation = "", "", ""  # Initialize

    if shape == 'square':
        side = random.randint(2, 25);
        calc = random.choice(['area', 'perimeter'])
        question = f"A square has a side length of {side} {unit}. What is its {calc}?"
        if calc == 'area':
            answer_val = side * side;
            answer = f"{answer_val} {unit}^2"
            explanation = f"The area of a square is given by the formula $A = s^2$. So, Area = ${side}^2 = {answer_val}$ {unit}$^2$."
            if side > 12: difficulty = "medium";
            if side > 20: difficulty = "hard"
        else:
            answer_val = 4 * side;
            answer = f"{answer_val} {unit}"
            explanation = f"The perimeter of a square is $P = 4s$. So, Perimeter = $4 \\times {side} = {answer_val}$ {unit}."
            if side > 15: difficulty = "medium";
            if side > 22: difficulty = "hard"
    elif shape == 'rectangle':
        length = random.randint(2, 25);
        width = random.randint(2, length)
        while width == length and random.random() < 0.5: width = random.randint(2, length)
        calc = random.choice(['area', 'perimeter'])
        question = f"A rectangle has a length of {length} {unit} and a width of {width} {unit}. Calculate its {calc}."
        if calc == 'area':
            answer_val = length * width;
            answer = f"{answer_val} {unit}^2"
            explanation = f"The area of a rectangle is $A = l \\times w$. So, Area = ${length} \\times {width} = {answer_val}$ {unit}$^2$."
            if length > 12 or width > 10: difficulty = "medium"
            if length > 20 or width > 15: difficulty = "hard"
        else:
            answer_val = 2 * (length + width);
            answer = f"{answer_val} {unit}"
            explanation = f"The perimeter of a rectangle is $P = 2(l + w)$. So, Perimeter = $2 \\times ({length} + {width}) = {answer_val}$ {unit}."
            if length > 15 or width > 12: difficulty = "medium"
            if length > 22 or width > 18: difficulty = "hard"
    elif shape == 'triangle_area':
        base = random.randint(5, 30);
        height = random.randint(4, 25)
        question = f"Find the area of a triangle with a base of {base} {unit} and a height of {height} {unit}."
        answer_val = (base * height) / 2.0
        answer = f"{answer_val:.1f}".rstrip('0').rstrip('.') + f" {unit}^2"
        explanation = f"The area of a triangle is $A = \\frac{{1}}{{2}}bh$. So, Area = $\\frac{{1}}{{2}} \\times {base} \\times {height} = {answer}$."
        if base > 15 or height > 15: difficulty = "medium"
        if base > 25 or height > 20: difficulty = "hard"
    elif shape == 'circle_area':
        radius = random.randint(2, 15)
        question = f"What is the area of a circle with radius {radius} {unit}? (Use 'pi' in your answer)"
        answer = f"{radius * radius}pi {unit}^2"
        explanation = f"The area of a circle is $A = \pi r^2$. So, Area = $\pi \\times {radius}^2 = {radius * radius}\pi$ {unit}$^2$."
        if radius > 7: difficulty = "medium"
        if radius > 12: difficulty = "hard"
    elif shape == 'circle_circumference':
        radius = random.randint(2, 15)
        question = f"Calculate the circumference of a circle with radius {radius} {unit}. (Use 'pi' in your answer)"
        answer = f"{2 * radius}pi {unit}"
        explanation = f"The circumference of a circle is $C = 2 \pi r$. So, Circumference = $2 \\times \pi \\times {radius} = {2 * radius}\pi$ {unit}."
        if radius > 8: difficulty = "medium"
        if radius > 13: difficulty = "hard"
    elif shape == 'cube_volume':
        side = random.randint(2, 12)
        question = f"What is the volume of a cube with an edge length of {side} {unit}?"
        answer_val = side ** 3;
        answer = f"{answer_val} {unit}^3"
        explanation = f"The volume of a cube is $V = s^3$. So, Volume = ${side}^3 = {answer_val}$ {unit}$^3$."
        if side > 6: difficulty = "medium"
        if side > 10: difficulty = "hard"

    # Override to latex if question text itself contains specific LaTeX commands
    # (explanations can use inline $...$ which markdown handles)
    # Descriptive questions like "A rectangle has..." should remain markdown.
    if any(tex_char in question for tex_char in
           ["\\frac", "\\times", "\\sqrt", "\\sum", "\\int"]) and not question.startswith(
            "A ") and not question.startswith("What is the") and not question.startswith(
            "Calculate the") and not question.startswith("Find the"):
        render_as = "latex"

    return "Geometry", question, answer, explanation, difficulty, render_as


def generate_word_problem():
    problem_type = random.choice(["one_step_easy", "one_step_medium", "two_step_medium", "two_step_hard"])
    item = random.choice(["apples", "bananas", "marbles", "books", "pencils", "cookies"])
    name1 = random.choice(["John", "Sarah", "Mike", "Lisa", "Tom", "Emily"])
    name2 = random.choice(["friend", "brother", "sister", "teacher"])
    render_as = "markdown";
    difficulty = "easy"
    question, answer, explanation = "", "", ""  # Initialize
    if problem_type == "one_step_easy":
        val1 = random.randint(5, 50);
        val2 = random.randint(5, val1 - 1 if val1 > 10 else 20)
        action = random.choice(["sum", "difference"])
        if action == "sum":
            question = f"{name1} has {val1} {item}. They get {val2} more {item}. How many {item} does {name1} have in total?"
            answer = str(val1 + val2);
            explanation = f"{name1} starts with {val1} {item} and adds {val2} more. Total = {val1} + {val2} = {answer} {item}."
        else:
            question = f"{name1} has {val1} {item}. They give {val2} {item} to their {name2}. How many {item} does {name1} have left?"
            answer = str(val1 - val2);
            explanation = f"{name1} starts with {val1} {item} and gives away {val2}. Remaining = {val1} - {val2} = {answer} {item}."
    elif problem_type == "one_step_medium":
        difficulty = "medium"
        if random.random() < 0.5:
            val1 = random.randint(3, 12);
            val2 = random.randint(5, 20)
            question = f"There are {val1} boxes. Each box contains {val2} {item}. How many {item} are there in total?"
            answer = str(val1 * val2);
            explanation = f"To find the total, multiply: {val1} * {val2} = {answer} {item}."
        else:
            total_items = random.randint(20, 100);
            num_people = random.randint(2, 10)
            while total_items % num_people != 0: total_items = random.randint(20, 100); num_people = random.randint(2,
                                                                                                                    10)
            question = f"{name1} has {total_items} {item} to share equally among {num_people} people. How many {item} does each person get?"
            answer = str(total_items // num_people);
            explanation = f"To share equally, divide: {total_items} / {num_people} = {answer} {item} each."
    elif problem_type == "two_step_medium":
        difficulty = "medium"
        val1 = random.randint(10, 30);
        val2 = random.randint(5, 15);
        val3 = random.randint(3, 10)
        question = f"{name1} baked {val1} {item}. They gave {val2} {item} to {name2} and then baked {val3} more. How many {item} does {name1} have now?"
        answer_val = val1 - val2 + val3;
        answer = str(answer_val)
        explanation = f"Step 1: {val1} - {val2} = {val1 - val2}. Step 2: {val1 - val2} + {val3} = {answer_val} {item}."
        if answer_val > 30 or val1 > 25: difficulty = "hard"
    elif problem_type == "two_step_hard":
        difficulty = "hard"
        cost_per_item = random.randint(2, 8);
        num_item1 = random.randint(3, 7);
        num_item2 = random.randint(2, 5)
        item1_name = random.choice(["pens", "notebooks"]);
        item2_name = random.choice(["erasers", "folders"])
        cost_item2_each = cost_per_item + random.randint(1, 3)
        question = f"{name1} bought {num_item1} {item1_name} at ${cost_per_item} each, and {num_item2} {item2_name} at ${cost_item2_each} each. How much did {name1} spend in total?"
        cost1 = num_item1 * cost_per_item;
        cost2 = num_item2 * cost_item2_each
        answer_val = cost1 + cost2;
        answer = f"${answer_val}"
        explanation = f"Cost of {item1_name}: {num_item1} * ${cost_per_item} = ${cost1}. Cost of {item2_name}: {num_item2} * ${cost_item2_each} = ${cost2}. Total = ${cost1} + ${cost2} = ${answer_val}."
    return "Word Problems", question, answer, explanation, difficulty, render_as


def generate_dataset(num_rows=250):
    dataset = []
    question_generators = [
        generate_algebra_question, generate_quadratic_question,
        generate_arithmetic_question, generate_geometry_question,
        generate_word_problem
    ]
    dataset.append(
        {"question_id": "Q_PARTIAL_001", "topic": "Algebra", "question_text": "Solve for x: x^2 - 5x + 6 = 0",
         "correct_answer": "x = 2 or x = 3",
         "solution_explanation": "Factor as (x-2)(x-3)=0. Solutions are x=2 and x=3.", "difficulty": "medium",
         "label": "correct", "render_as": "latex"})
    dataset.append({"question_id": "Q_INCORRECT_001", "topic": "Algebra", "question_text": "Solve for x: 2x + 5 = 1",
                    "correct_answer": "x = -2", "solution_explanation": "2x = 1 - 5 => 2x = -4 => x = -2.",
                    "difficulty": "easy", "label": "correct", "render_as": "latex"})
    dataset.append({"question_id": "Q_ERROR_001", "topic": "Geometry",
                    "question_text": "What is the area of a square with side 5 cm? Include units in your answer.",
                    "correct_answer": "25 cm^2",
                    "solution_explanation": "Area = side^2 = $5^2 = 25$. Units are cm$^2$.", "difficulty": "medium",
                    "label": "correct", "render_as": "markdown"})
    dataset.append({"question_id": "Q_FRACTION_001", "topic": "Algebra", "question_text": "Solve for x: 3x + 4 = 9",
                    "correct_answer": "x = 5/3",
                    "solution_explanation": "3x = 9 - 4 => 3x = 5. Divide by 3: x = $5/3$.", "difficulty": "medium",
                    "label": "correct", "render_as": "latex"})

    generated_difficulties = {"easy": 0, "medium": 0, "hard": 0}
    target_hard = num_rows * 0.30
    target_medium = num_rows * 0.40
    current_q_num = len(dataset)
    while current_q_num < num_rows:
        q_id = f"Q{str(current_q_num + 1).zfill(3)}"
        gen_difficulty_target = None
        if generated_difficulties['hard'] < target_hard:
            gen_difficulty_target = 'hard'
        elif generated_difficulties['medium'] < target_medium:
            gen_difficulty_target = 'medium'

        topic, q_text, c_ans, sol_exp, difficulty, render_mode = None, None, None, None, None, None
        attempts = 0
        # Initialize candidate variables to avoid UnboundLocalError if loop doesn't run
        topic_cand, q_text_cand, c_ans_cand, sol_exp_cand, difficulty_cand, render_mode_cand = "", "", "", "", "easy", "markdown"

        while attempts < 10:
            chosen_generator = random.choice(question_generators)
            topic_cand, q_text_cand, c_ans_cand, sol_exp_cand, difficulty_cand, render_mode_cand = chosen_generator()
            if gen_difficulty_target is None or difficulty_cand == gen_difficulty_target:
                topic, q_text, c_ans, sol_exp, difficulty, render_mode = topic_cand, q_text_cand, c_ans_cand, sol_exp_cand, difficulty_cand, render_mode_cand
                break
            attempts += 1
        else:  # If loop finishes without break (couldn't get target difficulty)
            topic, q_text, c_ans, sol_exp, difficulty, render_mode = topic_cand, q_text_cand, c_ans_cand, sol_exp_cand, difficulty_cand, render_mode_cand  # Use last generated

        generated_difficulties[difficulty] += 1
        dataset.append({
            "question_id": q_id, "topic": topic, "question_text": q_text,
            "correct_answer": c_ans, "label": "correct",
            "solution_explanation": sol_exp, "difficulty": difficulty, "render_as": render_mode
        })
        current_q_num += 1
    return dataset


if __name__ == "__main__":
    num_questions_to_generate = 250
    questions_data = generate_dataset(num_questions_to_generate)
    fieldnames = ["question_id", "topic", "question_text", "correct_answer", "label", "solution_explanation",
                  "difficulty", "render_as"]
    with open(DATA_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(questions_data)
    print(f"Successfully generated {len(questions_data)} questions in '{DATA_FILE}'")
    if questions_data:
        df_temp = pd.DataFrame(questions_data)
        print("\nGenerated Difficulty Distribution:")
        print(df_temp['difficulty'].value_counts(normalize=True).apply(lambda x: f"{x:.0%}").to_string())
        print("\nGenerated Render_as Distribution:")
        print(df_temp['render_as'].value_counts().to_string())

