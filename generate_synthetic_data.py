# Adaptive-Tutor/generate_synthetic_data.py
import csv
import pandas as pd
import random
import os
from math import gcd  # For simplifying fractions

# Ensure 'data' directory exists
os.makedirs("data", exist_ok=True)
DATA_FILE = "data/math_questions.csv"
UNIT_CHOICES = ["cm", "m", "mm", "inches", "feet", "units"]  # Added "units" as generic


def generate_algebra_question():
    """Generates an algebra question, can be linear with integer or fractional solutions."""
    # Determine if the question should have an integer or fractional solution
    # Increase probability of harder questions (often fractional or larger numbers)
    if random.random() < 0.4:  # 40% chance for fractional/more complex
        type_choice = 'fraction_sol'
    else:
        type_choice = 'integer_sol'

    a = random.randint(1, 15)  # Allow slightly larger coefficients
    b = random.randint(-25, 25)  # Allow negative b
    c = random.randint(-50, 50)  # Allow negative c

    render_as = "latex"  # Algebra questions are generally good for LaTeX

    if type_choice == 'integer_sol':
        # Ensure solution (c-b)/a is an integer
        while a == 0 or (c - b) % a != 0:
            a = random.randint(1, 15)
            b = random.randint(-25, 25)
            c = random.randint(-50, 50)

        solution_val = (c - b) // a  # Use integer division
        solution = f"x = {solution_val}"
        explanation = f"Given: {a}x + {b} = {c}. Subtract {b} from both sides: {a}x = {c - b}. Then divide by {a}: x = {solution_val}."

        # Difficulty for integer solutions
        if abs(solution_val) > 15 or abs(a) > 10 or abs(b) > 20 or abs(c) > 40:
            difficulty = "hard"
        elif abs(solution_val) > 7 or abs(a) > 6 or abs(b) > 10 or abs(c) > 25:
            difficulty = "medium"
        else:
            difficulty = "easy"

    else:  # fraction_sol
        # Ensure a is not 0, and try to make it a non-integer solution
        max_tries = 10
        for _ in range(max_tries):
            a = random.randint(2, 15)  # Denominator usually > 1 for fractions
            b = random.randint(-25, 25)
            c = random.randint(-50, 50)
            if a != 0 and (c - b) != 0 and (c - b) % a != 0:  # Check for non-zero and non-integer result
                break
        else:  # Fallback to integer if couldn't generate fraction easily
            return generate_algebra_question()  # Recurse might be too much, let's make it integer instead
            # Fallback strategy: generate an integer solution question instead
            # while a == 0 or (c - b) % a != 0:
            #     a = random.randint(1, 15); b = random.randint(-25, 25); c = random.randint(-50, 50)
            # solution_val = (c - b) // a
            # solution = f"x = {solution_val}"
            # explanation = f"Given: {a}x + {b} = {c}. Subtract {b} from both sides: {a}x = {c - b}. Then divide by {a}: x = {solution_val}."
            # difficulty = "medium" # Fractions are generally at least medium
            # return "Algebra", f"Solve for x: {a}x + {b} = {c}", solution, explanation, difficulty, render_as

        numerator = c - b
        denominator = a

        common_divisor = gcd(numerator, denominator)
        s_numerator = numerator // common_divisor
        s_denominator = denominator // common_divisor

        # Ensure the sign is handled cleanly (e.g., -1/2, not 1/-2)
        if s_denominator < 0:
            s_numerator = -s_numerator
            s_denominator = -s_denominator

        # If it simplified to an integer (e.g., 4/2 = 2), re-generate to ensure a fraction.
        if s_denominator == 1:
            # Call again, hoping for a fractional result next time
            # This recursive call should ideally have a depth limit or alternative strategy
            # For simplicity here, we call it once. If still integer, it's okay.
            return generate_algebra_question()

        solution_val = numerator / denominator  # For explanation
        solution = f"x = {s_numerator}/{s_denominator}"
        explanation = (f"Given: {a}x + {b} = {c}. Subtract {b} from both sides: {a}x = {c - b}. "
                       f"So, x = {numerator}/{denominator}. Simplified, x = {s_numerator}/{s_denominator}.")

        # Difficulty for fractional solutions
        if abs(s_denominator) > 7 or abs(s_numerator) > 15 or abs(a) > 10:
            difficulty = "hard"
        elif abs(s_denominator) > 4 or abs(s_numerator) > 7:
            difficulty = "medium"
        else:  # Simpler fractions
            difficulty = "easy"

    question = f"Solve for x: {a}x + {b} = {c}"
    return "Algebra", question, solution, explanation, difficulty, render_as


def generate_quadratic_question():
    """Generates a quadratic equation question that can be factored."""
    # (x - r1)(x - r2) = x^2 - (r1+r2)x + r1*r2 = 0
    r1 = random.randint(-8, 8)  # Expanded range for roots
    while r1 == 0:
        r1 = random.randint(-8, 8)
    r2 = random.randint(-8, 8)
    while r2 == 0 or r2 == r1:  # Ensure distinct, non-zero roots
        r2 = random.randint(-8, 8)

    # Coefficients
    # x^2 + Bx + C = 0, where B = -(r1+r2), C = r1*r2
    B_coeff = -(r1 + r2)
    C_coeff = r1 * r2

    # Construct question string
    question = "Solve for x: x^2"
    if B_coeff > 0:
        question += f" + {B_coeff}x"
    elif B_coeff < 0:
        question += f" - {abs(B_coeff)}x"
    # If B_coeff is 0, no x term is added.

    if C_coeff > 0:
        question += f" + {C_coeff} = 0"
    elif C_coeff < 0:
        question += f" - {abs(C_coeff)} = 0"
    else:  # C_coeff == 0, e.g., x^2 + Bx = 0
        question += " = 0"

    solutions = sorted([r1, r2])
    correct_answer = f"x = {solutions[0]} or x = {solutions[1]}"
    explanation = (f"The equation is x^2 - ({r1 + r2})x + ({r1 * r2}) = 0. "
                   f"Factor it as (x - {r1})(x - {r2}) = 0. So, the solutions are x = {r1} or x = {r2}.")

    # Difficulty based on magnitude of coefficients or roots
    if abs(C_coeff) > 25 or abs(B_coeff) > 10 or max(abs(r1), abs(r2)) > 6:
        difficulty = "hard"
    elif abs(C_coeff) > 10 or abs(B_coeff) > 6 or max(abs(r1), abs(r2)) > 4:
        difficulty = "medium"
    else:
        difficulty = "easy"

    render_as = "latex"
    return "Algebra", question, correct_answer, explanation, difficulty, render_as


def generate_arithmetic_question():
    """Generates an arithmetic question, including potential for fractional results from division."""
    op_type = random.choice(['add', 'subtract', 'multiply', 'divide', 'percentage', 'mixed_easy', 'mixed_hard'])
    n1 = random.randint(1, 100)
    n2 = random.randint(1, 100)
    difficulty = "easy"  # Default
    render_as = "markdown"  # Basic arithmetic often doesn't need LaTeX

    if op_type == 'add':
        question = f"What is {n1} + {n2}?"
        answer_val = n1 + n2
        answer = str(answer_val)
        explanation = f"{n1} + {n2} = {answer_val}."
        if n1 > 75 or n2 > 75 or answer_val > 120: difficulty = "medium"
        if answer_val > 200: difficulty = "hard"

    elif op_type == 'subtract':
        if n1 < n2 and random.random() < 0.7:  # 70% chance to ensure n1 >= n2 for easier questions
            n1, n2 = n2, n1  # Ensure positive result more often for easy/medium
        question = f"What is {n1} - {n2}?"
        answer_val = n1 - n2
        answer = str(answer_val)
        explanation = f"{n1} - {n2} = {answer_val}."
        if abs(answer_val) > 75 or n1 > 100: difficulty = "medium"
        if abs(answer_val) > 150: difficulty = "hard"


    elif op_type == 'multiply':
        n1 = random.randint(2, 30)  # Wider range for multiplication
        n2 = random.randint(2, 20)
        question = f"What is {n1} * {n2}?"
        answer_val = n1 * n2
        answer = str(answer_val)
        explanation = f"{n1} * {n2} = {answer_val}."
        if n1 > 15 or n2 > 12 or answer_val > 150: difficulty = "medium"
        if answer_val > 300: difficulty = "hard"

    elif op_type == 'divide':
        # Choose if integer or fractional result
        if random.random() < 0.5:  # 50% chance for clean integer division
            answer_val = random.randint(2, 25)
            n2 = random.randint(2, 12)
            n1 = answer_val * n2
            question = f"What is {n1} / {n2}?"
            answer = str(answer_val)
            explanation = f"{n1} / {n2} = {answer_val}."
            if n1 > 150 or answer_val > 20: difficulty = "medium"
            if n1 > 300: difficulty = "hard"
        else:  # Fractional result
            n1_orig = random.randint(5, 70)
            n2_orig = random.randint(2, 25)
            # Ensure not cleanly divisible and n2 is not zero
            while n2_orig == 0 or n1_orig % n2_orig == 0:
                n1_orig = random.randint(5, 70)
                n2_orig = random.randint(2, 25)

            common = gcd(n1_orig, n2_orig)
            s_n1 = n1_orig // common
            s_n2 = n2_orig // common

            question = f"What is {n1_orig} / {n2_orig}? Express as a simplified fraction."
            answer = f"{s_n1}/{s_n2}"  # s_n2 should not be 1 due to the while loop
            explanation = f"{n1_orig} / {n2_orig} simplifies to the fraction {s_n1}/{s_n2}."
            if s_n2 > 10 or s_n1 > 20:
                difficulty = "hard"
            elif s_n2 > 5 or s_n1 > 10:
                difficulty = "medium"
            else:
                difficulty = "easy"  # Simple fractions
            render_as = "latex"  # Fractions look better in LaTeX

    elif op_type == 'percentage':
        percent = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90])
        num = random.randint(10, 300)

        # Allow non-integer results for percentages for medium/hard
        if random.random() < 0.6:  # 60% chance for integer result
            while (percent * num) % 100 != 0:
                num = random.randint(10, 300)
            answer_val = (percent * num) // 100
            answer = str(answer_val)
        else:  # Decimal result for percentage
            answer_val = (percent * num) / 100.0
            answer = f"{answer_val:.2f}".rstrip('0').rstrip(
                '.')  # Format to 2 decimal places, remove trailing .0 or .00

        question = f"What is {percent}% of {num}?"
        explanation = f"{percent}% of {num} is ({percent}/100) * {num} = {answer}."
        if num > 150 or percent > 50 or (isinstance(answer_val, float) and answer_val != int(answer_val)):
            difficulty = "medium"
        if num > 250 or (isinstance(answer_val, float) and abs(answer_val) > 100):
            difficulty = "hard"

    elif op_type == 'mixed_easy':
        n3 = random.randint(1, 10)
        op1 = random.choice(['+', '-'])
        op2 = random.choice(['*', '+'])
        question = f"Calculate: {n1} {op1} {n2} {op2} {n3}"
        if op1 == '+' and op2 == '*':
            answer_val = n1 + (n2 * n3)
        elif op1 == '-' and op2 == '*':
            answer_val = n1 - (n2 * n3)
        elif op1 == '+' and op2 == '+':
            answer_val = n1 + n2 + n3
        elif op1 == '-' and op2 == '+':
            answer_val = n1 - n2 + n3
        # Note: Does not handle order of operations fully for all combos, simplified for 'easy'
        answer = str(answer_val)
        explanation = f"Following order of operations for {question}, the result is {answer}."
        difficulty = "easy"  # Could be medium depending on numbers

    elif op_type == 'mixed_hard':
        n3 = random.randint(2, 15)
        n4 = random.randint(2, 10)
        ops = random.sample(['+', '-', '*', '/'], 2)
        # Example: (n1 op1 n2) op2 n3, or n1 op1 (n2 op2 n3)
        # For simplicity, let's do a specific structure with parentheses for clarity
        # (a * b) + (c / d) where c/d is integer
        term1_n1 = random.randint(5, 15)
        term1_n2 = random.randint(3, 10)
        term2_d = random.randint(2, 5)
        term2_n = term2_d * random.randint(3, 12)  # ensure integer division

        question = f"Calculate: ({term1_n1} * {term1_n2}) + ({term2_n} / {term2_d})"
        answer_val = (term1_n1 * term1_n2) + (term2_n // term2_d)
        answer = str(answer_val)
        explanation = f"First, calculate inside parentheses: ({term1_n1} * {term1_n2}) = {term1_n1 * term1_n2}, and ({term2_n} / {term2_d}) = {term2_n // term2_d}. Then add the results: {term1_n1 * term1_n2} + {term2_n // term2_d} = {answer_val}."
        difficulty = "hard"
        render_as = "latex"

    return "Arithmetic", question, answer, explanation, difficulty, render_as


def generate_geometry_question():
    """Generates a geometry question with units."""
    shape = random.choice(
        ['square', 'rectangle', 'circle_area', 'circle_circumference', 'cube_volume', 'triangle_area'])
    difficulty = "easy"  # Default
    unit = random.choice(UNIT_CHOICES)
    render_as = "latex"  # Geometry often uses symbols or formulae

    if shape == 'square':
        side = random.randint(2, 25)
        calc = random.choice(['area', 'perimeter'])
        if calc == 'area':
            question = f"What is the area of a square with side length {side} {unit}?"
            answer_val = side * side
            answer = f"{answer_val} {unit}^2"
            explanation = f"The area of a square is given by the formula $A = s^2$. So, Area = ${side}^2 = {answer_val}$ {unit}$^2$."
            if side > 12: difficulty = "medium"
            if side > 20: difficulty = "hard"
        else:  # perimeter
            question = f"What is the perimeter of a square with side length {side} {unit}?"
            answer_val = 4 * side
            answer = f"{answer_val} {unit}"
            explanation = f"The perimeter of a square is $P = 4s$. So, Perimeter = $4 \\times {side} = {answer_val}$ {unit}."
            if side > 15: difficulty = "medium"
            if side > 22: difficulty = "hard"

    elif shape == 'rectangle':
        length = random.randint(2, 25)
        width = random.randint(2, length)  # width <= length
        while width == length and random.random() < 0.5:  # try to make it not a square sometimes
            width = random.randint(2, length)

        calc = random.choice(['area', 'perimeter'])
        if calc == 'area':
            question = f"A rectangle has a length of {length} {unit} and a width of {width} {unit}. What is its area?"
            answer_val = length * width
            answer = f"{answer_val} {unit}^2"
            explanation = f"The area of a rectangle is $A = l \\times w$. So, Area = ${length} \\times {width} = {answer_val}$ {unit}$^2$."
            if length > 12 or width > 10: difficulty = "medium"
            if length > 20 or width > 15: difficulty = "hard"
        else:  # perimeter
            question = f"A rectangle has a length of {length} {unit} and a width of {width} {unit}. What is its perimeter?"
            answer_val = 2 * (length + width)
            answer = f"{answer_val} {unit}"
            explanation = f"The perimeter of a rectangle is $P = 2(l + w)$. So, Perimeter = $2 \\times ({length} + {width}) = {answer_val}$ {unit}."
            if length > 15 or width > 12: difficulty = "medium"
            if length > 22 or width > 18: difficulty = "hard"

    elif shape == 'triangle_area':
        base = random.randint(5, 30)
        height = random.randint(4, 25)
        question = f"Calculate the area of a triangle with a base of {base} {unit} and a height of {height} {unit}."
        # Ensure area can be integer or .5
        if (base * height) % 2 == 0:
            answer_val = (base * height) // 2
            answer = f"{answer_val} {unit}^2"
        else:
            answer_val = (base * height) / 2.0
            answer = f"{answer_val:.1f} {unit}^2"  # e.g. 12.5 unit^2

        explanation = f"The area of a triangle is $A = \\frac{{1}}{{2}} \\times base \\times height$. So, Area = $\\frac{{1}}{{2}} \\times {base} \\times {height} = {answer}$."
        if base > 15 or height > 15: difficulty = "medium"
        if base > 25 or height > 20: difficulty = "hard"


    elif shape == 'circle_area':
        radius = random.randint(2, 15)
        question = f"What is the area of a circle with radius {radius} {unit}? (Use $\pi$ or 'pi' in your answer)"
        answer = f"{radius * radius}pi {unit}^2"
        explanation = f"The area of a circle is $A = \pi r^2$. So, Area = $\pi \\times {radius}^2 = {radius * radius}\pi$ {unit}$^2$."
        if radius > 7: difficulty = "medium"
        if radius > 12: difficulty = "hard"

    elif shape == 'circle_circumference':
        radius = random.randint(2, 15)
        question = f"What is the circumference of a circle with radius {radius} {unit}? (Use $\pi$ or 'pi' in your answer)"
        answer = f"{2 * radius}pi {unit}"
        explanation = f"The circumference of a circle is $C = 2 \pi r$. So, Circumference = $2 \\times \pi \\times {radius} = {2 * radius}\pi$ {unit}."
        if radius > 8: difficulty = "medium"
        if radius > 13: difficulty = "hard"

    elif shape == 'cube_volume':
        side = random.randint(2, 12)
        question = f"Calculate the volume of a cube with edge length {side} {unit}."
        answer_val = side ** 3
        answer = f"{answer_val} {unit}^3"
        explanation = f"The volume of a cube is $V = s^3$. So, Volume = ${side}^3 = {answer_val}$ {unit}$^3$."
        if side > 6: difficulty = "medium"
        if side > 10: difficulty = "hard"

    return "Geometry", question, answer, explanation, difficulty, render_as


def generate_word_problem():
    """Generates a simple word problem."""
    problem_type = random.choice(["one_step_easy", "one_step_medium", "two_step_medium", "two_step_hard"])
    item = random.choice(["apples", "bananas", "marbles", "books", "pencils", "cookies"])
    name1 = random.choice(["John", "Sarah", "Mike", "Lisa", "Tom", "Emily"])
    name2 = random.choice(["friend", "brother", "sister", "teacher"])
    render_as = "markdown"  # Word problems are typically text

    if problem_type == "one_step_easy":
        val1 = random.randint(5, 50)
        val2 = random.randint(5, val1 - 1 if val1 > 10 else 20)  # Ensure val2 is reasonable
        action = random.choice(["sum", "difference"])
        if action == "sum":
            question = f"{name1} has {val1} {item}. They get {val2} more {item}. How many {item} does {name1} have in total?"
            answer = str(val1 + val2)
            explanation = f"{name1} starts with {val1} {item} and adds {val2} more. Total = {val1} + {val2} = {answer} {item}."
            difficulty = "easy"
        else:  # difference
            question = f"{name1} has {val1} {item}. They give {val2} {item} to their {name2}. How many {item} does {name1} have left?"
            answer = str(val1 - val2)
            explanation = f"{name1} starts with {val1} {item} and gives away {val2}. Remaining = {val1} - {val2} = {answer} {item}."
            difficulty = "easy"

    elif problem_type == "one_step_medium":
        # e.g. multiplication or division based
        if random.random() < 0.5:  # Multiplication
            val1 = random.randint(3, 12)  # Number of groups
            val2 = random.randint(5, 20)  # Items per group
            question = f"There are {val1} boxes. Each box contains {val2} {item}. How many {item} are there in total?"
            answer = str(val1 * val2)
            explanation = f"To find the total, multiply the number of boxes by the items per box: {val1} * {val2} = {answer} {item}."
            difficulty = "medium"
        else:  # Division (sharing)
            total_items = random.randint(20, 100)
            num_people = random.randint(2, 10)
            while total_items % num_people != 0:  # Ensure clean division for medium
                total_items = random.randint(20, 100)
                num_people = random.randint(2, 10)
            question = f"{name1} has {total_items} {item} to share equally among {num_people} people. How many {item} does each person get?"
            answer = str(total_items // num_people)
            explanation = f"To share equally, divide the total items by the number of people: {total_items} / {num_people} = {answer} {item} each."
            difficulty = "medium"

    elif problem_type == "two_step_medium":
        # e.g. (add then subtract) or (multiply then add)
        val1 = random.randint(10, 30)
        val2 = random.randint(5, 15)
        val3 = random.randint(3, 10)
        question = f"{name1} baked {val1} {item}. They gave {val2} {item} to {name2} and then baked {val3} more. How many {item} does {name1} have now?"
        answer_val = val1 - val2 + val3
        answer = str(answer_val)
        explanation = f"First, subtract the given away: {val1} - {val2} = {val1 - val2}. Then add the newly baked: {val1 - val2} + {val3} = {answer_val} {item}."
        difficulty = "medium"
        if answer_val > 30 or val1 > 25: difficulty = "hard"  # Make some two-steps harder

    elif problem_type == "two_step_hard":
        # e.g., multiple multiplications and addition, or rates
        cost_per_item = random.randint(2, 8)  # dollars, for example
        num_item1 = random.randint(3, 7)
        num_item2 = random.randint(2, 5)
        item1_name = random.choice(["pens", "notebooks"])
        item2_name = random.choice(["erasers", "folders"])
        question = (f"{name1} bought {num_item1} {item1_name} at ${cost_per_item} each, "
                    f"and {num_item2} {item2_name} at ${cost_per_item + random.randint(1, 3)} each. "
                    f"How much did {name1} spend in total?")
        cost1 = num_item1 * cost_per_item
        cost_item2_each = cost_per_item + random.randint(1, 3)  # Recalculate to match question
        cost2 = num_item2 * cost_item2_each
        answer_val = cost1 + cost2
        answer = f"${answer_val}"  # Assuming monetary answer
        explanation = (f"Cost of {item1_name}: {num_item1} * ${cost_per_item} = ${cost1}. "
                       f"Cost of {item2_name}: {num_item2} * ${cost_item2_each} = ${cost2}. "
                       f"Total spent = ${cost1} + ${cost2} = ${answer_val}.")
        difficulty = "hard"

    return "Word Problems", question, answer, explanation, difficulty, render_as


def generate_dataset(num_rows=250):  # Increased default number of questions
    """Generates the full dataset."""
    dataset = []
    question_generators = [
        generate_algebra_question,
        generate_quadratic_question,
        generate_arithmetic_question,
        generate_geometry_question,
        generate_word_problem
    ]

    # Weights for generators to get more variety, e.g. more algebra/geometry for harder Qs
    # Simple approach: just ensure enough calls for each type during generation loop.
    # Or, explicitly try to balance difficulty levels.

    # Add specific examples for partial/incorrect classification training (as before)
    # Ensure these have the new 'render_as' column
    dataset.append({
        "question_id": "Q_PARTIAL_001", "topic": "Algebra",
        "question_text": "Solve for x: x^2 - 5x + 6 = 0",
        "correct_answer": "x = 2 or x = 3",
        "solution_explanation": "Factor as (x-2)(x-3)=0. Solutions are x=2 and x=3.",
        "difficulty": "medium", "label": "correct", "render_as": "latex"
    })
    dataset.append({
        "question_id": "Q_INCORRECT_001", "topic": "Algebra",
        "question_text": "Solve for x: 2x + 5 = 1",
        "correct_answer": "x = -2",
        "solution_explanation": "2x = 1 - 5 => 2x = -4 => x = -2. A common mistake is x=2 (sign error).",
        "difficulty": "easy", "label": "correct", "render_as": "latex"
    })
    dataset.append({
        "question_id": "Q_ERROR_001", "topic": "Geometry",
        "question_text": "What is the area of a square with side 5 cm? Include units in your answer.",
        # Added clarity for user
        "correct_answer": "25 cm^2",
        "solution_explanation": "Area = side^2 = $5^2 = 25$. Units are cm$^2$.",
        "difficulty": "medium", "label": "correct", "render_as": "latex"
    })
    # Add a specific fractional question for testing
    dataset.append({
        "question_id": "Q_FRACTION_001", "topic": "Algebra",
        "question_text": "Solve for x: 3x + 4 = 9",  # 3x = 5, x = 5/3
        "correct_answer": "x = 5/3",
        "solution_explanation": "Subtract 4 from both sides: 3x = 9 - 4 => 3x = 5. Divide by 3: x = 5/3.",
        "difficulty": "medium", "label": "correct", "render_as": "latex"
    })

    # Aim for a better difficulty distribution
    # This is a simple way; a more complex approach would track counts per difficulty
    generated_difficulties = {"easy": 0, "medium": 0, "hard": 0}
    target_hard_questions = num_rows // 5  # Aim for 20% hard questions
    target_medium_questions = num_rows // 2  # Aim for 50% medium

    for i in range(len(dataset), num_rows):
        q_id = f"Q{str(i + 1).zfill(3)}"  # Ensure unique IDs even with pre-defined ones

        # Simple strategy to try and balance difficulty
        # This is a basic attempt, can be improved
        chosen_generator = random.choice(question_generators)
        topic, q_text, c_ans, sol_exp, difficulty, render_mode = chosen_generator()

        # Try to get more hard/medium questions if lacking
        if generated_difficulties['hard'] < target_hard_questions and difficulty != 'hard' and random.random() < 0.3:
            # Try to regenerate for a harder question
            for _ in range(3):  # try a few times
                topic, q_text, c_ans, sol_exp, difficulty_new, render_mode_new = random.choice(question_generators)()
                if difficulty_new == 'hard':
                    difficulty, render_mode = difficulty_new, render_mode_new
                    break
        elif generated_difficulties[
            'medium'] < target_medium_questions and difficulty == 'easy' and random.random() < 0.4:
            for _ in range(3):
                topic, q_text, c_ans, sol_exp, difficulty_new, render_mode_new = random.choice(question_generators)()
                if difficulty_new == 'medium':
                    difficulty, render_mode = difficulty_new, render_mode_new
                    break

        generated_difficulties[difficulty] += 1

        dataset.append({
            "question_id": q_id,
            "topic": topic,
            "question_text": q_text,
            "correct_answer": c_ans,
            "label": "correct",  # All base questions are 'correct' by definition
            "solution_explanation": sol_exp,
            "difficulty": difficulty,
            "render_as": render_mode
        })

    return dataset


if __name__ == "__main__":
    num_questions_to_generate = 250  # Increased from 200
    questions_data = generate_dataset(num_questions_to_generate)

    # Added 'render_as' to fieldnames
    fieldnames = ["question_id", "topic", "question_text", "correct_answer", "label", "solution_explanation",
                  "difficulty", "render_as"]

    with open(DATA_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(questions_data)

    print(f"Successfully generated {len(questions_data)} questions in '{DATA_FILE}'")

    # Print a few samples with the new column
    print("\nSample Questions (first 5):")
    for i in range(min(5, len(questions_data))):
        print(questions_data[i])

    # Print difficulty distribution
    if questions_data:
        df_temp = pd.DataFrame(questions_data)
        print("\nGenerated Difficulty Distribution:")
        print(df_temp['difficulty'].value_counts().to_string())