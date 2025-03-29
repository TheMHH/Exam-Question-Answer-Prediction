EXAM_QUESTION_PROMPT = """Let's analyze this exam question step by step:
1. First, carefully read the question text and understand what is being asked.
2. Look at the image provided below and understand the context it provides.
3. Look at each option (A, B, C, D, E) and understand what they mean.
4. Think about which option best answers the question based on both the text and the image context.
5. Select the correct answer choice (A, B, C, D, or E) based on the image and the question."""

ANSWER_TO_IDX = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4
}

IDX_TO_ANSWER = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'
} 