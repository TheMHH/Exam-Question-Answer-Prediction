EXAM_QUESTION_PROMPT =  """Let's analyze this exam question step by step:
    1. Read the Question Carefully:

        Identify key terms, concepts, and what is explicitly being asked.

        Note any specific conditions or constraints mentioned.

    2. Examine All Provided Visuals (Images, Tables, Graphs, Shapes, etc.):

        If there's an image, analyze its components (labels, arrows, structures, etc.).

        If there's a table, check rows, columns, and highlighted data for patterns or clues.

        If there's a graph or diagram, interpret axes, trends, or relationships depicted.

    3. Break Down Each Option (A, B, C, D, E):

        Understand what each option represents.

        Compare them against the question's requirements and visual context.

        Eliminate clearly incorrect choices first.

    4. Apply Logical Reasoning:

        Relate the question to known concepts or formulas.

        Cross-check visuals with textual information for consistency.

    5. Select the Best Answer:

        Choose the option that best fits both the question and the supporting visuals.

        Justify why other options are incorrect (if possible).
        
        Do not add any text after providing Final Answer"""

EXAM_QUESTION_CHAT_TEMPLATE = [
    {
        "role": "user", 
        "content": [
            {"type": "image"},
            {"type": "text", "text": EXAM_QUESTION_PROMPT},
        ]
    },
]

EXTRACT_ANSWER_PROMPT = """
    After analyzing the question and options, respond with only the correct answer choice (A, B, C, D, or E) in a single word. Do not include explanations, reasoning, or any other text.
"""

ANSWER_TO_IDX = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4
}

IDX_TO_ANSWER = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'
} 