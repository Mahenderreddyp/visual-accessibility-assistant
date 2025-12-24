from collections import Counter

def get_majority_answer(answers_list):
    """
    Returns the most common answer from a list of crowdsourced answers.
    Handles ties by picking the first most common.
    """
    if not answers_list:
        return "Unanswerable"
    
    # Filter out empty strings if any
    clean_answers = [a for a in answers_list if a]
    if not clean_answers:
        return "Unanswerable"

    counts = Counter(clean_answers)
    return counts.most_common(1)[0][0]

def format_vizwiz_entry(example):
    """
    Transforms a raw VizWiz entry into the Unsloth/Qwen conversational format.
    """
    # 1. Check answerability (0 means unanswerable in VizWiz)
    if example.get('answerable', 0) == 0:
        best_answer = "I cannot answer this question because the image is blurry, dark, or the subject is not visible."
    else:
        best_answer = get_majority_answer(example['answers'])

    # 2. Construct Conversation
    # Qwen2.5-VL expects a specific list of dictionaries
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example['image']},
                {"type": "text", "text": example['question']}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": best_answer}
            ]
        }
    ]
    
    return {"messages": messages}