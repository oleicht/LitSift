def process_text(input_text):
    # This is a placeholder function that processes the input text
    # For example, it could split the text into sentences
    sentences = []
    for ii, title in enumerate(input_text.split('.')):
        sentences += [(title, f"looong descriotion {ii}")]
    
    # Remove any empty sentences and strip leading/trailing whitespace
    # sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences