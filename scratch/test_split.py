import re

def split_stream_sentences(buffer: str) -> tuple[list[str], str]:
    pattern = r'([^.?!。？！\n\r]*[.?!。？！]+(?:\s+|\Z)|[^.?!。？！\n\r]*[\n\r]+)'
    
    matches = list(re.finditer(pattern, buffer))
    if not matches:
        if len(buffer) > 150:
            split_pts = [m.start() for m in re.finditer(r'[\s,、]', buffer)]
            if split_pts:
                split_pt = max([p for p in split_pts if p <= 150] or [split_pts[-1]])
                sentence = buffer[:split_pt + 1]
                remaining = buffer[split_pt + 1:]
                return [sentence], remaining
        return [], buffer
        
    sentences = []
    last_end = 0
    for match in matches:
        sentence = match.group(0)
        sentences.append(sentence)
        last_end = match.end()
        
    remaining = buffer[last_end:]
    return sentences, remaining

# Test extremely long buffer without standard punctuation
long_input = "This is a very long sentence without any punctuation that is meant to test the length-based splitting logic so that we do not have a huge delay when the model outputs long paragraphs without dots"
sentences, remaining = split_stream_sentences(long_input)
print(f"Sentences: {sentences}")
print(f"Remaining: {repr(remaining)}")
