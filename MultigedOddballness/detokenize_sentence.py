def detokenize_sentence(tokens):
    sentence = ""
    for token in tokens:
        if token[0].isalnum() or token == '<mask>' or token[0] == "(":
            sentence += f" {token}"
        else:
            sentence += token
        
    sentence = sentence.strip()
    return sentence
