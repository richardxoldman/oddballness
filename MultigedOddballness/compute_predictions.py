from get_optimal_threshold import get_optimal_threshold
from generate_outputs import generate_outputs


def compute_predictions(dev_file_path, test_file_path, method, result_file_path, model, tokenizer, device):
    threshold, best_fscore = get_optimal_threshold(dev_file_path, method, model, tokenizer, device)
        
    with open(test_file_path, "r") as file:
        data = file.readlines()
        data = [x.strip() for x in data]

    sentences = []
    current_sentence_tokens = []
    for item in data:
        if item == "":
            sentences.append(list(current_sentence_tokens))
            current_sentence_tokens = []
        else:
            current_sentence_tokens.append(item.replace(r"\"", "\""))

    outputs = generate_outputs(sentences, method, model, tokenizer, device) 
    res = []
    idx_sen = 0
    for sentence_preds in outputs:
        if method == "oddballness" or method == "topk":
            for pred in sentence_preds:
                if pred > threshold:
                    res.append(f"{data[idx_sen]}\ti\n")
                    idx_sen += 1
                else:
                    res.append(f"{data[idx_sen]}\tc\n")
                    idx_sen += 1
        else:
            for pred in sentence_preds:
                if pred < threshold:
                    res.append(f"{data[idx_sen]}\ti\n")
                    idx_sen += 1
                else:
                    res.append(f"{data[idx_sen]}\tc\n")
                    idx_sen += 1
        res.append("\n")
        idx_sen += 1

    with open(result_file_path, "w") as file:
        file.writelines(res)
