from sklearn.metrics import fbeta_score
from generate_outputs import generate_outputs


def get_optimal_threshold(data_path, method, model, tokenizer, device):
    with open(data_path, "r") as file:
        data = file.readlines()
        data = [x.strip() for x in data]
        labels = [x.split("\t")[1] for x in data if len(x) > 0]
        data = [x.split("\t")[0] if len(x) > 0 else x for x in data]

    labels_dict = {
        "c": 0,
        "i": 1
    }
    labels = [labels_dict[l] for l in labels]
    sentences = []
    current_sentence_tokens = []
    for item in data:
        if item == "":
            sentences.append(list(current_sentence_tokens))
            current_sentence_tokens = []
        else:
            current_sentence_tokens.append(item.replace(r"\"", "\""))
    
    outputs = generate_outputs(sentences, method, model, tokenizer, device)
    outputs_single_list = []
    for output in outputs:
        for o in output:
            outputs_single_list.append(o)

    best_threshold = 0
    if method == "oddballness":
        threshold_values = [x/100 for x in range(1, 100)]
    elif method == "topk":
        threshold_values = [x*2 for x in range(50)] + [110 + x*10 for x in range(50)]
    else:
        threshold_values = []
        for x in range(1, 7):
            for y in range(1, 10):
                threshold_values.append(y/(10**x))

    
    best_threshold = 0
    best_fscore = 0
    for t in threshold_values:
        if method == "oddballness":
            preds = [int(val > t) for val in outputs_single_list]
        elif method == "topk":
            preds = [int(val > t) for val in outputs_single_list]
        else:
            preds = [int(val < t) for val in outputs_single_list]
        score = fbeta_score(labels, preds, beta=0.5)

        if score > best_fscore:
            best_threshold = float(t)
            best_fscore = float(score)
    
    
    print(f"Best threshold on dev set:{best_threshold}")
    print(f"Best f0.5 score on dev set:{best_fscore}")

    return best_threshold, best_fscore