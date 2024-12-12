import torch


relu = torch.nn.ReLU()

def get_topk_values(text, model, tokenizer, device):
    model_input = tokenizer(text, return_tensors='pt').to(device)
    tokens_tensor = model_input.input_ids[0]
    
    with torch.no_grad():
        outputs = model(**model_input)
    output = torch.softmax(outputs[0][0], dim=1)

    tokens_tensor = tokens_tensor[1:]
    output = output[:-1]
    ranks = []
    sorted_probs, sorted_indices = torch.sort(output, descending=True)

    for x in range(output.shape[0]):
        rank = torch.where(sorted_indices[x] == tokens_tensor[x])[0][0].item()+1
        ranks.append(rank)

    return torch.Tensor(ranks)


def get_topk_values_with_prompt(text, model, tokenizer, device):
    prompt = "An example of a grammatically correct text in any language that may be out of context: "
    model_input = tokenizer(prompt + text, return_tensors='pt').to(device)
    tokens_tensor = model_input.input_ids[0]
    
    prompt_token_length = tokenizer(prompt, return_tensors='pt').input_ids.shape[1]
    with torch.no_grad():
        outputs = model(**model_input)
    output = torch.softmax(outputs[0][0], dim=1)

    tokens_tensor = tokens_tensor[prompt_token_length-1:]
    output = output[prompt_token_length-2:-1]
    ranks = []
    sorted_probs, sorted_indices = torch.sort(output, descending=True)

    for x in range(output.shape[0]):
        rank = torch.where(sorted_indices[x] == tokens_tensor[x])[0][0].item()+1
        ranks.append(rank)

    return torch.Tensor(ranks)
