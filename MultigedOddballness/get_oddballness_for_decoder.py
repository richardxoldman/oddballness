import torch


relu = torch.nn.ReLU()

def get_oddballness_for_decoder(text, model, tokenizer, device):
    model_input = tokenizer(text, return_tensors='pt').to(device)
    tokens_tensor = model_input.input_ids[0]
    
    with torch.no_grad():
        outputs = model(**model_input)
    output = torch.softmax(outputs[0][0], dim=1)

    tokens_tensor = tokens_tensor[1:]
    output = output[:-1]

    probabilities = output[torch.arange(output.shape[0]), tokens_tensor]
    oddballness = torch.sum(relu(output - probabilities[:, None]), dim=1)

    return oddballness


def get_oddballness_for_decoder_with_prompt(text, model, tokenizer, device):
    prompt = "An example of a grammatically correct text in any language that may be out of context: "
    model_input = tokenizer(prompt + text, return_tensors='pt').to(device)
    tokens_tensor = model_input.input_ids[0]
    
    prompt_token_length = tokenizer(prompt, return_tensors='pt').input_ids.shape[1]
    with torch.no_grad():
        outputs = model(**model_input)
    output = torch.softmax(outputs[0][0], dim=1)

    tokens_tensor = tokens_tensor[prompt_token_length-1:]
    output = output[prompt_token_length-2:-1]

    probabilities = output[torch.arange(output.shape[0]), tokens_tensor]
    oddballness = torch.sum(relu(output - probabilities[:, None]), dim=1)

    return oddballness
