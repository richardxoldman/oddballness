from detokenize_sentence import detokenize_sentence
from get_oddballness_for_decoder import get_oddballness_for_decoder, get_oddballness_for_decoder_with_prompt
from get_topk_values import get_topk_values, get_topk_values_with_prompt
from get_probability_for_decoder import get_probability_for_decoder, get_probability_for_decoder_with_prompt


def generate_outputs(sentences, method, model, tokenizer, device):
    decoder_outputs = []
    if method == "oddballness":
        negative_pred_value = 0.0
    elif method == "topk":
        negative_pred_value = 0
    else:
        negative_pred_value = 1.0

    for idx, sentence_tokens in enumerate(sentences):
        sentence_predictions = []
        sentence = detokenize_sentence(sentence_tokens)

        model_tokens = tokenizer(sentence, return_tensors='pt').to(device).input_ids
        if method == "oddballness":
            values = get_oddballness_for_decoder(sentence, model, tokenizer, device)
        elif method == "topk":
            values = get_topk_values(sentence, model, tokenizer, device)
        else:
            values = get_probability_for_decoder(sentence, model, tokenizer, device)
        
        # 0.0 oddballness for the first word in the sentence
        sentence_predictions.append(negative_pred_value)

        # model_tokens_idx = 1 for llama/mistral
        model_tokens_idx = 1
        sentence_tokens_idx = 0

        word = ""
        word_to_match = sentence_tokens[sentence_tokens_idx]
        while word != word_to_match:
            increase = 1
            while "�" in tokenizer.decode(model_tokens[0][model_tokens_idx:model_tokens_idx+increase]).strip():
                increase += 1
            word += tokenizer.decode(model_tokens[0][model_tokens_idx:model_tokens_idx+increase]).strip()
            model_tokens_idx += increase - 1

            while len(word) > len(word_to_match):
                word_to_match += sentence_tokens[sentence_tokens_idx+1]
                sentence_predictions.append(negative_pred_value)
                sentence_tokens_idx += 1

            model_tokens_idx += 1
        sentence_tokens_idx += 1

        last_model_tokens_idx = model_tokens_idx-1
        while model_tokens_idx < model_tokens.size()[1]:
            last_model_tokens_idx = model_tokens_idx-1
            word = ""
            word_to_match = sentence_tokens[sentence_tokens_idx]

            while word != word_to_match:
                increase = 1
                while "�" in tokenizer.decode(model_tokens[0][model_tokens_idx:model_tokens_idx+increase]).strip():
                    increase += 1
                word += tokenizer.decode(model_tokens[0][model_tokens_idx:model_tokens_idx+increase]).strip()
                model_tokens_idx += increase - 1
                    
                while len(word) > len(word_to_match):
                    word_to_match += sentence_tokens[sentence_tokens_idx+1]
                    sentence_predictions.append(negative_pred_value)
                    sentence_tokens_idx += 1

                model_tokens_idx += 1
            
            if method == "oddballness" or method == "topk":
                val = max(values[last_model_tokens_idx:model_tokens_idx-1].tolist())
            else:
                val = min(values[last_model_tokens_idx:model_tokens_idx-1].tolist())
            last_model_tokens_idx = model_tokens_idx
            
            sentence_predictions.append(val)
            sentence_tokens_idx += 1
        
        decoder_outputs.append(sentence_predictions)
        if len(sentence_tokens) != len(sentence_predictions):
            print(sentence_tokens)
            print(sentence_predictions)
                
    return decoder_outputs
