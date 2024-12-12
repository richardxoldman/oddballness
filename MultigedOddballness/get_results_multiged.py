import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from compute_predictions import compute_predictions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder_model_name = 'mistralai/Mistral-7B-v0.1'
data_directory = "multiged-2023"
results_directory = "AllResults"

# below add your huggingface access token
token = "huggingface_access_token"

tokenizer = AutoTokenizer.from_pretrained(decoder_model_name, trust_remote_code=True, token=token)
model = AutoModelForCausalLM.from_pretrained(decoder_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, token=token, device_map="auto")
model.eval()

METHODS = ["oddballness", "probability", "topk"]
for method in METHODS:
    print(f"EN FCE {method}")
    compute_predictions(dev_file_path=f"{data_directory}/english/en_fce_dev.tsv", 
                        test_file_path=f"{data_directory}/english/en_fce_test_unlabelled.tsv", 
                        method=method, 
                        result_file_path=f"{results_directory}/en_fce_{method}.tsv",
                        model=model,
                        tokenizer=tokenizer,
                        device=device)
for method in METHODS:
    print(f"EN REALEC {method}")
    compute_predictions(dev_file_path=f"{data_directory}/english/en_realec_dev.tsv", 
                        test_file_path=f"{data_directory}/english/en_realec_test_unlabelled.tsv", 
                        method=method, 
                        result_file_path=f"{results_directory}/en_realec_{method}.tsv",
                        model=model,
                        tokenizer=tokenizer,
                        device=device)
for method in METHODS:
    print(f"CS {method}")
    compute_predictions(dev_file_path=f"{data_directory}/czech/cs_geccc_dev.tsv", 
                        test_file_path=f"{data_directory}/czech/cs_geccc_test_unlabelled.tsv", 
                        method=method, 
                        result_file_path=f"{results_directory}/cs_{method}.tsv",
                        model=model,
                        tokenizer=tokenizer,
                        device=device)
for method in METHODS:
    print(f"DE {method}")
    compute_predictions(dev_file_path=f"{data_directory}/german/de_falko-merlin_dev.tsv", 
                        test_file_path=f"{data_directory}/german/de_falko-merlin_test_unlabelled.tsv", 
                        method=method, 
                        result_file_path=f"{results_directory}/de_{method}.tsv",
                        model=model,
                        tokenizer=tokenizer,
                        device=device)
for method in METHODS:
    print(f"IT {method}")
    compute_predictions(dev_file_path=f"{data_directory}/italian/it_merlin_dev.tsv", 
                        test_file_path=f"{data_directory}/italian/it_merlin_test_unlabelled.tsv", 
                        method=method, 
                        result_file_path=f"{results_directory}/it_{method}.tsv",
                        model=model,
                        tokenizer=tokenizer,
                        device=device)
for method in METHODS:
    print(f"SV {method}")
    compute_predictions(dev_file_path=f"{data_directory}/swedish/sv_swell_dev.tsv", 
                        test_file_path=f"{data_directory}/swedish/sv_swell_test_unlabelled.tsv", 
                        method=method, 
                        result_file_path=f"{results_directory}/sv_{method}.tsv",
                        model=model,
                        tokenizer=tokenizer,
                        device=device)