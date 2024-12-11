# Oddballness: universal anomaly detection with language models
This repository contains scripts for the paper 

Oddballness: universal anomaly detection with language models (https://arxiv.org/abs/2409.03046)

# How to run experiments
To run MultiGED-2023 experiments it is needed to:
1) Create conda environment from environment.yml file
2) Add HuggingFace token (to access Mistral 7b v1 model)
3) Run get_multiged_results.py script (with the conda environment from 1.)


The get_multiged_results.py script will compute thresholds for dev sets and generate outputs that can be submitted to MultiGED-2023 Shared Task at CodaLab:
https://codalab.lisn.upsaclay.fr/competitions/9784


The code in generate_outputs.py is complex because there is a need for proper alignment of tokens produced by the model tokenizer with tokens in the dataset.


To switch between the versions of the experiments (with/without additional prompt), it is necessary to change 3 functions in lines 22, 24, 26 of the generate_outputs.py file:

* get_{oddballness/probability}_for_decoder, get_topk_values functions **without** additional prompt

* get_{oddballness/probability}_for_decoder_with_prompt, get_topk_values_with_prompt functions **with** additional prompt


We had to make a change in the it_merlin_dev.tsv file in one line, because there was a space within a token.
line 5351: ques ' -> ques'
# Authors
Filip Graliński (UAM, Snowflake)

Ryszard Staruch (UAM, Center for Artificial Intelligence in Poznań)

Krzysztof Jurkiewicz (UAM)
