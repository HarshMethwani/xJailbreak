# Data source

1. begnin_prompt
    1. `b_custom.txt` [(DRO) On Prompt-Driven Safeguarding for Large Language Models](https://github.com/chujiezheng/LLM-Safeguard/blob/main/code/data_harmless/custom.txt)
    2. `testset.txt` [(DRO) On Prompt-Driven Safeguarding for Large Language Models](https://github.com/chujiezheng/LLM-Safeguard/blob/main/code/data_harmless/testset.txt)
2. harmful_prompt
    1. `h_custom.txt` [(DRO) On Prompt-Driven Safeguarding for Large Language Models](https://github.com/chujiezheng/LLM-Safeguard/blob/main/code/data/custom.txt)
    2. `MaliciousInstruct.txt` [(Huang et al.) Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation](https://github.com/Princeton-SysML/Jailbreak_LLM/blob/main/data/MaliciousInstruct.txt)
    3. `harmful_behaviors.csv` [(advBench, GCG) Universal and Transferable Adversarial Attacks on Aligned Language Models](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv)
    4. `harmful_strings.csv` [(advBench, GCG) Universal and Transferable Adversarial Attacks on Aligned Language Models](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_strings.csv)
3. template_prompt
    1. `our_template.txt` modified from [RL-JACK: Reinforcement Learning-powered Black-box Jailbreaking Attack against LLMs](http://arxiv.org/abs/2406.08725)
4. asr_word
    1. `asr_keyword.txt` [Automated Progressive Red Teaming](http://arxiv.org/abs/2407.03876) 和 [RL-JACK: Reinforcement Learning-powered Black-box Jailbreaking Attack against LLMs](http://arxiv.org/abs/2406.08725)
# Data summary

1. begnin_prompt
    1. `b_custom.txt`: 100, ordinary malicious instructions
    2. `testset.txt`: 100 malicious instructions used by the original paper for testing
2. harmful_prompt
    1. `h_custom.txt`: 100, corresponding to the custom dataset in benign prompts, retain related verbs, that is, ensure a certain text similarity, but change the semantics to benign
    2. `MaliciousInstruct.txt`: 100, ordinary malicious instructions
    3. `harmful_behaviors.csv`: 520, ordinary malicious instructions
    4. `harmful_strings.csv`: 574, malicious strings
3. template_prompt
    1. `our_template.txt`: 10
4. asr_word
    1. `asr_keyword.txt`: 31 keywords. If these keywords are detected in VictimLLM’s answer, the attack will fail.

> [!NOTE]
> The two `custom` data are used for low-dimensional representation in the original paper to distinguish benign space from malicious space, and the prefixes `b_` and `h_` are added to distinguish them.

# Extractor

Q: Why do we need `Extractor.py`?

A: Different data formats are different, and this script performs special extractions and finally gives the data output in the form of a list, where each element is an instruction or statement.