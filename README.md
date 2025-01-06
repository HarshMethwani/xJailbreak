# Overview

The code of paper "RL-DROJ: RL-based Directional Representation Optimization for LLM Jailbreaking".

## Requirement

### Enviroment

Linux Ubuntu; Python 3.10; A800 (80G).

We deployed at least one non-safety aligned LLM locally, and we deploy 1 or 2 LLM in the experiment. We recommend that you have more than 60G of GPU memory. If you want to use multiple GPU, you may need to adjust the model loading method in `agent/LLM_agent.py`.

### Necessary packages

torch, transformers, numpy, pandas, sklearn, tqdm, openai

### LLM configuration

We deploy [Llama3-8B-Instruct-JB](https://huggingface.co/cooperleong00/Meta-Llama-3-8B-Instruct-Jailbroken) locally And **this is necessary** because we need a LLM without safety alignment. If you have other similar LLMs, you can replace it. We only support chat templates for Llama and Qwen. If you change to a new model, you may also need to modify the template, which is located in `agent/LLM_agent.py`.

We also deploy safety aligned LLM like [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) locally **but it is not necessary**. You can also replace it with any other local or API LLM. This is used as a target LLM with safety alignment.

We assume that you have a model path in the environment variables `model_path` that is completed in a form similar to the following code in `agent/LLM_agent.py`:

```python
# * ------ If you add a local model, pay attention here -------
if 'Llama' in agt_name:
    model_path_name = os.getenv('model_path') + 'Llama/' + agt_name
elif 'Qwen' in agt_name:
    model_path_name = os.getenv('model_path') + 'Qwen/' + agt_name
# * ------------------------------------------------------------
```

**Another way:** `ManageLLM` in `agent/LLM_agent.py` is the method by which we manage all local and API LLMs. If you think the package is difficult to modify, you can rewrite it. Your rewritten package should satisfy the call method of the `Llm` class in `agent/LLM_agent.py`.

## Run

At first you need to train a RL agent, run:

```shell
python train.py --special_place "train/" -w 1 --cuda 0
```

We recommend that you modify the `--cuda` parameters and set them to the GPU number that you currently have available. After training, the weights and training data of the RL agent will be saved in `log/train/`.

Now you can execute attack, run:

``````shell
python text.py --special_place "test/" -w 1 --cuda 0 --target "qwen" --weight_path "{your rl-agent model file, such as "log/train/ckpt/42_PPO_linux.pt"}"
``````

We also recommend that you modify the `--cuda` parameter. After the attack is completed, the statistical results are in `log/test`.