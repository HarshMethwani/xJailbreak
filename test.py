import argparse
import os
import torch
import random
import numpy as np
from utils.Evaluator import Evaluator
from tqdm import trange
from net import PolicyNet, ValueNet
from data.Extraction import get_data_list
from agent.LLM_agent import Llm_manager
from jailbreak_env import JbEnv
import warnings
warnings.filterwarnings('ignore')

# * ---------------------- terminal parameters -------------------------

parser = argparse.ArgumentParser(description='Test agent')
parser.add_argument('-n', '--note', default='None', type=str, help='task note')
parser.add_argument('--special_place', default='val', help='Customize special saving location of experiment result, such as "logs/special_place/..."')
parser.add_argument('--weight_path', type=str, help='your RL agent model file, such as "logs/train/ckpt/42_PPO_linux.pt..."')
parser.add_argument('-t', '--target', default='qwen', help='target model, qwen, llama, gpt')
parser.add_argument('-w', '--save', default=0, type=int, help='data saving type, 0: not saved, 1: local')
parser.add_argument('-c', '--cuda', nargs='+', default=[0], type=int, help='CUDA order')
parser.add_argument('-s', '--seed', nargs='+', default=[42, 42], type=int, help='The start and end seeds')
args = parser.parse_args()


# CUDA
if isinstance(args.cuda, int):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.cuda}"  # 1 -> '1'
elif isinstance(args.cuda, list):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.cuda))  # [1, 2] -> "1,2"

# * ------ task ------
device = "cuda" if torch.cuda.is_available() else "cpu"
env_kwargs = {}
env_kwargs['max_step'] = 10  # How many steps are the maximum number of training steps per prompt
env_kwargs['train_ratio'] = 0.8

train_kwargs = {}
train_kwargs['val_interval'] = 5
train_kwargs['val_num'] = 3  # How many samples should be tested per validation
train_kwargs['val_max_step'] = 10  # How many times to iterate when verifying each sample

# * ------ model ------
# Helper models must be able to override safety instructions, highly recommend https://huggingface.co/cooperleong00/Meta-Llama-3-8B-Instruct-Jailbroken
helper_api = {
    'model_path': './Meta-Llama-3-8B-Instruct-Jailbroken/'
}
helpLLM = Llm_manager(helper_api)
helpLLM.load_model()


reprLLM = helpLLM.embedding

# preload harmful_emb_refer and benign_emb_refer for saving time
# the method of embedding them: torch.save(reprLLM(get_data_list(['h_custom'])['data_list']).cpu(), 'data/pre_load/harmful_emb_refer.pt')
env_kwargs['harmful_emb_refer'] = torch.load('data/preload/harmful_emb_refer.pt').to('cuda')
env_kwargs['benign_emb_refer'] = torch.load('data/preload/benign_emb_refer.pt').to('cuda')

# VictimLLM must be safety aligned
if args.target == 'qwen':
    victim_api = {'model_name': 'qwen',
                  'api': 'sk-2233...',
                  'url': 'https:// ...'}
    victimLLM = Llm_manager(victim_api)
elif args.target == 'llama':
    victim_api = {
        'model_path': './Mistral-7B-Instruct-v0.2 /'
    }
    victimLLM = Llm_manager(victim_api)
elif args.target == 'gpt':
    victim_api = {'model_name': 'gpt',
                  'api': 'sk-2233...',
                  'url': 'https:// ...'}
    victimLLM = Llm_manager(victim_api)
victimLLM.load_model('How are you?')


judgeLLM = helpLLM

# * ------ data ------
# ['h_custom', 'harmful_behaviors', 'MaliciousInstruct']
test_harmful_prompt = get_data_list(['MaliciousInstruct'])['data_list']
template = 'our_template'
ASR = 'asr_keyword'

# * ------ environment ------

env = JbEnv(helpLLM, reprLLM, victimLLM, judgeLLM, None, template, ASR, **env_kwargs)
# neural network
state_dim = env.observation_space
hidden_dim = env.observation_space // 4
action_dim = env.action_space

# * ------ RL agent PPO ------
actor_lr = 1e-4
critic_lr = 2e-4
lmbda = 0.97  # Discount factor for balanced advantage
gamma = 0.9  # The temporal difference learning rate, also used as one of the factors for discounting advantage. Here we are more concerned with short-term rewards.
inner_epochs = 10
eps = 0.2

policy_net = PolicyNet(state_dim, hidden_dim, action_dim)
value_net = ValueNet(state_dim, hidden_dim)

agent = torch.load(args.weight_path)

evaluator = Evaluator(args.special_place, print_info=False)

# * ------------------------ Training ----------------------------
print(f'[ Start >>> Note: {args.note} | save: {args.save} | cuda: {args.cuda} | seed: {args.seed} ]')
for seed in trange(args.seed[0], args.seed[-1] + 1, mininterval=40, ncols=70):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    test_info, val_info_whole = env.validation(RL_agent=agent, test_num=0, max_step=5, other_val_data=test_harmful_prompt)
    print('--------------')
    print(test_info)
    print('--------------')
    print(val_info_whole)
    print('--------------')
    evaluator.val_save(args.save, seed, test_info, val_info_whole, args.special_place)
