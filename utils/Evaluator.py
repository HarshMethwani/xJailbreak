import torch
import sys
import os
import pandas as pd
import time

class Evaluator:
    def __init__(self, special_place: str=None, print_info: bool=True):
        self.time_val_list = []
        self.log_time_val_list = []
        self.avg_reward_val_list = []
        self.llm_judge_val_list = []
        self.asr_judge_val_list = []
        self.intent_score_val_list = []
        self.soft_jb_step_val_list = []
        self.hard_jb_step_val_list = []
        self.success_rate = []

        self.special_place = special_place
        self.episode = 0
        self.time_flag = time.time()
        if print_info:
            print(f"\n| `seed`: Random seed of algorithm."
                  f"\n| `episode`: Number of current episode."
                  f"\n| `time`: Time spent (minute) from the start of training to this moment."
                  f"\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
                  f"\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
                  f"\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
                  f"\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
                  f"\n| `s_jb_stp`: Number of steps of soft jailbreak. Require current prompt has a similar intent of the origin prompt. -1 for fail."
                  f"\n| `jb_stp`: Number of steps of hard jailbreak. Require current prompt has the same intent of the origin prompt. -1 for fail")
            self.col = f"\n| {'seed':>3}  {'episode':>5}  {'time':>5}  |  {'avgR':>7}  {'stdR':>9}     | {'objC':>8}  {'objA':>8}  |  {'s_jb_stp':>4}   {'h_jb_stp':>4}"
            print(self.col)

    def print_col(self):
        print(self.col)

    def evaluate_and_save(self, writer, return_list, time_list, seed_list, episode, agent, seed,
                          actor_loss_list, critic_loss_list, llm_judge_list, asr_judge_list,
                          intent_score_list, soft_jb_step_list, hard_jb_step_list, **kwargs):


        # * ---- Arrange the path ----
        system_type = sys.platform
        if self.special_place:
            file_path = f"logs/{self.special_place}/plot_data"
            ckpt_path = f"logs/{self.special_place}/ckpt"
        else:
            file_path = f"logs/plot_data"
            ckpt_path = f"logs/ckpt"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        log_file_path = f"{file_path}/{seed}_PPO_{system_type}.csv"
        ckpt_file_path = f"{ckpt_path}/{seed}_PPO_{system_type}.pt"

        # * ---- Storage Information ----
        return_save = pd.DataFrame()
        return_save["Seed"] = seed_list
        return_save["Return"] = return_list
        return_save['llm_judge'] = llm_judge_list
        return_save['asr_judge_list'] = asr_judge_list
        return_save['intent_score_list'] = intent_score_list
        return_save["hard_JB_step"] = hard_jb_step_list
        return_save["soft_JB_step"] = soft_jb_step_list
        return_save['Actor_loss'] = sum(actor_loss_list) / len(actor_loss_list) if actor_loss_list else None
        return_save['Critic_loss'] = sum(critic_loss_list) / len(critic_loss_list) if critic_loss_list else None
        return_save["Log_time"] = time_list
        if writer:
            return_save.to_csv(log_file_path, index=False, encoding='utf-8-sig')
            if agent.agent_type != 'random':
                torch.save(agent, ckpt_file_path)


        # * ---- Print information ----
        self.episode = episode + 1
        used_time = (time.time() - self.time_flag) / 60
        avg_r = return_save["Return"].mean()
        std_r = return_save["Return"].std()
        actor_loss = return_save['Actor_loss'].mean()
        critic_loss = return_save['Critic_loss'].mean()
        soft_jb_steps = soft_jb_step_list[-1]
        hard_jb_steps = hard_jb_step_list[-1]
        print(f"| {seed:3d}  {self.episode:5d}    {used_time:5.1f}   "
              f"| {avg_r:9.2f}  {std_r:9.2f}    "
              f"| {critic_loss:8.1f}  {actor_loss:8.1f}  "
              f"| {soft_jb_steps:4d}      {hard_jb_steps:4d}")

    def val_save(self, writer, seed: int, val_info: dict, val_info_whole: dict, special_place: str=None):
        if writer:
            self.time_val_list.append(val_info['time (min)'])
            self.avg_reward_val_list.append(val_info['avg_reward (1)'])
            self.llm_judge_val_list.append(val_info['avg_llm_judge (1)'])
            self.asr_judge_val_list.append(val_info['avg_asr_judge (1)'])
            self.intent_score_val_list.append(val_info['avg_intent_score (1)'])
            self.soft_jb_step_val_list.append(val_info['avg_h_jb_step (0)'])
            self.hard_jb_step_val_list.append(val_info['avg_s_jb_step (0)'])
            self.success_rate.append(val_info['success_rate'])
            self.log_time_val_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))

            # * ---- Arrange the path ----
            if special_place:
                file_path = f"logs/{special_place}/"
            elif not special_place and self.special_place:
                file_path = f"logs/{self.special_place}/plot_val_data/"
            else:
                file_path = f"logs/{self.special_place}/"
            system_type = sys.platform
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            result_log_file_path = f"{file_path}/{seed}_{system_type}.csv"

            # * ---- Storage Information ----
            return_save = pd.DataFrame()
            return_save["Seed"] = [seed] * len(self.time_val_list)
            return_save['time'] = self.time_val_list
            return_save['success_rate'] = self.success_rate
            return_save["avg_reward"] = self.avg_reward_val_list
            return_save["avg_llm_judge"] = self.llm_judge_val_list
            return_save["avg_asr_judge"] = self.asr_judge_val_list
            return_save['avg_intent_score'] = self.intent_score_val_list
            return_save['avg_h_jb_step'] = self.soft_jb_step_val_list
            return_save['avg_s_jb_step'] = self.hard_jb_step_val_list
            return_save["Log_time"] = self.log_time_val_list

            if val_info_whole:
                detail_log_file_path = f"{file_path}/{seed}_{system_type}_detail.csv"
                pd.DataFrame(val_info_whole).to_csv(detail_log_file_path)

            return_save.to_csv(result_log_file_path, index=False, encoding='utf-8-sig')