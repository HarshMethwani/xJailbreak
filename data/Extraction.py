import pandas as pd
import re

# 数据文件类型
txt_data_name = ['b_custom', 'testset', 'h_custom', 'MaliciousInstruct', 'our_template']
csv_data_name = ['harmful_behaviors', 'harmful_strings', 'add_harmful_data', 'add_benign_data']

# 数据性质
harmful_data_name = ['h_custom', 'MaliciousInstruct', 'harmful_behaviors', 'harmful_strings', 'add_harmful_data']
benign_data_name = ['b_custom', 'testset', 'add_benign_data']
template_data_name = ['our_template']
asr_data_name = ['asr_keyword']

all_data_name = harmful_data_name + benign_data_name + template_data_name + asr_data_name

def extract_text(text):
    # 使用正则表达式提取 <new prompt> 和 </new prompt> 之间的内容
    match = re.search(r'<new prompt>(.*?)</new prompt>', text, re.DOTALL)
    return match.group(1) if match else ''

def get_data_list(data_name: list[str] | str) -> dict:
    '''提取数据，需要导航至主目录

    用法
    ---
    ```python
    # 在主目录时
    from data.Extraction import get_data_list
    prompt, template = get_data_list(['h_custom', 'b_custom', 'harmful_behaviors', 'our_template'])  # our_template 是越狱模板数据
    # 返回一个字典 {'data_list': [...], 'template_list': [...], 'asr_list': [...]}
    ```
    '''
    if isinstance(data_name, str):
        data_name = [data_name]
    assert set(data_name).issubset(set(all_data_name)), f'{set(all_data_name) - set(data_name)} 数据未注册，请检查数据名称'

    data_list = []
    template_list = []
    asr_list = []

    harmful_data_attr_list = []
    benign_data_attr_list = []
    template_data_attr_list = []
    asr_data_attr_list = []
    for name in data_name:
        # 模板 prompt 数据
        if name in template_data_name:
            attr = 'template_prompt'
            file_type = 'txt'
            template_data_attr_list.append(name)
            with open(f'data/{attr}/{name}.{file_type}', 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f]
            template_list.extend(lines)
        # ASR 关键词
        elif name in asr_data_name:
            attr = 'asr_word'
            file_type = 'txt'
            asr_data_attr_list.append(name)
            with open(f'data/{attr}/{name}.{file_type}', 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f]
            asr_list.extend(lines)
        # prompt 数据
        else:
            if name in txt_data_name:
                file_type = 'txt'
            else:
                file_type = 'csv'
            if name in harmful_data_name:
                attr = 'harmful_prompt'
                harmful_data_attr_list.append(name)
            elif name in benign_data_name:
                attr = 'benign_prompt'
                benign_data_attr_list.append(name)

            if name in txt_data_name:
                with open(f'data/{attr}/{name}.{file_type}', 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f]
                data_list.extend(lines)
            elif 'add_' in name:
            #     data_list.extend(pd.read_csv(f'data/{attr}/{name}.{file_type}')['goal'].apply(extract_text))
            # else:
                data_list.extend(pd.read_csv(f'data/{attr}/{name}.{file_type}')['goal'].to_list())

    data_dict = {}
    data_dict['data_list'] = data_list if data_list else None
    data_dict['template_list'] = template_list if template_list else None
    data_dict['asr_list'] = asr_list if asr_list else None

    print('----> Retrieving data <----')
    print('Retrieve harmful data: ', harmful_data_attr_list) if harmful_data_attr_list else None
    print('Retrieve benign data: ', benign_data_attr_list)  if benign_data_attr_list else None
    print('Retrieve template: ', template_data_attr_list)  if template_data_attr_list else None
    print('Data amount:', len(data_list)) if len(data_list) else None
    print('Template amount:', len(template_list)) if len(template_list) else None
    print('ASR keywords amount:', len(asr_list)) if len(asr_list) else None

    return data_dict

if __name__ == '__main__':
    # data_dict = get_data_list(['h_custom', 'MaliciousInstruct', 'our_template', 'asr_keyword'])  # our_template 是越狱模板数据
    data_dict = get_data_list(['add_harmful_data'])
    print(data_dict)