import torch
import clip
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nlpaug.augmenter.word as naw
import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 初始化混合增强器（在文件顶部）
# 1. 规则-based词序交换
swap_aug = naw.RandomWordAug(action="swap")
# 2. T5深度改写
t5_tokenizer = AutoTokenizer.from_pretrained("/mnt/C/znmd_lgs/GAP/T5/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("/mnt/C/znmd_lgs/GAP/T5/").to(device)

def paraphrase(text, max_length=50):
    """T5深度改写"""
    input_ids = t5_tokenizer.encode(
        f"paraphrase: {text}", return_tensors="pt", max_length=50, truncation=True
    ).to(device)
    outputs = t5_model.generate(input_ids, max_length=max_length, num_beams=3, temperature=0.7)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def mixed_augment(text,
                  use_swap=True,
                  use_paraphrase=True,
                  prob_swap=0.2,
                  prob_t5=0.4
                  ):
    
    assert prob_swap + prob_t5 <= 1.0
    if not isinstance(text, str):
        text = str(text)

    r = random.random()      
    if use_swap and r < prob_swap and len(text.split()) <= 3:
        return swap_aug.augment(text)
    elif use_paraphrase and r < prob_swap + prob_t5 and len(text.split()) > 3:
        return paraphrase(text)
    else:
        return text



    
label_text_map = []

with open('text/ntu120_label_map02.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        label_text_map.append(line.rstrip().lstrip())



paste_text_map0 = []

with open('text/synonym02.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(',')
        paste_text_map0.append(temp_list)
        

paste_text_map1 = []
sentence_text_map = {}

with open('text/sentence02.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split('.')
        while len(temp_list) < 4:
            temp_list.append(" ")
        paste_text_map1.append(temp_list)
        parts = line.rstrip().lstrip().split('.', 1)
        if len(parts) > 1:
            label = parts[0].strip()  # 第一部分是标签（动词短语）
            description = parts[1].strip()  # 第二部分是文本描述
            sentence_text_map[label] = description
        


paste_text_map2 = []

with open('text/pasta02.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        paste_text_map2.append(temp_list)

ucla_paste_text_map0 = []

with open('text/ucla_synonym02.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(',')
        ucla_paste_text_map0.append(temp_list)


ucla_paste_text_map1 = []

with open('text/ucla_pasta02.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        ucla_paste_text_map1.append(temp_list)




def text_prompt():
    text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for c in label_text_map])


    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug,text_dict


def text_prompt_openai_random(use_swap=True, use_paraphrase=False,prob_swap=0.3, prob_t5=0.2):
    """混合增强：同义词+句式变换+原句"""
    print("Use mixed augmentation: synonym + structure transformation")
    total_list = []
    for label_idx, pasta_list in enumerate(paste_text_map0):
        augmented_texts = []
        for raw_text in pasta_list:

            mixed_text = mixed_augment(raw_text, use_swap=use_swap, use_paraphrase=use_paraphrase,prob_swap=prob_swap, prob_t5=prob_t5)

            augmented_texts.append(clip.tokenize(mixed_text))
        total_list.append(augmented_texts)
    return total_list
def text_prompt_openai_random_bert():
    print("Use text prompt openai synonym random bert")
    
    total_list = []
    for pasta_list in paste_text_map0:
        temp_list = []
        for item in pasta_list:

            if torch.rand(1).item() < 0.2:
                item = paraphrase(item)
            temp_list.append(item)
        total_list.append(temp_list)
    return total_list


def text_prompt_openai_pasta_pool_4part(use_paraphrase = True,
                                        prob_t5 = 0.4):
    """
    • paste_text_map2:  List[List[str]]
    • 按 5 种策略拼句子：
        0) row[0]
        1) ','.join(row[:2])
        2) row[0] + ',' + ','.join(row[2:4])
        3) row[0] + ',' + row[4]
        4) row[0] + ',' + ','.join(row[5:])
    • 每条句子以 prob_t5 的概率跑一次 T5 改写
    """
    print("Use text prompt openai pasta pool (T5 only)")
    builders = [
        lambda r: r[0],
        lambda r: ','.join(r[:2]),
        lambda r: r[0] + ',' + ','.join(r[2:4]),
        lambda r: r[0] + ',' + r[4],
        lambda r: r[0] + ',' + ','.join(r[5:])
    ]

    text_dict = {}
    for idx, build in enumerate(builders):

        sentences = [build(row) for row in paste_text_map2]


        if use_paraphrase and prob_t5 > 0:
            sentences = [
                paraphrase(s) if random.random() < prob_t5 else s
                for s in sentences
            ]

        # token 化
        text_dict[idx] = torch.cat([clip.tokenize(s) for s in sentences])

    classes = torch.cat(list(text_dict.values()))
    num_text_aug = len(builders)
    return classes, num_text_aug, text_dict


def text_prompt_openai_random_ucla():
    print("Use mixed augmentation: synonym + structure transformation in UCLA")

    total_list = []
    for label_idx,pasta_list in enumerate(ucla_paste_text_map0):
        augmented_texts = []
        for raw_text in pasta_list:

            mixed_text = mixed_augment(raw_text)

            augmented_texts.append(clip.tokenize(mixed_text))
        total_list.append(augmented_texts)
    return total_list

def text_prompt_openai_pasta_pool_4part_ucla():
    print("Use text prompt openai pasta pool ucla")
    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            pasta_list = [paraphrase(item) if torch.rand(1).item() < 0.2 else item for item in
                          [pasta_list[ii] for pasta_list in ucla_paste_text_map1]]
            text_dict[ii] = torch.cat([clip.tokenize(item) for item in pasta_list])
        elif ii == 1:
            pasta_list = [paraphrase(item) if torch.rand(1).item() < 0.2 else item for item in
                          [','.join(pasta_list[0:2]) for pasta_list in ucla_paste_text_map1]]
            text_dict[ii] = torch.cat([clip.tokenize(item) for item in pasta_list])
        elif ii == 2:
            pasta_list = [paraphrase(item) if torch.rand(1).item() < 0.2 else item for item in
                          [pasta_list[0] + ','.join(pasta_list[2:4]) for pasta_list in ucla_paste_text_map1]]
            text_dict[ii] = torch.cat([clip.tokenize(item) for item in pasta_list])
        elif ii == 3:
            pasta_list = [paraphrase(item) if torch.rand(1).item() < 0.2 else item for item in
                          [pasta_list[0] + ',' + pasta_list[4] for pasta_list in ucla_paste_text_map1]]
            text_dict[ii] = torch.cat([clip.tokenize(item) for item in pasta_list])
        else:
            pasta_list = [paraphrase(item) if torch.rand(1).item() < 0.2 else item for item in
                          [pasta_list[0] + ',' + ','.join(pasta_list[5:]) for pasta_list in ucla_paste_text_map1]]
            text_dict[ii] = torch.cat([clip.tokenize(item) for item in pasta_list])
    
    classes = torch.cat([v for k, v in text_dict.items()])
    
    return classes, num_text_aug, text_dict




