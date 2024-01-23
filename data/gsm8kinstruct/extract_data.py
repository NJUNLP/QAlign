import json
import pdb
from collections import defaultdict
from langcodes import Language

data_path = "./MGSM8KInstruct_Parallel.json"
LANG = ['Swahili', 'English', 'Chinese', 'Bengali', 'German', 'Spanish', 'French', 'Japanese', 'Russian', 'Thai']
LANG_MAP = {'Swahili': 'sw', 'English': 'en', 'Chinese': 'zh', 'Bengali': 'bn', 'German': 'de', 'Spanish': 'es', 'French': 'fr', 'Japanese': 'ja', 'Russian': 'ru', 'Thai': 'th'}

# load json file
data = []
with open(data_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

lang2question = {lang: [] for lang in LANG}
lang2answer = {lang: [] for lang in LANG}
for d in data:
    answer = d['chosen'].strip()
    prompt = d['prompt'].strip()
    question = ((prompt.split('Instruction:\n'))[-1].split('\n\n')[0]).strip()

    for lang in LANG:
        if lang in prompt:
            lang2question[lang].append(question)
            lang2answer[lang].append(answer)
            break

for lang in LANG:
    assert len(lang2question[lang]) == len(lang2answer[lang])

    l = LANG_MAP[lang]
    with open('./{}/train.txt'.format(l), 'w') as f:
        for question, answer in zip(lang2question[lang], lang2answer[lang]):
            json.dump({'question': question, 'answer': answer}, f)
            f.write('\n')