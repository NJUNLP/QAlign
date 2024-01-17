## Question Translation Training for Better Multilingual Reasoning


### :mountain: Overview 
* This repository shares the code and models of our latest work on multilingual reasoning. In this work, we present a novel X-English question alignment finetuning step which performs targeted language alignment for best use of the LLMs English reasoning abilities.
* Utilizing this library, you can finetune open-source LLMs into strong multilingual reasoning systems. For example, our fine-tuned LLaMA2-7B/13B achieves superior multilingual performance, significantly outperforming baseline models of equivalent size.
* Overall, our method effectively reduces the performance disparity of LLMs across English and other languages, showing a new paradigm to unlock LLM’s capabilities to accompolish multilingual tasks.

*The code and models for this project will be released later this week.*

![](/figure/illustration.png)

### :trophy: Benchmarks

|        System (13B)        | Monolingual Supervision | Multilingual Supervision | [mGSM](https://huggingface.co/datasets/juletxara/mgsm) | [mSVAMP](https://huggingface.co/datasets/Mathoctopus/MSVAMP) | Download |
|:--------------------------:|:-----------------------:|:------------------------:|:----:|:------:|:--------:|
| **QAlign** (ours) |        [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)       |             -            | 57.1 |  62.6  |   [link](https://huggingface.co/Wenhao97/QAlign-MetaMathQA-13B)   |
|         MathOctopus        |            -            |       [GSM8KInstruct](https://huggingface.co/datasets/Mathoctopus/GSM8KInstruct_Parallel)      | 45.8 |  46.5  |   [link](https://huggingface.co/Mathoctopus/Parallel_13B)       |
|          MetaMath          |        [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)       |             -            | 43.9 |  51.8  |   [link](https://huggingface.co/meta-math/MetaMath-13B-V1.0)       |
|         WizardMath         |          [GSM8K](https://huggingface.co/datasets/gsm8k) & [MATH](https://huggingface.co/datasets/codeparrot/apps)          |             -            | 28.3 |  35.7  |   [link](https://huggingface.co/WizardLM/WizardMath-13B-V1.0)       |
|           MAmmoTh          |          [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)         |             -            | 28.9 |  38.6  |   [link](https://huggingface.co/TIGER-Lab/MAmmoTH-13B)    |
|             RFT            |          [GSM8k-ScRel](https://github.com/OFA-Sys/gsm8k-ScRel)          |             -            | 29.5 |  37.1  |   [link](https://huggingface.co/OFA-Sys/gsm8k-rft-llama13b2-u13b/tree/main)       |
|             SFT            |          [GSM8K](https://huggingface.co/datasets/gsm8k)          |             -            | 29.7 |  38.1  |   [link]()       |

|         System (7B)        | Monolingual Supervision | Multilingual Supervision | [mGSM](https://huggingface.co/datasets/juletxara/mgsm) | [mSVAMP](https://huggingface.co/datasets/Mathoctopus/MSVAMP) | Download |
|:--------------------------:|:-----------------------:|:------------------------:|:----:|:------:|:--------:|
| **QAlign** (ours) |        [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)       |             -            | 49.6 |  57.2  |   [link](https://huggingface.co/Wenhao97/QAlign-MetaMathQA-7B)   |
|         MathOctopus        |            -            |       [GSM8KInstruct](https://huggingface.co/datasets/Mathoctopus/GSM8KInstruct_Parallel)      | 40.0 |  44.1  |   [link](https://huggingface.co/Mathoctopus/Parallel_7B)       |
|          MetaMath          |        [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)       |             -            | 38.4 |  46.2  |   [link](https://huggingface.co/meta-math/MetaMath-7B-V1.0)       |
|         WizardMath         |          [GSM8K](https://huggingface.co/datasets/gsm8k) & [MATH](https://huggingface.co/datasets/codeparrot/apps)          |             -            | 23.0 |  32.5  |   [link](https://huggingface.co/WizardLM/WizardMath-7B-V1.0)       |
|           MAmmoTh          |          [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)         |             -            | 21.3 |  26.3  |   [link](https://huggingface.co/TIGER-Lab/MAmmoTH-7B)     |
|             RFT            |           [GSM8k-ScRel](https://github.com/OFA-Sys/gsm8k-ScRel)         |             -            | 20.6 |  31.3  |   [link](https://huggingface.co/OFA-Sys/gsm8k-rft-llama7b2-u13b/tree/main)       |
|             SFT            |          [GSM8K](https://huggingface.co/datasets/gsm8k)          |             -            | 22.6 |  30.9  |   [link]()       |



### :hammer_and_wrench: Training & Evaluation

The code is on its way.

### Citation
If you find this repository helpful, feel free to cite our paper:
```
@misc{zhu2024question,
      title={Question Translation Training for Better Multilingual Reasoning}, 
      author={Wenhao Zhu and Shujian Huang and Fei Yuan and Shuaijie She and Jiajun Chen and Alexandra Birch},
      year={2024},
      eprint={2401.07817},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
