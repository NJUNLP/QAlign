# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import json
import pdb
from iso639 import languages
import os
import re
import datasets
logger = datasets.logging.get_logger(__name__)

_INSTRUCTIONS = [
    "Translate the following sentences from {source_lang} to {target_lang}.", 
]

class TranslationDataConfig(datasets.BuilderConfig):
    """BuilderConfig for TranslationData."""

    def __init__(self, config: str, **kwargs):
        """BuilderConfig for TranslationData.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TranslationDataConfig, self).__init__(**kwargs)
        self.dataset, lang = config.rsplit("_", maxsplit=1)
        self.source_lang, self.target_lang = lang.split("-")


class TranslationData(datasets.GeneratorBasedBuilder):
    """This is an adapter for loading raw text parallel corpus."""
    BUILDER_CONFIG_CLASS = TranslationDataConfig

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "instruction": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "output": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        base_path = os.path.join(self.base_path, self.config.dataset, f"{self.config.source_lang}-{self.config.target_lang}")
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(base_path, "train")}),
            # datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(base_path, "test")}),
            # datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(base_path, "validation")}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0

        if self.config.dataset == 'gsm8kinstruct_question':
            if self.config.target_lang == 'all':
                source_name = languages.get(alpha2=self.config.source_lang).name
                lang_pool = ['en', 'sw', 'zh', 'bn', 'de', 'es', 'fr', 'ja', 'ru', 'th']
                lang_pool.remove(self.config.source_lang)
                for target_lang in lang_pool:
                    target_name = languages.get(alpha2=target_lang).name

                    source_path='/cpfs01/user/zhuwenhao/open-source/QAlign/code/data/gsmtrans/gsm8kinstruct_question/en-{}/train.{}'.format(target_lang, self.config.source_lang)
                    target_path='/cpfs01/user/zhuwenhao/open-source/QAlign/code/data/gsmtrans/gsm8kinstruct_question/en-{}/train.{}'.format(target_lang, target_lang)

                    if os.path.exists(source_path) and os.path.exists(target_path):
                        with open(source_path, encoding="utf-8") as source_f, open(target_path, encoding="utf-8") as target_f:
                            for source_line, target_line in zip(source_f, target_f):
                                if len(source_line.strip()) > 0 and len(target_line.strip()) > 0:
                                    yield key, {
                                        "id": key,
                                        "instruction": _INSTRUCTIONS[0].format_map({
                                            "source_lang": source_name, 
                                            "target_lang": target_name, 
                                        }) + ' ' + source_line.strip(),
                                        "input": "",
                                        "output": target_line.strip(),
                                    }
                                key += 1
            elif self.config.source_lang == 'all':
                target_lang = self.config.target_lang
                target_name = languages.get(alpha2=target_lang).name
                lang_pool = ['en', 'sw', 'zh', 'bn', 'de', 'es', 'fr', 'ja', 'ru', 'th']
                lang_pool.remove(target_lang)
                for source_lang in lang_pool:
                    source_name = languages.get(alpha2=source_lang).name

                    source_path='/cpfs01/user/zhuwenhao/open-source/QAlign/code/data/gsmtrans/gsm8kinstruct_question/{}-{}/train.{}'.format(source_lang, target_lang, source_lang)
                    target_path='/cpfs01/user/zhuwenhao/open-source/QAlign/code/data/gsmtrans/gsm8kinstruct_question/{}-{}/train.{}'.format(source_lang, target_lang, target_lang)

                    if os.path.exists(source_path) and os.path.exists(target_path):
                        with open(source_path, encoding="utf-8") as source_f, open(target_path, encoding="utf-8") as target_f:
                            for source_line, target_line in zip(source_f, target_f):
                                if len(source_line.strip()) > 0 and len(target_line.strip()) > 0:
                                    yield key, {
                                        "id": key,
                                        "instruction": _INSTRUCTIONS[0].format_map({
                                            "source_lang": source_name, 
                                            "target_lang": target_name, 
                                        }) + ' ' + source_line.strip(),
                                        "input": "",
                                        "output": target_line.strip(),
                                    }
                                key += 1

            else:
                source_name, target_name = (languages.get(alpha2=lang).name for lang in [self.config.source_lang, self.config.target_lang])

                if os.path.exists(f"{filepath}.{self.config.source_lang}") and os.path.exists(f"{filepath}.{self.config.target_lang}"):
                    with open(f"{filepath}.{self.config.source_lang}", encoding="utf-8") as source_f, open(f"{filepath}.{self.config.target_lang}", encoding="utf-8") as target_f:
                        for source_line, target_line in zip(source_f, target_f):
                            if len(source_line.strip()) > 0 and len(target_line.strip()) > 0:
                                yield key, {
                                    "id": key,
                                    "instruction": _INSTRUCTIONS[0].format_map({
                                        "source_lang": source_name, 
                                        "target_lang": target_name, 
                                    }) + ' ' + source_line.strip(),
                                    "input": "",
                                    "output": target_line.strip(),
                                }
                            key += 1

        elif self.config.dataset == 'gsm8kinstruct_answer':
            if self.config.target_lang == 'all':
                source_name = languages.get(alpha2=self.config.source_lang).name
                lang_pool = ['en', 'sw', 'zh', 'bn', 'de', 'es', 'fr', 'ja', 'ru', 'th']
                lang_pool.remove(self.config.source_lang)
                for target_lang in lang_pool:
                    target_name = languages.get(alpha2=target_lang).name

                    source_path='/cpfs01/user/zhuwenhao/open-source/QAlign/code/data/gsmtrans/gsm8kinstruct_answer/en-{}/train.{}'.format(target_lang, self.config.source_lang)
                    target_path='/cpfs01/user/zhuwenhao/open-source/QAlign/code/data/gsmtrans/gsm8kinstruct_answer/en-{}/train.{}'.format(target_lang, target_lang)

                    if os.path.exists(source_path) and os.path.exists(target_path):
                        with open(source_path, encoding="utf-8") as source_f, open(target_path, encoding="utf-8") as target_f:
                            for source_line, target_line in zip(source_f, target_f):
                                source_line = json.loads(source_line)
                                target_line = json.loads(target_line)
                                if len(source_line.strip()) > 0 and len(target_line.strip()) > 0:
                                    yield key, {
                                        "id": key,
                                        "instruction": _INSTRUCTIONS[0].format_map({
                                            "source_lang": source_name, 
                                            "target_lang": target_name, 
                                        }) + ' ' + source_line.strip(),
                                        "input": "",
                                        "output": target_line.strip(),
                                    }
                                key += 1

            elif self.config.source_lang == 'all':
                target_name = languages.get(alpha2=self.config.target_lang).name
                lang_pool = ['en', 'sw', 'zh', 'bn', 'de', 'es', 'fr', 'ja', 'ru', 'th']
                lang_pool.remove(self.config.target_lang)
                for source_lang in lang_pool:
                    source_name = languages.get(alpha2=source_lang).name

                    source_path='/cpfs01/user/zhuwenhao/open-source/QAlign/code/data/gsmtrans/gsm8kinstruct_answer/{}-en/train.{}'.format(source_lang, source_lang)
                    target_path='/cpfs01/user/zhuwenhao/open-source/QAlign/code/data/gsmtrans/gsm8kinstruct_answer/{}-en/train.{}'.format(source_lang, self.config.target_lang)

                    if os.path.exists(source_path) and os.path.exists(target_path):
                        with open(source_path, encoding="utf-8") as source_f, open(target_path, encoding="utf-8") as target_f:
                            for source_line, target_line in zip(source_f, target_f):
                                source_line = json.loads(source_line)
                                target_line = json.loads(target_line)
                                if len(source_line.strip()) > 0 and len(target_line.strip()) > 0:
                                    yield key, {
                                        "id": key,
                                        "instruction": _INSTRUCTIONS[0].format_map({
                                            "source_lang": source_name, 
                                            "target_lang": target_name, 
                                        }) + ' ' + source_line.strip(),
                                        "input": "",
                                        "output": target_line.strip(),
                                    }
                                key += 1

            else:
                source_name, target_name = (languages.get(alpha2=lang).name for lang in [self.config.source_lang, self.config.target_lang])

                if os.path.exists(f"{filepath}.{self.config.source_lang}") and os.path.exists(f"{filepath}.{self.config.target_lang}"):
                    with open(f"{filepath}.{self.config.source_lang}", encoding="utf-8") as source_f, open(f"{filepath}.{self.config.target_lang}", encoding="utf-8") as target_f:
                        for source_line, target_line in zip(source_f, target_f):
                            source_line = json.loads(source_line)
                            target_line = json.loads(target_line)
                            if len(source_line.strip()) > 0 and len(target_line.strip()) > 0:
                                yield key, {
                                    "id": key,
                                    "instruction": _INSTRUCTIONS[0].format_map({
                                        "source_lang": source_name, 
                                        "target_lang": target_name, 
                                    }) + ' ' + source_line.strip(),
                                    "input": "",
                                    "output": target_line.strip(),
                                }
                            key += 1