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

import pdb
import csv
from iso639 import languages
import os
import re
import datasets
import json
import random
logger = datasets.logging.get_logger(__name__)

_INSTRUCTIONS = {
}

_CITATION = """\
"""

_DESCRIPTION = """\
"""

class GSM8KInstructDataConfig(datasets.BuilderConfig):
    """BuilderConfig for TranslationData."""

    def __init__(self, config: str, **kwargs):
        """BuilderConfig for TranslationData.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(GSM8KInstructDataConfig, self).__init__(**kwargs)
        # self.lang, self.lang = config.rsplit("_", maxsplit=1)
        self.lang = config


class GSM8KInstructData(datasets.GeneratorBasedBuilder):
    """This is an adapter for loading raw text parallel corpus."""
    BUILDER_CONFIG_CLASS = GSM8KInstructDataConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "instruction": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "output": datasets.Value("string"),
                }
            ),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        base_path = os.path.join(self.base_path, f"{self.config.lang}")
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(base_path, "train")}),
            # datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(base_path, "test")}),
            # datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(base_path, "validation")}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0

        if self.config.lang == 'all': 
            filepath = "/cpfs01/user/zhuwenhao/open-source/QAlign/code/data/gsm8kinstruct/{}/train.txt"
            for lang in ['sw', 'en', 'zh', 'bn', 'de', 'es', 'fr', 'ja', 'ru', 'th']:
                with open(filepath.format(lang), encoding="utf-8") as f:
                    for line in f:
                        line = json.loads(line)
                        question, answer = line['question'], line['answer']
                        yield key, {
                            "id": key,
                            "instruction": question,
                            "input": "",
                            "output": answer,
                        }
                        key += 1

        else:
            with open(f"{filepath}.txt", encoding="utf-8") as f:
                for line in f:
                    line = json.loads(line)
                    question, answer = line['question'], line['answer']
                    yield key, {
                        "id": key,
                        "instruction": question,
                        "input": "",
                        "output": answer,
                    }
                    key += 1