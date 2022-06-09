[**中文说明**](https://github.com/zjunlp/OpenUE/blob/main/README.md) | [**English**](https://github.com/zjunlp/OpenUE/blob/main/README_EN.md)

<p align="center">
    <a href="https://github.com/zjunlp/openue"> <img src="https://github.com/zjunlp/OpenUE/blob/main/imgs/logo.png" width="400"/></a>
</p>

<p align="center">
<strong> OpenUE is a lightweight toolkit for knowledge graph extraction. 
    </strong>
</p>
    <p align="center">
    <a href="https://badge.fury.io/py/openue">
        <img src="https://badge.fury.io/py/openue.svg">
    </a>
    <a href="https://github.com/zjunlp/OpenUE/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/zjunlp/openue.svg?color=green">
    </a>
        <a href="http://openue.zjukg.org">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
</p>

[OpenUE](https://aclanthology.org/2020.emnlp-demos.1/) is a lightweight knowledge graph extraction tool.

**Features**


  - Knowledge extraction task based on pre-training language model (compatible with pre-training models such as BERT and Roberta.)
    - Named Entity Extraction
    - Event Extraction
    - Slot filling and intent detection
    - <em> more tasks </em>
  - Training and testing interface
  - fast deployment of your extraction models

## Environment

  - python3.8
  - requirements.txt


## Architecture

![框架](./imgs/overview1.png)

It mainly includes **three** modules, as `models`,`lit_models` and `data`.

### models module

It stores our three main models, the relationship recognition model for the single sentence, the named entity recognition model for the relationship in the known sentence, and the inference model that integrates the first two. It is mainly derived from the defined pre-trained models in the `transformers` library.

### lit_models module

The code is mainly inherited from `pytorch_lightning.Trainer`. It can automatically build model training under different hardware such as single card, multi-card, GPU, TPU, etc. We define `training_step` and `validation_step` in it to automatically build training logic fo