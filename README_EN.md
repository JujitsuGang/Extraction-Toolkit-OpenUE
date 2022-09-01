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

The code is mainly inherited from `pytorch_lightning.Trainer`. It can automatically build model training under different hardware such as single card, multi-card, GPU, TPU, etc. We define `training_step` and `validation_step` in it to automatically build training logic for training. 

Because its hardware is not sensitive, we can call the OpenUE training module in a variety of different environments.

### data module

The code for different operations on different data sets is stored in `data`. The `tokenizer` in the `transformers` library is used to segment the data and then turn the data into the features we need according to different datasets.

## Quick start

### Install

#### Anaconda 

```
conda create -n openue python=3.8
conda activate openue
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia # depend on your GPU driver version
python setup.py install
```

#### pip

```shell
pip install openue
```

#### pip dev

```shell
python setup.py develop
```

#### How to use

The data format is a `json` file, the specific example is as follows. (in the ske dataset)

```json
{
	"text": "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部",
	"spo_list": [{
		"predicate": "出生地",
		"object_type": "地点",
		"subject_type": "人物",
		"object": "圣地亚哥",
		"subject"