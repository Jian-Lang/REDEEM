# REDEEM
This repo is the official implementation of _REDEEMing Modality Information Loss: Retrieval-Guided Conditional Generation for Severely Modality Missing Learning_ accepted by KDD 2025. 

This work, [REDEEM](https://jianlang.org/assets/papers/KDD-2025-REDEEM.pdf), is an extension of my previous work on [RAGPT](https://doi.org/10.1609/aaai.v39i17.33984), which is accepted by AAAI 2025. The main improvements lie in: 

+ More fine-grained missing modality reconstruction through a Conditional MoE Generator, which leverages the remaining modalities as conditions to guide the aggregation of retrieved experts. 

+ Inter-modal dynamic prompting, which generates sample-specific prompts from retrieved instances to guide pretrained multimodal transformers in understanding the semantic correspondence across modalities. This helps compensate for missing modalities in the current input by providing contextual cross-modal cues.

## Abstract
Traditional multimodal learning approaches often assume that all modalities are available during both the training and inference phases. However, this assumption is often impractical in real-world scenarios due to challenges such as sensor failures, data corruption, or privacy concerns. While recent efforts focus on enhancing the robustness of pre-trained Multimodal Transformers (MTs) under missing modality conditions, they often overlook reconstructing the missing modalities and rely on static, sample-agnostic prompt-tuning techniques, undermining their efficacy in severe modality missing scenarios. To address these limitations, we propose REDEEM, a novel REtrieval-guiDEd conditional gEnerative fraMework that largely alleviates the modality missing problems on pre-trained MTs. REDEEM consists of a new adaptive retrieval mechanism to identify relevant instances for both modality-complete and -incomplete samples. It then conditions on the remaining modalities and utilizes the retrieved data as experts to effectively recover the missing ones in modality-incomplete instances. Finally, REDEEM generates sample-aware dynamic prompts from the retrieved instances to guide MTs in tackling severe modality missing challenges. Comprehensive experiments on three diverse multimodal classification benchmarks demonstrate that REDEEM significantly outperforms competitive baselines.

## Framework

<img width="884" alt="image" src="https://github.com/user-attachments/assets/b10d37b3-3ab3-4025-81d4-5d68418910da" />


## Environment Configuration

First, clone this repo:

```shell
git clone https://github.com/Jian-Lang/REDEEM.git

cd REDEEM
```

First, create a new conda env for REDEEM:

```shell
conda create -n REDEEM python=3.9
```

Next, activate this env and install the dependencies from the requirements.txt:

```shell
conda activate REDEEM

pip install -r requirements.txt
```

## Data Preparation

### MM-IMDb

First, download the dataset from this link: https://archive.org/download/mmimdb/mmimdb.tar.gz

Then, place the raw images in folder **dataset/mmimdb/image** and put the json files in folder **dataset/mmimdb/meta_data**.

### HateMemes

First, download the dataset from this link: https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset

Then, place the raw images in folder **dataset/hatememes/image** and put the json files in folder **dataset/hatememes/metadata**.

Next, replace the **test.json** in metadata with **test_seen.json** downloaded from this link: https://www.kaggle.com/datasets/williamberrios/hateful-memes as the test.json downloaded from the prior website has no label information for evaluation. (Do not change other files, only replace the test.json with test_seen.json)

### Food101

First, download the dataset from this link: https://www.kaggle.com/datasets/gianmarco96/upmcfood101

Then, place the raw images in folder **dataset/food101/image** and put the csv files in folder **dataset/food101/meta_data**.

## Code Running

### Dataset Initiation

Run the following script to init the dataset:

```shell
sh src/scripts/init_data.sh
```

### Training & Evaluation

Run the following script to training our model and evaluate the results:

```shell
sh src/scripts/eval.sh
```

All the parameters have the same meaning as describe in our paper and you can simply config them in **src/config/config.yaml** or in command line.

## Citation

If you find the code useful for your research, please give us a star ⭐⭐⭐ and consider citing:

```
@inproceedings{lang2025redeeming,
    author = {Lang, Jian and Hong, Rongpei and Cheng, Zhangtao and Zhong, Ting and Wang, Yong and Zhou, Fan},
    booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
    year = {2025},
    pages = {1241--1252},
    doi = {10.1145/3711896.3737101},
    title = {REDEEMing Modality Information Loss: Retrieval-Guided Conditional Generation for Severely Modality Missing Learning},
}
```
