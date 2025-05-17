# REDEEM
This repo is the official implementation of _REDEEMing Modality Information Loss: Retrieval-Guided Conditional Generation for Severely Modality Missing Learning_ accepted by KDD 2025. 

This work, REDEEM, is an extension of my previous work on [RAGPT](https://doi.org/10.1609/aaai.v39i17.33984), which is accepted by AAAI 2025. The main improvements lie in: 

(1) More fine-grained missing modality reconstruction through a Conditional MoE Generator, which leverages the remaining modalities as conditions to guide the aggregation of retrieved experts. 

(2) Inter-modal dynamic prompting, which generates sample-specific prompts from retrieved instances to guide pretrained multimodal transformers in understanding the semantic correspondence across modalities. This helps compensate for missing modalities in the current input by providing contextual cross-modal cues.

## Abstract
Traditional multimodal learning approaches often assume that all modalities are available during both the training and inference phases. However, this assumption is often impractical in real-world scenarios due to challenges such as sensor failures, data corruption, or privacy concerns. While recent efforts focus on enhancing the robustness of pre-trained Multimodal Transformers (MTs) under missing modality conditions, they often overlook reconstructing the missing modalities and rely on static, sample-agnostic prompt-tuning techniques, undermining their efficacy in severe modality missing scenarios. To address these limitations, we propose REDEEM, a novel REtrieval-guiDEd conditional gEnerative fraMework that largely alleviates the modality missing problems on pre-trained MTs. REDEEM consists of a new adaptive retrieval mechanism to identify relevant instances for both modality-complete and -incomplete samples. It then conditions on the remaining modalities and utilizes the retrieved data as experts to effectively recover the missing ones in modality-incomplete instances. Finally, REDEEM generates sample-aware dynamic prompts from the retrieved instances to guide MTs in tackling severe modality missing challenges. Comprehensive experiments on three diverse multimodal classification benchmarks demonstrate that REDEEM significantly outperforms competitive baselines.

## Framework

<img width="884" alt="image" src="https://github.com/user-attachments/assets/b10d37b3-3ab3-4025-81d4-5d68418910da" />

