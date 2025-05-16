# RAGPT
This repo is the official implementation of _REDEEMing Modality Information Loss: Retrieval-Guided Conditional Generation for Severely Modality Missing Learning_ accepted by KDD 2025. 

## Abstract
Traditional multimodal learning approaches often assume that all modalities are available during both the training and inference phases. However, this assumption is often impractical in real-world scenarios due to challenges such as sensor failures, data corruption, or privacy concerns. While recent efforts focus on enhancing the robustness of pre-trained Multimodal Transformers (MTs) under missing modality conditions, they often overlook reconstructing the missing modalities and rely on static, sample-agnostic prompt-tuning techniques, undermining their efficacy in severe modality missing scenarios. To address these limitations, we propose REDEEM, a novel REtrieval-guiDEd conditional gEnerative fraMework that largely alleviates the modality missing problems on pre-trained MTs. REDEEM consists of a new adaptive retrieval mechanism to identify relevant instances for both modality-complete and -incomplete samples. It then conditions on the remaining modalities and utilizes the retrieved data as experts to effectively recover the missing ones in modality-incomplete instances. Finally, REDEEM generates sample-aware dynamic prompts from the retrieved instances to guide MTs in tackling severe modality missing challenges. Comprehensive experiments on three diverse multimodal classification benchmarks demonstrate that REDEEM significantly outperforms competitive baselines.

## Framework


