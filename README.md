# BCPL: Boundary-Confusion Prototype Learning

## Hard-Sample-Aware Steel Surface Defect Detection via Boundary-Confusing Sample Clustering

BCPL is a hard-sample-aware optimization framework for steel surface defect detection. It focuses on boundary-confusing samples, namely samples near inter-class decision boundaries where visually similar defect categories are easily misidentified.

Unlike conventional hard-sample mining methods that reweight isolated samples according to confidence, loss, gradient, or matching quality, BCPL explicitly models the latent structure of boundary-confusing samples. It first discovers local boundary-confusion groups, summarizes them with representative prototypes, and then uses these prototypes to guide detector training through staged dynamic loss re-weighting.

## Overview

BCPL consists of two coupled modules:

1. **Adaptive Clustering-based Boundary Prototype Learning (AC-BPL)**  
   AC-BPL selects high-uncertainty samples using joint Margin and Entropy criteria, groups boundary-confusing samples through adaptive density clustering, and extracts a Medoid from each cluster as a boundary prototype.

2. **Boundary-aware Dynamic Loss Re-weighting (BCDR)**  
   BCDR appends the discovered boundary prototypes as boundary-prototype output dimensions to the learned class-center projector. It then compares original-class responses and boundary-prototype responses to estimate boundary hardness and convert it into staged dynamic loss weights for downstream detector training.

This design changes hard-sample optimization from sample-level emphasis to prototype-level boundary-structure modeling.

## Method Pipeline

### Stage 1: Class-Center Learning

BCPL uses a DINO-pretrained ViT-B/16 backbone and a class-center projector to learn discriminative defect representations. Only the last Transformer block of the DINO ViT-B/16 backbone is fine-tuned.

The class-center learning objective combines:

- supervised classification loss for label-aware semantic anchoring;
- supervised contrastive loss for intra-class compactness and inter-class separation;
- unsupervised contrastive loss for cross-view consistency;
- clustering regularization with self-distillation and maximum-entropy constraints.

Two augmented views are generated for each image during representation learning.

### Stage 2: Boundary Prototype Discovery with AC-BPL

After class-center learning, all training samples are scored by the learned projector. Boundary candidates are selected by the intersection of:

- low-Margin samples, where the top two class responses are close;
- high-Entropy samples, where the prediction distribution is broadly uncertain.

The selected samples are clustered with adaptive density clustering. For each discovered boundary-confusion cluster, BCPL selects the Medoid, a real sample closest to other samples in the same cluster, as the boundary prototype.

### Stage 3: Boundary-Aware Detector Training with BCDR

The learned class-center projector is expanded with boundary-prototype output dimensions. The original dimensions describe original defect-class responses, and the appended dimensions describe boundary-prototype responses.

For each training sample, BCDR compares:

- the strongest original-class response;
- the strongest boundary-prototype response.

Their relative strength is mapped to a dynamic classification loss weight. The weight is introduced with a staged schedule: a warmup stage, a transition stage, and a final boundary-aware training stage.

## Key Results

### Main Detection Results

The paper evaluates BCPL on three steel surface defect datasets and three representative detectors. Results are reported as mAP@50. Values in parentheses denote percentage-point changes over the corresponding detector baseline.

| Method | NEU-DET Hard | NEU-DET Overall | Severstal Hard | Severstal Overall | MaSteel Hard | MaSteel Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s | 0.324 | 0.671 | 0.369 | 0.393 | 0.828 | 0.671 |
| YOLOv8s+BCPL | **0.357 (+3.3%)** | **0.692 (+2.1%)** | **0.409 (+4.0%)** | **0.399 (+0.6%)** | **0.854 (+2.6%)** | **0.692 (+2.1%)** |
| YOLOv26s | 0.356 | 0.688 | 0.598 | 0.495 | 0.848 | 0.701 |
| YOLOv26s+BCPL | **0.407 (+5.1%)** | **0.705 (+1.7%)** | **0.606 (+0.8%)** | 0.492 (-0.3%) | **0.863 (+1.5%)** | **0.712 (+1.1%)** |
| RF-DETRS | 0.357 | 0.672 | 0.556 | 0.441 | 0.858 | 0.712 |
| RF-DETRS+BCPL | **0.374 (+1.7%)** | **0.688 (+1.6%)** | **0.572 (+1.6%)** | **0.452 (+1.1%)** | **0.872 (+1.4%)** | **0.727 (+1.5%)** |

### Comparison with Hard-Sample Methods

The comparison with representative hard-sample and quality-aware reweighting methods is conducted on NEU-DET with YOLOv8s.

| Method | Venue | Hard-class mAP@50 | Overall mAP@50 |
| --- | --- | ---: | ---: |
| YOLOv8s Origin | -- | 0.324 | 0.671 |
| YOLOv8s + Focal Loss | ICCV 2017 | 0.326 (+0.2%) | 0.667 (-0.4%) |
| YOLOv8s + OHEM | CVPR 2016 | 0.339 (+1.5%) | 0.673 (+0.2%) |
| YOLOv8s + LRM Loss | IJCNN 2018 | 0.314 (-1.0%) | 0.680 (+0.9%) |
| YOLOv8s + Quality Focal Loss | NeurIPS 2020 | 0.335 (+1.1%) | 0.688 (+1.7%) |
| YOLOv8s + Varifocal Loss | CVPR 2021 | 0.336 (+1.2%) | **0.697 (+2.6%)** |
| YOLOv8s + TF Loss | IEEE TGRS 2025 | 0.310 (-1.4%) | 0.654 (-1.7%) |
| YOLOv8s + MAL | CVPR 2025 | 0.319 (-0.5%) | 0.674 (+0.3%) |
| YOLOv8s + BCPL | Ours | **0.357 (+3.3%)** | 0.692 (+2.1%) |

These results show that BCPL achieves the best hard-class mAP among the compared methods, supporting the paper's main claim that explicit boundary-confusion structure modeling is more effective than directly reweighting individual hard samples.

## Datasets

The experiments use three steel surface defect datasets:

- **NEU-DET**: a benchmark steel surface defect dataset with six defect classes, 300 images per class, and 200x200 grayscale images.
- **Severstal-Steel-Defect**: an industrial strip-steel defect dataset with about 12,568 training images and pixel-level annotations covering four major defect categories and their combinations.
- **MaSteel**: a private production-line steel defect dataset annotated by quality-inspection experts.

The paper reports both overall mAP@50 over all classes and hard-class mAP@50 over visually ambiguous defect categories.

## Experimental Details

For class-center learning, the paper uses:

- DINO ViT-B/16 as the pretrained backbone;
- input image size of 224;
- feature dimension of 768;
- a 3-layer projection head;
- SGD optimizer with initial learning rate 0.1, momentum 0.9, weight decay 1e-4;
- 100 training epochs with cosine annealing.

BCPL-specific hyperparameters are fixed across datasets and detectors:

- class-center learning loss weights: lambda_cls = 0.65, lambda_cluster = 0.35, lambda_unsup = 0.35, lambda_sup = 0.65;
- boundary selection: alpha_M = 0.1 and alpha_H = 0.1;
- adaptive density clustering: q = 0.3, lower epsilon bound = 0.05, fallback radius multipliers = 1.5, 2.0, and 3.0;
- BCDR weight mapping: alpha = 5, beta = 3, eta = 1.0;
- detector training: 300 epochs, staged activation starts at epoch 30 and uses a 60-epoch transition stage.

## Usage

The public code repository is maintained at:

```text
https://github.com/445pop/BCPL
```

The implementation follows the paper's three-stage workflow:

1. learn class-center representations;
2. mine boundary-confusing samples and extract Medoid prototypes;
3. train the detector with boundary-aware dynamic loss re-weighting.

Please configure dataset paths and detector settings according to your local environment before running experiments.

## Citation

This manuscript is currently under review. A formal citation entry will be provided after publication.

For now, please cite the work as:

```text
Boundary-Confusion Prototype Learning (BCPL), manuscript under review.
```

## Acknowledgements

This work builds on prior research in self-supervised representation learning, prototype learning, density-based clustering, and hard-sample-aware object detection. The implementation also benefits from open-source detector and representation-learning codebases.

## License

Please refer to the repository license file for usage terms.
