# Pose Estimation Datasets Reference

This document catalogs popular pose estimation datasets for animals and humans that can be used with the SuperAnimal behavior analysis pipeline.

## Animal Pose Datasets

### 1. AP-10K (Animal Pose - 10K)
- **Size**: 10,015 images, 60 species, 23 animal families
- **Keypoints**: 17 keypoints per animal
- **Format**: COCO format
- **License**: Apache 2.0
- **Download**: https://github.com/AlexTheBad/AP-10K
- **Paper**: [AP-10K: A Benchmark for Animal Pose Estimation in the Wild](https://arxiv.org/abs/2108.12617)
- **Use case**: General animal pose, cross-species transfer learning

### 2. APT-36K / APTv2 (Animal Pose Tracking)
- **Size**: 2,400+ video clips, 30 species, 36,000+ frames
- **Keypoints**: 17 keypoints (eyes, nose, neck, tail, shoulders, elbows, knees, hips, paws)
- **Format**: COCO format with tracking IDs
- **License**: CC BY-NC 4.0
- **Download**: https://github.com/pandorgan/APT-36K
- **Paper**: [APT-36K: A Large-scale Benchmark for Animal Pose Estimation and Tracking](https://arxiv.org/abs/2206.05683)
- **Use case**: Multi-animal tracking, temporal analysis

### 3. Animal3D
- **Size**: 3,379 images, 40 mammal species
- **Keypoints**: 26 keypoints + 3D SMAL mesh parameters
- **Format**: Custom JSON + SMAL
- **Download**: https://xujiacong.github.io/Animal3D/
- **Paper**: [Animal3D: A Comprehensive Dataset of 3D Animal Pose and Shape](https://arxiv.org/abs/2308.11737)
- **Use case**: 3D pose and shape estimation

### 4. DeepLabCut Model Zoo (SuperAnimal)
- **Models**: TopViewMouse, Quadruped, Fish, etc.
- **Pre-trained**: Yes, ready to use
- **Species**: 45+ species supported
- **Paper**: [SuperAnimal pretrained pose estimation models](https://www.nature.com/articles/s41467-024-48792-2)
- **Use case**: Zero-shot pose estimation, this pipeline's default

### 5. MacaquePose
- **Size**: 13,000+ images of macaque monkeys
- **Keypoints**: 17 keypoints
- **Download**: https://www.pri.kyoto-u.ac.jp/datasets/macaquepose/
- **Use case**: Primate behavior analysis

### 6. AnimalWeb
- **Size**: 22,451 faces, 334 species
- **Focus**: Animal face landmarks
- **Download**: https://fdmaproject.wordpress.com/animalweb/
- **Use case**: Face/head pose analysis

### 7. MARS (Multi-Animal Resident-Intruder Social)
- **Size**: Multi-animal social behavior dataset
- **Keypoints**: 7 keypoints × 2 mice (top-view 2D)
- **Classes**: 4 behavior classes (attack, mount, investigation, other)
- **Format**: NPZ (aligned keypoint sequences)
- **License**: CC BY 4.0
- **Download**: https://data.caltech.edu/records/s0vdx-0k302
- **Paper**: [MARS: Multi-Animal Resident-Intruder Social Interaction](https://elifesciences.org/articles/63720)
- **Use case**: Social behavior classification, multi-animal interaction analysis

### 8. UCLA Mouse Dataset
- **Size**: Mouse behavior recordings
- **Keypoints**: Variable (depends on annotation)
- **Classes**: Multiple behavior categories
- **Format**: Video + annotations
- **Use case**: Mouse behavior analysis, action recognition demo

---

## Human Pose Datasets

### 1. COCO Keypoints
- **Size**: 250,000+ person instances with keypoints
- **Keypoints**: 17 body keypoints
- **Format**: COCO JSON
- **License**: CC BY 4.0
- **Download**: https://cocodataset.org/#download
- **Use case**: Industry standard, benchmarking

### 2. COCO-WholeBody
- **Size**: Same as COCO + additional annotations
- **Keypoints**: 133 keypoints (17 body + 6 feet + 68 face + 42 hands)
- **Download**: https://github.com/jin-s13/COCO-WholeBody
- **Use case**: Full body + face + hands estimation

### 3. MPII Human Pose
- **Size**: 25,000 images, 40,000+ people
- **Keypoints**: 16 body keypoints
- **Activities**: 410 human activity categories
- **Download**: http://human-pose.mpi-inf.mpg.de/
- **License**: Non-commercial only
- **Use case**: Activity recognition, diverse poses

### 4. PoseTrack
- **Size**: 1,356 video sequences
- **Keypoints**: 15 body keypoints
- **Download**: https://posetrack.net/
- **Use case**: Multi-person tracking in videos

### 5. Human3.6M
- **Size**: 3.6 million frames, 11 subjects
- **Keypoints**: 32 joints (3D)
- **Format**: Proprietary
- **Download**: http://vision.imar.ro/human3.6m/
- **License**: Research only (requires agreement)
- **Use case**: 3D pose estimation

### 6. Leeds Sports Pose (LSP)
- **Size**: 2,000 images
- **Keypoints**: 14 keypoints
- **Focus**: Sports activities
- **Download**: https://sam.johnson.io/research/lsp.html
- **Use case**: Sports pose analysis

---

## Dataset Selection Guide

| Use Case | Recommended Dataset | Notes |
|----------|-------------------|-------|
| Mouse behavior | DeepLabCut TopViewMouse | Pre-trained, ready to use |
| Quadruped (dog, cat) | DeepLabCut Quadruped | Pre-trained, 39 keypoints |
| Multi-animal social behavior | MARS | 7 keypoints × 2 mice, 4 behavior classes |
| Mouse action recognition | UCLA Mouse Dataset | Demo-ready behavior categories |
| Multi-species comparison | AP-10K | Unified keypoint format |
| Human pose | COCO Keypoints | Industry standard |
| Full body + face + hands | COCO-WholeBody | 133 keypoints |
| Video tracking | APT-36K or PoseTrack | Temporal consistency |
| 3D reconstruction | Animal3D or Human3.6M | Requires 3D annotations |

---

## Keypoint Mapping

For cross-species comparison, we map keypoints to a unified schema:

### Common Keypoints (Cross-Species)
| ID | Name | Animal | Human |
|----|------|--------|-------|
| 0 | nose | nose | nose |
| 1 | center | mouse_center / back_middle | hip_center |
| 2 | head | head_midpoint | head |
| 3 | tail_base | tail_base | - |
| 4 | left_front | left_shoulder | left_shoulder |
| 5 | right_front | right_shoulder | right_shoulder |
| 6 | left_back | left_hip | left_hip |
| 7 | right_back | right_hip | right_hip |

---

## Download Instructions

### Automated Download
```bash
# Download all configured datasets
python -m src.scripts.download_datasets

# Download specific dataset
python -m src.scripts.download_datasets --dataset ap10k
python -m src.scripts.download_datasets --dataset coco-pose
```

### Manual Download
See individual dataset websites for manual download instructions.

---

## References

1. Yu, H., et al. "AP-10K: A Benchmark for Animal Pose Estimation in the Wild." NeurIPS 2021.
2. Yang, Y., et al. "APT-36K: A Large-scale Benchmark for Animal Pose Estimation and Tracking." NeurIPS 2022.
3. Xu, J., et al. "Animal3D: A Comprehensive Dataset of 3D Animal Pose and Shape." ICCV 2023.
4. Ye, S., et al. "SuperAnimal pretrained pose estimation models for behavioral analysis." Nature Communications 2024.
5. Lin, T.Y., et al. "Microsoft COCO: Common Objects in Context." ECCV 2014.
6. Andriluka, M., et al. "2D Human Pose Estimation: New Benchmark and State of the Art Analysis." CVPR 2014.
7. Segalin, C., et al. "The Mouse Action Recognition System (MARS) software pipeline for automated analysis of social behaviors in mice." eLife 2021.
