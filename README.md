<h2 align="center">[JSTARS 2026] Controllable Reference-Guided Diffusion with Local–Global Fusion for Real-World Remote Sensing Image Super-Resolution</h2>

<p align="center">
<a href="http://ieeexplore.ieee.org/document/11355810">
📄 Paper Link
</a>
</p>

<p align="center">
<strong>Ce Wang</strong><sup>1</sup>, 
<strong>Wanjie Sun</strong><sup>1</sup>
</p>

<p align="center">
<sup>1</sup> School of Remote Sensing and Information Engineering, Wuhan University
</p>


</div>

<p align="center">
    <img src="assets/arch.png" style="border-radius: 15px">
</p>

---

## 📚 Table of Contents

* [Visual Results](#visual_results)
* [Installation](#installation)
* [Pretrained Models](#pretrained_models)
* [Dataset](#dataset)
* [Train](#train)
* [Inference](#inference)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

---

## <a name="visual_results"></a>👁️ Visual Results

### Results Real-RefRSSRD

<img src="assets/visual_results/compare.png"/>

### For High Spatialtempoal Image Generation

<img src="assets/visual_results/spa-temp-sys.png"/>

### Results of Global-Local Control

<img src="assets/visual_results/control.png"/>

---

## <a name="installation"></a>⚙️ Installation

```bash
# Clone this repository
git clone https://github.com/wwangcece/CRefDiff.git

# Create a conda environment with Python >= 3.9
conda create -n CRefDiff python=3.9
conda activate CRefDiff

# Install required packages
pip install -r requirements.txt
```

---

## <a name="pretrained_models"></a>🧬 Pretrained Models

Download the pretrained models from the link below and place them in the `checkpoints/` directory:

[Download from HuggingFace](https://huggingface.co/wangcce/RefSR_x10)

---

## <a name="dataset"></a>📊 Dataset
### Geographic Coordinate Sampling Points
<p align="center">
    <img src="assets/sample_points.png" style="border-radius: 15px">
</p>

### Data Samples
<p align="center">
    <img src="assets/real_refsr_dataset.png" style="border-radius: 15px">
</p>

1. Refer to the [Real-RefRSSRD](https://huggingface.co/datasets/wangcce/Real-RefRSSRD) for downloading.
2. Use the script `dataset/prepare_lr.py` to upscale the LR images to match the size of HR.

---

## <a name="train"></a>:stars:Train
Firstly load pretrained SD parameters:
```bash
python scripts/init_weight_refsr.py \
--cldm_config configs/model/refsr_dino.yaml \
--sd_weight checkpoints/v2-1_512-ema-pruned.ckpt \
--output checkpoints/init_weight/init_weight-refsr.pt
```
Secondly please modify the training configuration files at configs/train_refsr.yaml.
Finally you can start training:
```bash
python train.py \
--config configs/train_refsr.yaml
```

## <a name="inference"></a>⚔️ Inference

1. Modify the validation dataset configuration in `configs/dataset/reference_sr_test.yaml` and update the pretrained model path in `inference_refsr_batch.py`.
2. Run the inference script:

```bash
python inference_refsr_batch.sh --ckpt path/to/pretrained/model --output path/tp/out/dir --global_ref_scale 1 --device cuda:0 
```

---

---

## <a name="citation"></a>📖 Citation

If you find this work helpful, please consider citing:

```bibtex
@misc{wang2025controllablereferencebasedrealworldremote,
      title={Controllable Reference-Based Real-World Remote Sensing Image Super-Resolution with Generative Diffusion Priors}, 
      author={Ce Wang and Wanjie Sun},
      year={2025},
      eprint={2506.23801},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.23801}, 
}
```

---

## <a name="acknowledgements"></a>🙏 Acknowledgements

This project is based on [DiffBIR](https://github.com/XPixelGroup/DiffBIR). We thank the authors for their excellent work.

---

## <a name="contact"></a>📨 Contact

If you have any questions, feel free to reach out to:
**Ce Wang** — [cewang@whu.edu.cn](mailto:cewang@whu.edu.cn)
