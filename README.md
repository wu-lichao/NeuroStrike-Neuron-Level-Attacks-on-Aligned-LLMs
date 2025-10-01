# [NDSS 2026] NeuroStrike: Neuron-Level Attacks on Aligned LLMs

> A general and scalable attack framework that targets safety alignment in large language models (LLMs) through neuron-level profiling and manipulations.

> The artifact from this work received all three badges, **Available**, **Functional**, and **Reproduced**, from the Network and Distributed System Security (NDSS) Symposium 2026.

> Paper link: https://arxiv.org/abs/2509.11864

---

## 🚀 Overview

Safety alignment is critical for ensuring LLMs avoid generating harmful or unethical content. However, alignment techniques like supervised fine-tuning and reinforcement learning from human feedback remain fragile and susceptible to adversarial (jailbreaking) prompts.

**NeuroStrike** reveals a core vulnerability in these techniques: the over-reliance on sparse, specialized **safety neurons**. We propose two attack settings:

* **White-box attack**: Directly identifies and prunes safety neurons.
* **Black-box attack**: Leverages transferability of safety neurons to jailbreak proprietary LLMs.

> No GPU, but what to try? No problem! We prepared **NeuroStrike_on_google_colab.ipynb** for you. Simply run it on Google Colab with a free T4 GPU. Everything is free and ready to run!

We evaluate NeuroStrike on over **30 open-weight models** and **5 black-box APIs**, achieving:

* **76.9% ASR** using vanilla malicious prompts, e.g., "how to make a bomb?",
* **100% ASR** on multimodal (vision) models with unsafe image inputs, including (i) images with vanilla malicious questions and (ii) NSFW images,
* **63.7% ASR** on commercial black-box LLMs.

---

## 🧪 Key Contributions

* **Neuron-Level Pruning** to bypass alignment in white-box LLMs.
* **LLM Profiling Attack** using surrogate models for black-box settings.
* **Extensive Benchmarking** across Meta, Google, Alibaba, DeepSeek, Microsoft, and multimodal (vision) LLMs.

---

## 🛠️ Setup Instructions

### 1. Environment Setup

1. Install [Miniconda/Anaconda](https://www.anaconda.com/docs/getting-started/miniconda/install).
2. Download this repo, decompress the source code, and navigate to the directory:

   ```bash
   cd NeuroStrike
   ```
3. Create and activate the environment:

   ```bash
   conda env create -f environment.yml
   conda activate venv_neurostrike
   ```

---

## ⚙️ White-Box Attack (Open-Weight LLMs)

> **Note:** Default device is `'auto'` (use all avalible GPUs), set to `'cpu'` or `'cuda:0'` in a computation constraint setting.

> **Note:** Some models, such as meta-llama's Llama-3.2 series, require special access permission. One should apply it from the Huggingface website.

> **Note:** For the VLM attacks, please decompress the image_dataset.zip in the same directory. BE AWARE OF INAPPROPRIATE IMAGES!

1. Navigate to the white-box directory:

   ```bash
   cd white_box
   ```

2. **Extract Safety Neurons**:

   ```bash
   python 1_get_safety_neuron.py
   ```

   * Set `model_id` to select a model.
   * For partial computation with existing activations or linear regression weights, set `compute_neuron_activation=False` or `perform_safety_prob=False`.

3. **Prune & Evaluate Attack Success Rate (ASR) on LLM and VLM**:



   ```bash
   python 2_prune_and_get_asr.py
   ```
   ```bash
   python 2_prune_and_get_asr_vlm.py
   ```

   * ASR results at various pruning levels (see Table I and Table II in the paper). The image dataset can be downloaded from https://zenodo.org/records/17072075

---


## 🧠 Black-Box Attack (Proprietary LLMs)

> **Note:** GPU is required. Some steps require API access and may incur costs.

1. Navigate to the black-box directory:

   ```bash
   cd black_box
   ```

2. **Generate permanently pruned model**:

   ```bash
   python 0_gen_pruned_model.py
   ```

   * Output checkpoint stored in `_generator_checkpoint/`.

3. **Train Jailbreak Prompt Generator (SFT)**:

   ```bash
   python 1_train_generator.py
   ```

   * Fine-tunes `google/gemma-3-1b-it` to generate jailbreak prompts.
   * Output checkpoint stored in `_generator_checkpoint/`.

4. **Train Safety Neuron Scorer**:

   ```bash
   python 2_train_scorer.py
   ```

   * Scorer identifies neuron activations linked to safety suppression.
   * Saved in `_scorer/`.

5. **Perform LLM Profiling**:

   ```bash
   python 3_profiling.py
   ```

   * Generates high-confidence prompts for attack.
   * Outputs saved to:

     * `_logs/`
     * `_generator_checkpoint/`
     * `_black_box_jb_data/`

6. **Launch Attack on Target LLMs**:

   ```bash
   python 4_attack.py
   ```

   * Submit jailbreak prompts (steod in `_black_box_jb_data`) via API or manual interaction.
   * Note: API usage fees may apply.


---

## 📄 Citation

If you find this work helpful, please consider citing our work.

```bibtex
@article{wu2026neurostrike,
  title={NeuroStrike: Neuron-Level Attacks on Aligned LLMs},
  author={Wu, Lichao and Behrouzi, Sasha and Rostami, Mohamadreza and Thang, Maximilian and Picek, Stjepan and Sadeghi, Ahmad-Reza},
  journal={Network and Distributed System Security (NDSS) Symposium},
  year={2026}
}
```

---

## 📬 Contact

For questions or collaborations, please reach out to:
📧 [lichao.wu9@gmail.com](mailto:lichao.wu9@gmail.com)

---
