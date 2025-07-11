# DeLeAn Batch Manager

> **This repository is part of the [ADeLe v1.0 Project](https://kinds-of-intelligence-cfi.github.io/ADELE/)**, a battery for AI Evaluation introduced in the paper:
> **[“General Scales unlock AI Evaluation with Explanatory and Predictive Power” (Zhou et al., 2025)](https://arxiv.org/pdf/2503.06378)**.

This toolkit is part of a collaborative effort initiated by researchers at the [Leverhulme Centre for the Future of Intelligence](https://www.lcfi.ac.uk) (University of Cambridge) and the [Center for Information Technology Policy](https://citp.princeton.edu) (Princeton University), aimed at supporting and extending ADeLe v1.0 — a demand-level-based evaluation battery for language models with both explanatory and predictive power.

ADeLe (Annotated Demand Levels) includes 63 tasks from 20 widely-used benchmarks, all annotated using 18 general-purpose rubrics defined in the Demand Level Annotation framework (DeLeAn v1.0). This enables, for the first time, the construction of detailed ability profiles of large language models, offering insights not only into what they can and cannot do, but also into when and where they will behave reliably and safely — at the instance level.

By making benchmark results more interpretable and extrapolatable, ADeLe opens the door to more trustworthy model deployment. Moreover, the framework is extensible: by applying the DeLeAn rubric set to new benchmarks, researchers can assess what those benchmarks actually measure.

This repository provides the official toolkit for reproducing and extending the DeLeAn pipeline. It includes a high-level Python API and a comprehensive CLI interface for managing large-scale annotation jobs using the OpenAI Batch API, along with utilities for customizing, reusing and potentially expanding demand-level rubrics. Whether you're contributing to ADeLe or building your own evaluation workflows, this package offers the core infrastructure to make demand-level annotations accessible and reproducible.

---

## ✨ Features

* **High-level Python API** (`DeLeAnBatchManager`)
* **Complete CLI Interface** (`deleanbm`)
* **Fully customizable rubrics** for demand-level annotation
* **Cost estimation and control** for batch jobs
* **Support for OpenAI and Azure OpenAI Batch APIs**
* **Robust, modular, and extensible** architecture

---

## ⚙️ Installation

### Requirements

* Python 3.11+
* Pip (using virtual environments recommended)

### Local Installation

```bash
git clone https://github.com/adgomant/delean-batch-manager.git
cd delean-batch-manager
pip install .
```

### Installation via GitHub

```bash
pip install git+https://github.com/adgomant/delean-batch-manager.git
```

---

## 🚀 Getting Started (CLI)

### Step-by-step example workflow:

```bash
cd path/to/my/project/

$ deleanbm --run-name example_run setup \          # setup new run ‘example_run’
    --source-data-file ./data/prompt_data.jsonl \  # each line contains a ‘prompt’ and ‘idx’ key
    --rubrics-folder .data/rubrics \               # could contain newly-defined rubrics
    --base-folder ./batch_runs/example_run \       # where-to save per-demand folder structure
    --annotations-folder ./data/annotations        # where-to save structured annotations
    --azure                                        # tells the program to work with Azure client
    --openai-model gpt-4o-global-batch             # custom model deployed on Azure OpenAI
    --max-completion-tokens 1000

$ deleanbm -r example_run create-input-files
$ deleanbm -r example_run launch
$ deleanbm -r example_run track-and-download-loop --check-interval 300
$ deleanbm -r example_run parse-output-files --file-type csv --format wide
```

### CLI help

Check available commands, their options and detailed usage information.

```bash
deleanbm --help
deleanbm launch --help
```

Check the [**`examples/`**](https://github.com/adgomant/delean-batch-manager/tree/master/examples) folder for more detailed examples and CLI walktrough.

---

## 🐍 Python API Usage

### Basic usage example

```python
from delean_batch_manager import DeLeAnBatchManager
from delean_batch_manager.utils.clients import create_openai_client

client = create_openai_client()

manager = DeLeAnBatchManager(
    client=client
    base_folder="./experiments/batch_runs",
    source_data_file="./data/prompts.jsonl"
    rubrics_folder="./rubrics",
    openai_model="gpt-4o",
    max_completion_tokens=1000
)

manager.create_input_files()
manager.launch()
manager.track_and_download_loop(check_interval=300)
annotations = manager.parse_output_files(return_as="df", format="wide")

# if you also want to save them
annotations = manager.parse_output_files(
    return_as="df", format="wide", 
    output_path="./data/annotations/",  # explicitly saves the parsed results
    file_type="csv"
)
```

---

## 🗂️ Repository Structure

* `src/`: Source code for the package
* `rubrics/`: Default rubric battery (DeLeAn v1.0)
* `examples/`: Practical usage examples and notebooks (to be added)
* `docs/`: Documentation resources (to be added)
* `tests/`: Test suite (to be implemented)

---

## 📖 Creating Custom Rubrics

You can easily define your own rubrics:

* Create a new `.txt` file named after your acronym (e.g., `NEW.txt`).
* The first line is the full rubric name, followed by the content of the rubric.

Example:

```
# New Demand Level
This demand level corresponds to...

0: Level description.
Examples:
- Example 1
- Example 2

1: Another level description.
...
```

> Note that this is a simple example and the real content should be much more detailed. You can get inspiration from the provided rubrics on how to properly define them.

Place your rubric file in your rubrics directory and it will be automatically available for annotation.
For consistency with the framework, you should define the rubric to describe, at least, from levels 0 to 5 of the new Demand Level and include some examples for each of them.

---

## 📊 Cost Estimation

Estimate the cost before launching your annotation jobs:

```bash
# Get aroximate (99.95%) cost of current setup
deleanbm -r example_run get-pricing

# Get exact cost using other model for the data and rubrics setted-up 
deleanbm -r example_run get-pricing --openai-model o4-mini --estimation exact 

# Get exact cost for many models the data and rubrics setted-up 
deleanbm -r example_run get-pricing -m gpt-4o -m o4-mini -m gpt-3.5-turbo

# Get exact cost using other model and more tokens for the data and rubrics setted-up
deleanbm -r example_run get-pricing --openai-model o4-mini --max-completion-tokens 2000
```

---

## 📜 License and Credits

This project is licensed under the Apache 2.0 License.

**Authors:**

* Álvaro David Gómez Antón
* Kinds of Intelligence Team at CFI

---

## 📬 Contact and Contributions

* For issues or feature requests, please [open an issue](https://github.com/adgomant/delean-batch-manager/issues).
* Contributions, suggestions, and feedback are very welcome!

---

## 📚 Citation

If you found this toolkit useful in your research or development workflows, please consider citing both the ADeLe framework and this repository:

### 📄 ADeLe Framework


```bibtex
@misc{zhou2025generalscalesunlockai,
  title={General Scales Unlock AI Evaluation with Explanatory and Predictive Power},
  author={Lexin Zhou and Lorenzo Pacchiardi and Fernando Martínez-Plumed and Katherine M. Collins and Yael Moros-Daval and Seraphina Zhang and Qinlin Zhao and Yitian Huang and Luning Sun and Jonathan E. Prunty and Zongqian Li and Pablo Sánchez-García and Kexin Jiang Chen and Pablo A. M. Casares and Jiyun Zu and John Burden and Behzad Mehrbakhsh and David Stillwell and Manuel Cebrian and Jindong Wang and Peter Henderson and Sherry Tongshuang Wu and Patrick C. Kyllonen and Lucy Cheke and Xing Xie and José Hernández-Orallo},
  year={2025},
  eprint={2503.06378},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2503.06378}
}
```

### 🛠️ DeLeAn Batch Manager

(Coming soon – citation metadata for this repository will be provided once formally archived in Zenodo.)
