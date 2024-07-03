
<div align="center">

# Investigating the Role of Instruction Variety and Task Difficulty in Robotic Manipulation Tasks


<a href="https://www.python.org/"><img alt="Python 3.11" src="https://img.shields.io/badge/Python 3.11-blue?logo=python&logoColor=white"></a>
<a href="https://pdm-project.org/en/latest/"><img alt="PDM" src="https://img.shields.io/badge/PDM-AC75D7?logo=pdm&logoColor=white"></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://lightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=lightning&logoColor=white"></a>
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright - checked](https://img.shields.io/badge/basedpyright-checked-42b983)](https://detachhead.github.io/basedpyright)
[![CI](https://github.com/amitkparekh/CoGeLoT/actions/workflows/ci.yml/badge.svg)](https://github.com/amitkparekh/CoGeLoT/actions/workflows/ci.yml)

[[arXiv]](https://arxiv.org/abs/1234.56789) | [[Training Data]](https://huggingface.co/datasets/amitkparekh/vima) | [[Checkpoints]](https://huggingface.co/amitkparekh/cogelot)


</div>

---

Evaluating the generalisation capabilities of multimodal models based solely on their performance on out-of-distribution data fails to capture their true robustness. This work introduces a comprehensive evaluation framework that systematically examines the role of instructions and inputs in the generalisation abilities of such models, considering architectural design, input perturbations across language and vision modalities, and increased task complexity. The proposed framework uncovers the resilience of multimodal models to extreme instruction perturbations and their vulnerability to observational changes, raising concerns about overfitting to spurious correlations. By employing this evaluation framework on current Transformer-based multimodal models for robotic manipulation tasks, we uncover limitations and suggest future advancements should focus on architectural and training innovations that better integrate multimodal inputs, enhancing a model's generalisation prowess by prioritising sensitivity to input content over incidental correlations.

<br/>

## Our Evaluation Framework


![Table of perturbations from the paper](docs/PERT%20Table.png)

<div align='center'>
<small><i>Our evaluation framework. Each perturbation affects the instruction or observation inputs, which can be linguistic, visual, or a combination of both. The plausibility of a perturbation relates to a model's expected performance. Sensitivity to unreasonable conditions (:heavy_multiplication_x:) indicates that a model should not perform the task successfully given the perturbation, while plausible perturbations (:heavy_check_mark:) suggest that it should still perform successfully.</i></small>
</div>



## What is included?

Everything. You should be able to run every single experiment from the paper. Datasets and models are hosted on HF.

While I tried to bring everything front and centre, some things might be buried. If you think these things should be brought forward, feel free to open a PR and bring them forward! I'll definitely be taking opinions regarding this into consideration for future projects.

Additionally, I've tried to work in a constrained, clean, and robust manner. I hope that helps you as much as it helped me.


> [!NOTE]
> This project is codenamed `cogelot` so that's what the library is called to prevent needing to rewrite everything.


### Limitations on further development

> [!WARNING]
> This is a research project, and so the scope of it is limited to the paper it is attached to.

It is unlikely that I will maintaining this repository to ensure it continues to work with the latest dependencies outside those pinned. That said, I've tried to be lax in my pinning of deps so that building on this project will not be overly restrictive.




### Model Architectures and Checkpoints

All model checkpoints are stored on Hugging Face, but *they will not work with the Transformers library out-of-the-box*. This library contains multiple methods and functions to make the checkpoints work on our framework, but it's all included for you.

Below is a table of each model run and where to find the checkpoints. We're providing the checkpoint stored at the end of each training epoch. I detail how I ran things in a later section.


| Instruction-style | Instruction Modalities | Prompt-conditioning | Vision Encoder | Shuffled Objects? | Model ID | Experiment ID |
|:--|:--|:--|:--|:--:|:--:|:---|
| Original | Text + Visual | Cross-Attention | Object-Centric | False | [`8lkml12g`](https://huggingface.co/amitkparekh/cogelot/tree/main/8lkml12g) | `01_their_vima` |
| Original | Text + Visual | Cross-Attention | Object-Centric | True | [`ftwoyjb1`](https://huggingface.co/amitkparekh/cogelot/tree/main/ftwoyjb1) | `01_their_vima_shuffle_obj` |
| Original | Text + Visual | Cross-Attention | Image-Patches | N/A | [`ln4nrqhg`](https://huggingface.co/amitkparekh/cogelot/tree/main/ln4nrqhg) | `01_their_vima_patches` |
| Original | Text + Visual | Concatenate | Object-Centric | False | [`bhuja4vo`](https://huggingface.co/amitkparekh/cogelot/tree/main/bhuja4vo) | `08_their_gpt` |
| Original | Text + Visual | Concatenate | Object-Centric | True | [`wn9jc5l8`](https://huggingface.co/amitkparekh/cogelot/tree/main/wn9jc5l8) | `08_their_gpt_shuffle_obj` |
| Original | Text + Visual | Concatenate | Image-Patches | N/A | [`efxugme9`](https://huggingface.co/amitkparekh/cogelot/tree/main/efxugme9) | `08_their_gpt_patches` |
| Paraphrases | Text + Visual | Cross-Attention | Object-Centric | False | [`2df3mwfn`](https://huggingface.co/amitkparekh/cogelot/tree/main/2df3mwfn) | `02_their_vima` |
| Paraphrases | Text + Visual | Cross-Attention | Object-Centric | True | [`0nsnkaer`](https://huggingface.co/amitkparekh/cogelot/tree/main/0nsnkaer) | `02_their_vima_shuffle_obj` |
| Paraphrases | Text + Visual | Cross-Attention | Image-Patches | N/A | [`ah5btw8w`](https://huggingface.co/amitkparekh/cogelot/tree/main/ah5btw8w) | `02_their_vima_patches` |
| Paraphrases | Text + Visual | Concatenate | Object-Centric | False | [`fs5v61mz`](https://huggingface.co/amitkparekh/cogelot/tree/main/fs5v61mz) | `09_their_gpt` |
| Paraphrases | Text + Visual | Concatenate | Object-Centric | True | [`xb3yttg9`](https://huggingface.co/amitkparekh/cogelot/tree/main/xb3yttg9) | `09_their_gpt_shuffle_obj` |
| Paraphrases | Text + Visual | Concatenate | Image-Patches | N/A | [`zby6xk27`](https://huggingface.co/amitkparekh/cogelot/tree/main/zby6xk27) | `09_their_gpt_patches` |




## How I ran things


> [!IMPORTANT]
> **Everything that was run, in some shape or form, starts from a module in `src/cogelot/entrypoints/`.** This is what was used to run the dataset creation, train models, evaluate models, and more. Everything I ran started from that folder, every single time.


This is not a comprehensive library made for all use cases and every possible scenario. It's a research project. That said, I tried to make everything as clear as possible for you. In this section, I detailed how I did everything so that you can use it as an example for how to start yourself.

I have tried to make sure that docstrings and comments are relevant and detailed. If you want more information on what a function is doing or why it is doing that, feel free to make an issue. If you figure out something that I haven't described enough of, feel free to make a PR improving my documentation so that you, me, _and_ future people can benefit from your insight.



### How I managed and installed dependencies


I used [PDM](https://pdm-project.org/en/latest/) to manage this project. Everything you need to know about the installing the dependencies for this project can be found in the `pyproject.toml`.

To quickly install and get up and running, you can run the following:

```bash
pdm install
```


<details>
<summary><b>What if you use <code>requirements.txt</code>?</b></summary>

I have exported and included the `requirements.txt` from PDM. Using it is up to you. I'm not going to be maintaining it, but it's there if you need it.

</details>

<details>
<summary><b>How I install dependencies on every machine</b></summary>

I literally just run the following on the machines I use. I don't use Windows though so I can't help you there.

```bash
mise use python@3.11 pdm@latest
pdm install
```
</details>


<details>
<summary><b>How to make sure it works on your machine</b></summary>

The quickest way to make sure you're all setup is to run either of the following:

- If you know you've got a venv activated or something
    ```bash
    python -m cogelot
    ```

- If you're using PDM instead of activating the venv
    ```bash
    pdm run python -m cogelot
    ```

</details>



### How I checked that everything worked before I ran things

Things happen and things break. I needed a sense check to make sure everything worked. I developed everything using tests to verify that each pieces works in isolation and together. This is the first thing I did when using a new machine or node or whatever.

You can find all the tests in the `tests/` folder. The various tests are a good way of looking how different pieces were implemented and are used. While coverage is not 100%, I used the tests with breakpoints to verify things are working as expected.

<details>
<summary><b>How to make sure all tests can be loaded without errors</b></summary>

```bash
pdm run pytest --deselect tests/test_online_evaluation.py --collect-only
```

This is also useful for just making sure things installed correctly and that all tests can be found.

</details>

<details>
<summary><b>How to run all the tests</b></summary>

```bash
pdm run pytest --deselect tests/test_online_evaluation.py
```

I've also got CI doing this so you can check the badge at the top of the README to see if everything is working as expected.

Check out [pytest-xdist](https://github.com/pytest-dev/pytest-xdist) if you want to know more about running tests in parallel, or just throw ` -n auto` on the end of the above commands. It makes it go faster.[^1]

[^1]: I don't know what happens if you replace `auto` with a number that has more processes than your machine. Maybe don't do that.

</details>


> [!TIP]
> Before spawning an instance and starting to train with GPUs, you can run the above command on your machine to make sure everything works on CPU. As Lightning handles all of the GPU communication, if it works on CPU, there's a 99% chance it'll work on GPU.[^2]


[^2]: This number is made up, but I'm pretty sure about it.



### How I trained models

> [!IMPORTANT]
> **All model training was orchestrated with [Hydra](https://hydra.cc/), and can be found in the `configs/` folder.**

I went all out on the Hydra stuff and everything is pretty compositional. The `configs/experiments` sub-dir contains the experiments that were run (and directly connect to the checkpoints table). As a result, if you want to just train a model, you can run:


```bash
pdm run python src/cogelot/entrypoints/train.py --experiment=01_their_vima
```


> [!TIP]
> You can find the experiments in the folder, or check the `Experiment ID` column in the [above table](#model-architectures-and-checkpoints) for what each one means since the names aren't the clearest.



<details>
<summary><b>Training on different hardware</b></summary>

The `configs/hardware` folder contains the hardware configurations that were used to run the experiments. These are used to set the number of GPUs, the number of CPUs, and the memory available to the model. These were preset for the cluster I was using, but you can adjust them to your needs.

</details>



<details>
<summary><b>How to train models on OCI</b></summary>

This was a while ago now, but I had a setup script which you can find at `scripts/setup-oci-a100.sh`. This was used to setup the environment on the OCI instance I was using. It's not perfect, but it's a good starting point.

</details>



<details>
<summary><b>How I trained models on K8s</b></summary>

Running on K8s was a bit more involved but it's all here. That said, it will be different for your setups.

My pod spec was:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: &name cogelot-1
  namespace: ???
spec:
  restartPolicy: Never
  containers:
    - name: 1st
      image: amitkparekh/python-pdm-cuda:latest
      envFrom:
        - secretRef:
            name: amit-cogelot
      imagePullPolicy: Always
      command: ["/bin/bash", "-c"]
      args:
        - gh repo clone amitkparekh/cogelot cogelot &&
          cd cogelot &&
          bash ./scripts/setup-eidf.sh 2>&1 | tee setup-eidf.log &&
          sleep infinity
          #bash ./scripts/run-sweep-4.sh
      resources:
        requests:
          cpu: &num-cpu 10
          memory: &num-memory "150Gi"
          nvidia.com/gpu: &num-gpu 4
        limits:
          cpu: *num-cpu
          memory: *num-memory
          nvidia.com/gpu: *num-gpu
      volumeMounts:
        - mountPath: /mnt/ceph_rbd
          name: volume
          # this is necessary for training in distributed mode - used for different processes to communicate
        - mountPath: /dev/shm
          name: dshm1
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-A100-SXM4-40GB
  volumes:
    - name: volume
      persistentVolumeClaim:
        claimName: *name
    - name: dshm1
      emptyDir:
        medium: Memory
```


The Dockerfile is public and the secrets contained the following:

```
WANDB_API_KEY=???
HUGGING_FACE_HUB_TOKEN=???
GH_TOKEN=???

HF_HUB_VERBOSITY=info

WANDB_CONFIG_DIR=/mnt/ceph_rbd/wandb
WANDB_CACHE_DIR=/mnt/ceph_rbd/wandb
HF_HOME=/mnt/ceph_rbd/huggingface
TORCH_HOME=/mnt/ceph_rbd/torch
```

</details>




### How I ran checkpoints in the environment

Again, this uses Hydra so, like training, the entrypoint is `src/cogelot/entrypoints/evaluate.py` and the config for it is `configs/evaluate.yaml`.

To run the evaluation in the environment, I used the following command:

```bash
pdm run python src/cogelot/entrypoints/evaluate.py trainer.devices=1 model.model.wandb_run_id=8lkml12g
```


<details>
<summary><b>How to choose your checkpoint</b></summary>

`model.model.wandb_run_id` parameter is important used to get the checkpoint to evaluate. The checkpoint ID is the one from the table above.

By default, we use the epoch from the last checkpoint, but if you want to change the epoch, just add `model.model.epoch=<epoch_number>` to the command.

</details>

<details>
<summary><b>How to run multiple runs in parallel</b></summary>

The `trainer.devices` creates multiple CPU processes for evaluation as eval does not need the GPU.
Change the number in the command to however many processes you want.

Important things to note:

1. The more processes/devices you use, the more memeory you need since multiple instantiations of the model are loaded into the memory.
2. I did not do anything fancy with batching across instances. Since we use CPU for evaluation, I didn't need to.

</details>



<details>
<summary><b>How to perturb the instructions</b></summary>

You can find all of these in `configs/evaluation_instance_transform/`. For each file name, you can invoke them by appending `evaluation_instance_transform=<file_name>` to the command.

| Evaluation Instance Transform | Description |
|:--|:--|
| `noop` | Interleave modalities in the prompt, _by default_ |
| `gobbledygook_tokens` | Apply _Gobbledygook Tokens_ to the prompt |
| `gobbledygook_words` | Apply _Gobbledygook Words_ to the prompt |
| `reworded` | Use paraphrased instructions with interleaved modalities |
| `textual` | Convert visual referents to text |
| `textual_gobbledygook_words` | Convert visual referents to text and apply _Gobbledygook Words_ |
| `textual_gobbledygook_tokens` | Convert visual referents to text and apply _Gobbledygook Tokens_ |
| `textual_no_noun` | Convert visual referents to text, but remove the nouns |
| `textual_no_texture` | Convert visual referents to text, but remove the descriptions of nouns |
| `textual_generic_noun` | Convert visual referents to text, but replace each noun with a generic form (e.g. "block" becomes "thing") |

</details>


<details>
<summary><b>How to disable modalities in the prompt</b></summary>


You can find all of these in `configs/evaluation_prompt_modality/`. For each file name, you can invoke them by appending `evaluation_prompt_modality=<file_name>` to the command.

| Evaluation Prompt Modality | Description |
|:--|:--|
| `disable_none` | Do nothing |
| `disable_text` | Disable the text modality |
| `disable_visual` | Disable the visual modality |
| `disable_both` | Disable both modalities, basically masking _every token_ |

</details>




<details>
<summary><b>How to permute object token order for observations</b></summary>


Append `model.should_shuffle_obj_per_observations=true` to the command. This will shuffle the object tokens in the observation.

</details>




<details>
<summary><b>How to run on different difficulties</b></summary>

Append `model.difficulty=<difficulty>` to the command. The difficulties are:

- `easy`
- `medium` _(unused)_
- `hard` _(unused)_
- `extreme`
- `distracting`
- `extremely_distracting`

</details>


<details>
<summary><b>How I ran the checkpoint from VIMA in the environment</b></summary>

I downloaded the checkpoint from [VIMA's repo](https://github.com/vimalabs/VIMA), renamed it to `them.ckpt` and put it in `storage/data/models`. If you want to change the path used, you can change the path in `configs/model/from_their_policy.yaml`.

I used the following command to run the checkpoint from VIMA in the environment:

```bash
SLURM_JOB_NAME='bash' pdm run python src/cogelot/entrypoints/evaluate_theirs.py trainer.devices=20
```

You can use all of the other perturbations mentioned above.

</details>



<details>
<summary><b>How to run checkpoints with a live display</b></summary>

If you want to see what's going on live, you can append `environment@model.environment=display` onto the evaluate command.

Importantly, **only use one process** because I don't know what'll happen if you don't.

Also, this wasn't run on SLURM, just on my Mac. I can't speak for every machine so your mileage may vary.

</details>

<details>
<summary><b>How to evaluate models on SLURM</b></summary>

It is very unlikely that I ran things in a tmux session and just stared at it. I don't like copy-pasting hundreds of commands.

As experiments were often run on a compute cluster, I ran commands with SLURM. You can find these contained batch files in `./scripts/slurm/`. These were made for my system, so some adjustments are likely going to be needed, but I'm hoping it's obvious and not too complicated!

</details>


### How I prepared the dataset

So that things can be run quickly, the dataset was loaded and parsed with Pydantic, and then converted into a [HF dataset](https://huggingface.co/datasets/amitkparekh/cogelot). There are unit tests showing how this was done in `tests/test_dataset_creation.py`.

The dataset was processed in two steps. The first step was to parse the raw data and pickle it into individual files. This was done because it was the most time-consuming part of the process. The second step was to load the pickled files and convert them into a HF dataset.

To make loading data efficient when modelling, all the instances were tokenised in advanced. Similarly, this is also available on HF, as a different config name.


> [!NOTE]
> For each of the following commands, you can append `--help` to get more information on the command and what it does and the various arguments to control it. Alternatively, you can change things using the [Pydantic settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) in `src/cogelot/commmon/settings.py`.
>
> For example, each command has a way of distributing the load to multiple workers, and even splitting them across multiple SLURM jobs to make it go so much faster.


<details>
<summary><b>Step 1. Download the raw data from VIMA</b></summary>

The raw data was downloaded from VIMA. Each instance is a folder of multiple files. Once extracted, the folder structure looked like this:

```
<project_root>
└─ storage/
    └─ data/
        └─ raw/
            └─ vima_v6/
                └─ <task_name>/
                    └─ <instance_id>/
```

I used symlinks to make it easier to manage the data, but this is what the folder structure was. If you want to use a different directory, you can change it using the [Pydantic settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) in `src/cogelot/common/settings.py`.

</details>


<details>
<summary><b>Step 2. Parse the original data</b></summary>

The raw data was parsed and pickled into individual files, and then it was converted into a HF dataset. This is for speed.

```bash
pdm run python -m cogelot parse-original-dataset --replace-if-exists
pdm run python -m cogelot create-raw-dataset-per-task
```

I have separate SBATCH files for these steps:
  - `scripts/slurm/parse-original-dataset.sh`
  - `scripts/slurm/create-raw-dataset.sh`

</details>



<details>
<summary><b>Step 3. Tokenize and preprocess for faster training</b></summary>

Again, we preprocess and just dump each as pickles because it's faster before turning it into the HF dataset

```bash
pdm run python -m cogelot preprocess-instances
pdm run python -m cogelot create-preprocessed-dataset-per-task

```

I have separate SBATCH files for these steps:
  - `scripts/slurm/preprocess-instances.sh`
  - `scripts/slurm/create-preprocessed-dataset.sh`

</details>



<details>
<summary><b>Step 4. Create a dataset variant of paraphrased instructions</b></summary>

We use just the previous instances to make the new variations, and use environment variables to create the preprocessed versions of the dataset.

```bash
pdm run python -m cogelot create-reworded-dataset-per-task original reworded

DATASET_VARIANT=reworded pdm run python -m cogelot preprocess-instances
DATASET_VARIANT=reworded pdm run python -m cogelot create-preprocessed-dataset-per-task
```

Again, I have a SBATCH file for this: `scripts/slurm/create-reworded-dataset.sh`, or more conveniently, a bash script to submit SBATCH jobs: `scripts/submit-reworded-dataset-creation-jobs.sh`.

</details>


<details>
<summary><b>Step 5. Upload all the datasets</b></summary>

This one's just for me but refer to `scripts/submit-dataset-upload-jobs.sh` for uploading all the datasets to HF as fast as possible without hitting the rate limit.

</details>




## License

VIMA, VIMA-Bench, and all artefacts from VIMA are licensed under the MIT License. Everything within this repository continues to be licensed under the MIT License.

## Citation

```bibtex
@misc{parekh2024investigating
year = {2024}
}
```
