# AGENT.md

This file is for coding agents working inside this repository.

Start with `README.md`, then use this document as the fast path to the real code structure.

## What This Repo Is

AffectGPT is a multimodal emotion understanding project built around:

- a trainable multimodal LLM wrapper in `my_affectgpt/`
- dataset-specific loaders for MER-Caption+ and MER-UniBench in `my_affectgpt/datasets/`
- inference and benchmark scripts at the repo root
- older auxiliary code in `toolkit/` used mostly by preprocessing, evaluation, and utility functions

The current training entrypoint is `train.py`.

## Top-Level Map

- `train.py`: main training entrypoint
- `train_configs/`: YAML configs for training and inference settings
- `inference_hybird.py`: batch inference over benchmark datasets
- `inference_sample.py`: single-video inference
- `evaluation.py`: full evaluation pipeline
- `evaluation-scoreonly.py`: score calculation from saved inference outputs
- `config.py`: hard-coded local paths for models, datasets, and special multimodal tokens
- `my_affectgpt/`: the actual model, task, runner, processors, dataset builders, and inference chat wrapper
- `toolkit/`: utility functions, preprocessing helpers, older dataloaders, and evaluation helpers
- `output/`: saved checkpoints and benchmark inference outputs

## Training Flow From `train.py`

The runtime path is:

1. `train.py` parses `--cfg-path` and optional `--options`.
2. `my_affectgpt.common.config.Config` loads the YAML and merges six sections: `run`, `model`, `datasets`, `inference`, `grpo`, `rewards`.
3. `Config.build_runner_config()` also sets `run.output_dir` to `output/<config_basename>`.
4. `train.py` creates `job_id` as `<config_basename>_<timestamp>` and calls `init_distributed_mode(cfg.run_cfg)`.
5. `tasks.setup_task(cfg)` resolves `cfg.run.task` through the registry. Current training tasks are `video_text_pretrain` and `video_text_grpo`.
6. `task.build_datasets(cfg)` resolves each dataset name through the builder registry.
7. `task.build_model(cfg)` resolves `cfg.model.arch`, which is currently `affectgpt`.
8. The configured runner drives the full epoch loop and checkpoint saving. Current runners are `runner_base` and `runner_grpo`.

Important detail: the YAML can say `distributed: True`, but `init_distributed_mode()` turns it off unless `RANK/WORLD_SIZE/LOCAL_RANK` or SLURM env vars exist. Plain `python train.py ...` therefore runs in non-distributed mode.

## Two Training Paths

This repo now has two parallel training paths.

- SFT path:
  `train.py -> Config -> task=video_text_pretrain -> dataset builder -> AffectGPT.forward() -> RunnerBase.train()`
- GRPO path:
  `train.py -> Config -> task=video_text_grpo -> builder=mer2025ov_grpo -> AffectGPT.generate_group()/compute_response_logprobs() -> GRPORunner.train_epoch()`

Keep them separate when editing. The GRPO code reuses `AffectGPT` as policy, but it does not go through the SFT `forward(samples)["loss"]` path.

## Current Main Training Config

The default README training path points to:

- `train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml`

That config currently means:

- task: `video_text_pretrain`
- model arch: `affectgpt`
- dataset: `mercaptionplus`
- input mode: `multiface_audio_face_text`
- label mode: `hybird` (the typo is intentional in code and config)
- fusion: `attention` for video, audio, and multimodal fusion
- image fusion: `mean`
- query token counts: all set to `1`
- max epoch: `60`
- iters per epoch: `5000`
- AMP enabled

There is also a frame-based single-video inference config:

- `train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml`

There is now also a GRPO debug config:

- `train_configs/mer2025ov_grpo_debug.yaml`

That config currently means:

- task: `video_text_grpo`
- runner: `runner_grpo`
- model arch: `affectgpt`
- dataset: `mer2025ov_grpo`
- question type: `ovlabel`
- reward stack: `debug_length_reward`
- clipped-ratio objective with `clip_range: 0.2`
- `kl_coef: 0`, so the ref model path is disabled entirely

## How Datasets Are Built

Dataset construction goes through:

- `my_affectgpt/tasks/base_task.py`
- `my_affectgpt/datasets/builders/__init__.py`
- `my_affectgpt/datasets/builders/image_text_pair_builder.py`
- `my_affectgpt/datasets/builders/grpo_builder.py`

For the default training config:

- dataset key `mercaptionplus` maps to `MERCaptionPlus_Builder`
- that builder instantiates `MERCaptionPlus_Dataset`

For the GRPO config:

- dataset key `mer2025ov_grpo` maps to `MER2025OVGRPOBuilder`
- that builder instantiates `MER2025OV_GRPO_Dataset`
- this is separate from the existing inference-oriented `MER2025OV_Dataset`

`MERCaptionPlus_Dataset` reads:

- OV labels from `track2_train_mercaptionplus.csv`
- descriptions from `track3_train_mercaptionplus.csv`
- subtitles from `subtitle_chieng.csv`
- raw video from `../dataset/mer2025-dataset/video`
- raw audio from `../dataset/mer2025-dataset/audio`
- face features from `../dataset/mer2025-dataset/openface_face`

Those paths are not discovered dynamically. They come from hard-coded maps in `config.py`.

`MER2025OV_GRPO_Dataset` reads:

- labeled OV supervision from `config.PATH_TO_LABEL['MER2025OV']`
- subtitles from the same hard-coded subtitle sources used by `MER2025OV`
- multimodal payloads from the same video/audio/face roots already used elsewhere

Its purpose is different from SFT datasets:

- it builds prompt-only RL samples
- it returns `reward_meta` instead of teacher-forced targets
- it is the dataset behind the GRPO rollout path

## What A Dataset Sample Looks Like

`my_affectgpt/datasets/datasets/base_dataset.py` does most of the heavy lifting.

For each sample it:

- decides which supervision to use via `label_type`
- for `hybird`, randomly chooses one target type per sample
- loads the needed modalities based on `face_or_frame`
- builds a prompt with placeholders such as `<FaceHere>`, `<AudioHere>`, `<MultiHere>`
- expands those placeholders by repeating the special token string `num_*_query_token` times
- tokenizes prompt and target separately
- masks prompt tokens with `IGNORE_INDEX = -100`
- returns tensors plus modality payloads

For `MERCaptionPlus`, the two supervision targets are:

- `description`
- `ovlabel`

The prompt is always instruction-style text, and the target is appended with `###`.

For GRPO datasets, the batch shape is different:

- `prompt_texts`: list of prompt-only strings
- `reward_metas`: structured reward metadata per sample
- modality tensors such as `frames`, `faces`, `audios`, `images`
- no `labels`
- no teacher-forced target text

## Important Dataset Side Effects

`BaseDataset.__init__()` has non-trivial debug behavior:

- it immediately samples three random items
- prints decoded prompt and label text
- runs `collater()` on those samples
- prints training sample count

This means dataset construction is not cheap and may fail early if files are missing or broken. Do not assume dataset `__init__` is side-effect free.

## Processors and Raw Modalities

The training pipeline uses:

- `alpro_video_train` from `my_affectgpt/processors/video_processor.py`
- `blip_caption`
- `blip2_image_train`

Video and face loading use:

- `load_video()`
- `load_face()`

Audio loading uses:

- `my_affectgpt.models.ImageBind.data.load_audio`
- `transform_audio`

The default training path expects pre-extracted face arrays in `.npy` format. The main README checkpoint also assumes this.

## Model Construction

The core model is `my_affectgpt/models/affectgpt.py`.

`AffectGPT.from_config()`:

- reads encoder and LLM names from YAML
- builds the model
- optionally overlays up to three checkpoints in priority order: `ckpt`, `ckpt_2`, `ckpt_3`

The default stack is:

- LLM: `Qwen25` -> `models/Qwen2.5-7B-Instruct`
- visual encoder: `CLIP_VIT_LARGE`
- acoustic encoder: `HUBERT_LARGE`

Training behavior:

- base LLM weights are frozen
- LoRA adapters are attached to the LLM attention and MLP projections
- visual and acoustic backbones are frozen
- trainable parts depend on config flags, but usually include LoRA, fusion modules, and projection layers

`AffectGPT` now serves both SFT and GRPO. Besides `forward(samples)`, the key GRPO-facing methods are:

- `encode_modalities(samples)`
- `build_prompt_input_ids(prompt_texts, max_length)`
- `build_inputs_embeds_from_prompt_ids(input_ids, modal_llm_dict)`
- `generate_group(samples, generation_cfg, group_size)`
- `compute_response_logprobs(samples, prompt_input_ids, response_token_ids)`

Do not move multimodal placeholder replacement back into chat wrappers or dataset code. `AffectGPT` is now the single source of truth for prompt embedding injection.

## Forward Pass Summary

`AffectGPT.forward(samples)` does this:

1. Encode available modalities:
   `frames`, `faces`, `audios`, `images`
2. Optionally build a multimodal fused representation when `<MultiHere>` appears.
3. Replace multimodal placeholder token embeddings inside the tokenized prompt with encoded modality embeddings.
4. Call the LLM with `inputs_embeds`, `attention_mask`, and `labels`.
5. Return causal LM loss only.

This is a pure next-token training objective. There is no separate classification head.

For GRPO, the same model does this instead:

1. Tokenize prompt-only text with `build_prompt_input_ids()`.
2. Encode modalities with `encode_modalities()`.
3. Inject modality embeddings into prompt token embeddings.
4. Sample `group_size` responses with `generate_group()`.
5. Re-score sampled responses with `compute_response_logprobs()`.

`generate_group()` uses `return_dict_in_generate=True` and slices responses from `outputs.sequences` using `len(outputs.scores)`. This was added to reduce dependence on older HuggingFace generate output conventions.

## Runner Behavior

Training is driven by `my_affectgpt/runners/runner_base.py`.

Key behavior:

- output dir is `output/<config_basename>/<job_id>`
- `log.txt` stores the full config and per-epoch stats
- an initial checkpoint is saved before epoch 1 as epoch 0 with loss `0.000`
- each later epoch saves a checkpoint named like `checkpoint_000030_loss_0.602.pth`
- checkpoints only store trainable parameters, optimizer state, scaler state, config, and epoch

Because only trainable params are saved, successful inference always also requires the original base models under `models/`.

The train loop uses:

- AdamW with two parameter groups: decay and no-decay
- `linear_warmup_cosine_lr` by default
- `iters_per_epoch` from YAML, not from dataset length
- `IterLoader`, so the dataloader is effectively infinite inside each epoch

GRPO training is driven by `my_affectgpt/runners/runner_grpo.py`.

That runner does this per iteration:

1. get a prompt-only RL batch
2. call `task.rollout_step()` to sample grouped responses
3. call the configured reward stack
4. compute group-normalized advantages
5. compute actor logprobs on sampled responses
6. optionally compute ref logprobs
7. compute clipped-ratio GRPO loss
8. backward, gradient clip, optimizer step

Important implementation detail:

- if `grpo.kl_coef > 0`, `runner_grpo` lazily builds `self.ref_model`, moves it to device, and uses it for KL
- if `grpo.kl_coef == 0`, the runner never touches `self.ref_model`, so no ref model is instantiated and no ref-model GPU memory is consumed

This behavior is intentional and is currently the expected way to disable KL regularization for pure policy optimization runs.

GRPO logging adds:

- `reward`
- `kl`
- `ratio`
- `response_len`

Checkpoint format still follows `RunnerBase`: only trainable parameters plus optimizer/scaler/config metadata are saved.

## Reward System

Rewards are now registered through `my_affectgpt/common/registry.py`.

Current components:

- `my_affectgpt/rewards/base_reward.py`
- `my_affectgpt/rewards/composite_reward.py`
- `my_affectgpt/rewards/debug_reward.py`

Current debug rewards:

- `debug_constant_reward`
- `debug_length_reward`

`CompositeReward` is the orchestrator. It instantiates each reward from `rewards.items` in YAML, runs them in order, and returns the weighted sum plus a structured breakdown.

If you are adding a real reward, align it to the `BaseReward.score(samples, responses, reward_metas)` interface instead of hard-coding reward logic inside the GRPO task.

## Validation And Testing During Training

The default training config only declares:

- `train_splits: ["train"]`

So `RunnerBase.train()` does not run in-training validation by default.
Benchmark inference and scoring are handled later by the standalone scripts.

## Inference And Evaluation Structure

The repository separates training from benchmark evaluation:

- `inference_hybird.py` loads trained checkpoints and runs generation on MER-UniBench datasets
- `inference_sample.py` does single-sample generation
- `evaluation.py` and `evaluation-scoreonly.py` post-process `.npz` outputs into task metrics

Inference reuses dataset classes directly to resolve file paths and prompts. It does not go through the training runner.

## Path And Naming Landmines

Keep these in mind before changing anything:

- `hybird` is misspelled in configs and code on purpose; changing only one side will break things
- `inference_hybird.py` is also intentionally misspelled in filename
- dataset/model roots are hard-coded in `config.py`
- the repo expects model weights under `models/` and datasets under `../dataset/`
- several modules import everything for side-effect registration; deleting wildcard imports can break registry lookup

## Files To Read First For Common Tasks

If you are changing training behavior:

- `train.py`
- `train_configs/*.yaml`
- `my_affectgpt/common/config.py`
- `my_affectgpt/common/registry.py`
- `my_affectgpt/tasks/base_task.py`
- `my_affectgpt/tasks/video_text_grpo.py`
- `my_affectgpt/runners/runner_base.py`
- `my_affectgpt/runners/runner_grpo.py`
- `my_affectgpt/rl/grpo_loss.py`
- `my_affectgpt/models/affectgpt.py`
- `my_affectgpt/datasets/datasets/base_dataset.py`
- `my_affectgpt/datasets/datasets/base_grpo_dataset.py`

If you are changing dataset logic:

- `config.py`
- `my_affectgpt/datasets/builders/image_text_pair_builder.py`
- `my_affectgpt/datasets/datasets/*.py`
- `my_affectgpt/processors/video_processor.py`

If you are changing inference or evaluation:

- `inference_hybird.py`
- `inference_sample.py`
- `my_affectgpt/conversation/conversation_video.py`
- `evaluation.py`
- `evaluation-scoreonly.py`

## Practical Guidance For Future Agents

- Read `README.md` first for expected directory layout and external assets.
- Read this file second for the actual runtime graph.
- Before changing paths, check `config.py`; many scripts depend on those maps.
- Before changing training prompts or modality tokens, inspect both dataset prompt builders and `AffectGPT.forward()`.
- Before changing GRPO logic, inspect `video_text_grpo.py`, `runner_grpo.py`, `grpo_loss.py`, and the GRPO config together. These pieces are tightly coupled.
- If `kl_coef == 0`, do not accidentally reintroduce eager ref-model construction in the runner or task. That would silently waste GPU memory.
- `BaseGRPODataset.collater()` is intentionally strict about inconsistent modality shapes. Do not change it back to silent dropping unless you also redesign modality padding.
- Before changing checkpoint behavior, inspect both training save logic and inference load logic.
- Be careful with dirty worktrees. This repo may already contain local edits in core files.
