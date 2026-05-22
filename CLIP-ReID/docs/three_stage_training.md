# CLIP-ReID Three-Stage Training Plan

This document describes the intended three-stage CLIP-ReID experiment without changing the existing two-stage training entrypoint.

## Goal

Try a training flow that separates prompt-only learning, joint prompt/image adaptation, and final image-only finetuning:

1. Stage 1: train `prompt_learner` only.
2. Stage 2: train `prompt_learner` and `image_encoder` together while keeping `text_encoder` frozen.
3. Stage 3: train `image_encoder` only with prompt/text features fixed.

## Existing Baseline

The current implementation already supports the original paper's two-stage flow:

- `processor/processor_clipreid_stage1.py` trains the prompt learner against cached image features.
- `solver/make_optimizer_prompt.py::make_optimizer_1stage` selects only `prompt_learner` parameters.
- `processor/processor_clipreid_stage2.py` caches all text features once before image training.
- `solver/make_optimizer_prompt.py::make_optimizer_2stage` freezes `text_encoder` and `prompt_learner`, then trains the image side.

## Proposed Files For Implementation

Keep the existing files intact and add parallel files for the experiment:

- `train_clipreid_3stage.py`
  - Copy `train_clipreid.py`.
  - Run stage 1 with the existing `do_train_stage1`.
  - Run stage 2 with a new joint prompt/image processor.
  - Run stage 3 with the existing image-only stage 2 processor, or a copied stage 3 wrapper for clearer checkpoint names.
- `solver/make_optimizer_prompt_3stage.py`
  - Copy the current optimizer helpers.
  - Add a joint stage 2 optimizer that freezes only `text_encoder`.
  - Add a stage 3 optimizer equivalent to the current image-only optimizer.
- `processor/processor_clipreid_stage2_joint.py`
  - Copy `processor_clipreid_stage2.py`.
  - Recompute text features inside the training loop without `torch.no_grad()` so `prompt_learner` receives gradients.
  - Keep `text_encoder` frozen through the optimizer parameter selection.

## Config

`configs/person/vit_clipreid_3stage.yml` mirrors the current ViT CLIP-ReID settings and adds `SOLVER.STAGE3`.

The new `STAGE3` block intentionally copies `STAGE2` values so the experiment changes the training schedule, not learning rates, loss weights, batch sizes, or scheduler milestones. `OUTPUT_DIR` is separated to avoid mixing artifacts with the two-stage run.

In joint stage 2, prompt/text and image parameters use separate optimizers and schedulers. `SOLVER.STAGE2.TEXT_LR_FACTOR` controls their initial LR ratio:

`prompt_lr = SOLVER.STAGE2.BASE_LR * SOLVER.STAGE2.TEXT_LR_FACTOR`

The default experiment value is `70`, preserving the prior implicit ratio between the original prompt LR `0.00035` and image LR `0.000005`.

Joint stage 2 also combines two losses:

`loss = IMAGE_LOSS_WEIGHT * image_loss + TEXT_LOSS_WEIGHT * text_loss`

`image_loss` is the original ReID/image objective with `Li2tce`. `text_loss` is the stage1-style bidirectional contrastive loss between current batch image features and prompt-generated text features. Image features are detached for `text_loss`, so this term trains the prompt/text side while the image side is trained by `image_loss`.

The current two-stage entrypoint ignores `SOLVER.STAGE3`; use `train_clipreid_3stage.py` for this experiment.

## Mock Validation

For now, validate this work without GPU use:

- Parse the new yml with `yaml.safe_load`.
- Check that `SOLVER.STAGE1`, `SOLVER.STAGE2`, and `SOLVER.STAGE3` exist.
- Check that `STAGE3` matches `STAGE2` for the copied optimizer and scheduler keys.

Do not run `train_clipreid.py`, `train_clipreid_3stage.py`, or any command that builds the CLIP model on CUDA during mock validation.
