from .defaults import _C as _BASE_C


_C = _BASE_C.clone()

# Joint stage2 uses separate prompt/text and image optimizers.
# prompt/text optimizer lr = image optimizer lr * TEXT_LR_FACTOR.
_C.SOLVER.STAGE2.TEXT_LR_FACTOR = 1.0

# Joint stage2 loss = IMAGE_LOSS_WEIGHT * image_loss + TEXT_LOSS_WEIGHT * text_loss.
_C.SOLVER.STAGE2.IMAGE_LOSS_WEIGHT = 1.0
_C.SOLVER.STAGE2.TEXT_LOSS_WEIGHT = 1.0

# Stage3 is image-only finetuning after joint prompt/image adaptation.
_C.SOLVER.STAGE3 = _C.SOLVER.STAGE2.clone()
