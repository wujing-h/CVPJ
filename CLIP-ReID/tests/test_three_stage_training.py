import ast
import os
import unittest
from types import SimpleNamespace

import torch

from config import cfg


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def stage_cfg():
    return SimpleNamespace(
        IMS_PER_BATCH=64,
        OPTIMIZER_NAME="Adam",
        BASE_LR=0.000005,
        WARMUP_METHOD="linear",
        WARMUP_ITERS=10,
        WARMUP_FACTOR=0.1,
        WEIGHT_DECAY=0.0001,
        WEIGHT_DECAY_BIAS=0.0001,
        LARGE_FC_LR=False,
        MAX_EPOCHS=60,
        CHECKPOINT_PERIOD=60,
        LOG_PERIOD=50,
        EVAL_PERIOD=60,
        BIAS_LR_FACTOR=2,
        STEPS=[30, 50],
        GAMMA=0.1,
        MOMENTUM=0.9,
        CENTER_LR=0.5,
        LR_MIN=1e-6,
        WARMUP_LR_INIT=0.00001,
        WARMUP_EPOCHS=5,
        TEXT_LR_FACTOR=70,
    )


class TinyThreeStageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_learner = torch.nn.Linear(2, 2)
        self.text_encoder = torch.nn.Linear(2, 2)
        self.image_encoder = torch.nn.Linear(2, 2)
        self.classifier = torch.nn.Linear(2, 2)


class ThreeStageTrainingTest(unittest.TestCase):
    def test_three_stage_config_merges_with_defaults(self):
        local_cfg = cfg.clone()
        local_cfg.merge_from_file(os.path.join(REPO_ROOT, "configs", "person", "vit_clipreid_3stage.yml"))

        self.assertTrue(hasattr(local_cfg.SOLVER, "STAGE3"))
        self.assertEqual(local_cfg.SOLVER.STAGE2.TEXT_LR_FACTOR, 70)
        self.assertEqual(local_cfg.SOLVER.STAGE2.IMAGE_LOSS_WEIGHT, 1.0)
        self.assertEqual(local_cfg.SOLVER.STAGE2.TEXT_LOSS_WEIGHT, 1.0)
        self.assertEqual(local_cfg.SOLVER.STAGE3.BASE_LR, local_cfg.SOLVER.STAGE2.BASE_LR)
        self.assertEqual(local_cfg.SOLVER.STAGE3.STEPS, local_cfg.SOLVER.STAGE2.STEPS)

    def test_three_stage_entrypoint_runs_stage1_joint_stage2_then_stage3(self):
        path = os.path.join(REPO_ROOT, "train_clipreid_3stage.py")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source)
        called_names = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]

        self.assertLess(called_names.index("do_train_stage1"), called_names.index("do_train_stage2_joint"))
        self.assertLess(called_names.index("do_train_stage2_joint"), called_names.index("do_train_stage3"))
        self.assertIn("cfg.SOLVER.STAGE2.TEXT_LR_FACTOR", source)

    def test_joint_optimizer_returns_separate_prompt_and_image_optimizers(self):
        from solver.make_optimizer_prompt_3stage import make_optimizer_2stage_joint

        model = TinyThreeStageModel()
        cfg_mock = SimpleNamespace(SOLVER=SimpleNamespace(STAGE1=stage_cfg(), STAGE2=stage_cfg()))
        center = torch.nn.Linear(2, 2)

        optimizer_text, optimizer_image, _ = make_optimizer_2stage_joint(cfg_mock, model, center)
        text_optimized = {id(group["params"][0]) for group in optimizer_text.param_groups}
        image_optimized = {id(group["params"][0]) for group in optimizer_image.param_groups}

        self.assertIn(id(model.prompt_learner.weight), text_optimized)
        self.assertNotIn(id(model.image_encoder.weight), text_optimized)
        self.assertIn(id(model.image_encoder.weight), image_optimized)
        self.assertNotIn(id(model.prompt_learner.weight), image_optimized)
        self.assertNotIn(id(model.text_encoder.weight), text_optimized | image_optimized)
        self.assertEqual(optimizer_text.param_groups[0]["lr"], cfg_mock.SOLVER.STAGE2.BASE_LR * cfg_mock.SOLVER.STAGE2.TEXT_LR_FACTOR)
        self.assertEqual(optimizer_image.param_groups[0]["lr"], cfg_mock.SOLVER.STAGE2.BASE_LR)
        self.assertFalse(model.text_encoder.weight.requires_grad)
        self.assertTrue(model.prompt_learner.weight.requires_grad)

    def test_stage3_optimizer_trains_image_but_freezes_prompt_and_text(self):
        from solver.make_optimizer_prompt_3stage import make_optimizer_3stage

        model = TinyThreeStageModel()
        cfg_mock = SimpleNamespace(SOLVER=SimpleNamespace(STAGE3=stage_cfg()))
        center = torch.nn.Linear(2, 2)

        optimizer, _ = make_optimizer_3stage(cfg_mock, model, center)
        optimized = {id(group["params"][0]) for group in optimizer.param_groups}

        self.assertIn(id(model.image_encoder.weight), optimized)
        self.assertNotIn(id(model.prompt_learner.weight), optimized)
        self.assertNotIn(id(model.text_encoder.weight), optimized)
        self.assertFalse(model.prompt_learner.weight.requires_grad)
        self.assertFalse(model.text_encoder.weight.requires_grad)

    def test_joint_processor_recomputes_text_features_inside_training_loop(self):
        path = os.path.join(REPO_ROOT, "processor", "processor_clipreid_stage2_joint.py")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn("def do_train_stage2_joint", source)
        self.assertIn("text_features = build_all_text_features(", source)
        self.assertIn("detach=False", source)

    def test_joint_processor_combines_image_loss_and_text_loss(self):
        path = os.path.join(REPO_ROOT, "processor", "processor_clipreid_stage2_joint.py")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn("from loss.supcontrast import SupConLoss", source)
        self.assertIn("text_features_for_target = model(label=target, get_text=True)", source)
        self.assertIn("image_features_for_text_loss = image_features.detach()", source)
        self.assertIn("text_loss_i2t = xent(image_features_for_text_loss, text_features_for_target, target, target)", source)
        self.assertIn("text_loss_t2i = xent(text_features_for_target, image_features_for_text_loss, target, target)", source)
        self.assertIn("image_loss = loss_fn(score, feat, target, target_cam, logits)", source)
        self.assertIn("loss = cfg.SOLVER.STAGE2.IMAGE_LOSS_WEIGHT * image_loss + cfg.SOLVER.STAGE2.TEXT_LOSS_WEIGHT * text_loss", source)


if __name__ == "__main__":
    unittest.main()
