# External Assets

This directory is for heavyweight or separately managed assets that should not be copied into the clean repo.

Expected optional layout:

```text
external/
  TinyRecursiveModels/
  models/
    trm_arc_prize_verification/
      arc_v2_public/
        step_723914/
      arc_v1_public/
        step_518071/
    architects_lora/
```

`scripts/infer_trm.py` also accepts:

- `AGI2_TRM_DIR`
- `AGI2_TRM_CKPT_V2`
- `AGI2_TRM_CKPT_V1`

Qwen LoRA paths can be passed directly with `--lora`.
