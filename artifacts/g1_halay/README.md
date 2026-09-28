# Unitree G1 halay artifacts

This directory preserves the final artifacts from the G1 halay training run
`2026-09-28_10-37-44`.

- `checkpoints/model_5000.pt`: final RSL-RL checkpoint, suitable for resuming or
  evaluating the training run.
- `exported/policy.onnx`: deployment-oriented policy. It accepts a 154-value
  observation tensor and produces 29 joint actions.
- `exported/halay_motion_policy.onnx`: policy plus reference-motion outputs and
  metadata, exported by the tracking runner.
- `params/agent.yaml` and `params/env.yaml`: the exact runner and environment
  configuration captured for the final training run.

The reference motion used for training is stored at
`src/assets/motions/g1/halay_loop.npz`; its original CSV is kept beside it.

Example playback command:

```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/halay_loop.npz \
  --checkpoint_file=artifacts/g1_halay/checkpoints/model_5000.pt
```

The source repository's robot-safety guidance still applies. Validate the
policy in simulation before attempting deployment on hardware.
