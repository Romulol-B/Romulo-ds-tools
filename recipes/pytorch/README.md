# Optional PyTorch Recipe

PyTorch is intentionally not part of the V1 training path. Keep deep learning experiments here until a repeated pattern emerges, then promote the stable pieces into the core workflow.

Recommended approach:

1. Start with a plain `torch.nn.Module`.
2. Prove the model can overfit a tiny batch.
3. Move only the stable training structure into Lightning if the boilerplate becomes repetitive.
4. Track metrics and artifacts with MLflow using the same experiment naming style as the sklearn workflow.
