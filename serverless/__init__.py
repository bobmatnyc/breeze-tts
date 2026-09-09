"""Support modules for the RunPod Serverless worker (``rp_handler.py``).

These are deliberately import-light: nothing here pulls in torch or
transformers, so the handler's request contract and its on-volume storage stay
testable on a CPU-only machine.
"""
