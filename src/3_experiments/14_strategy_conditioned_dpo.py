#!/usr/bin/env python3
from cognitive_methods_common import method_entrypoint

if __name__ == "__main__":
    method_entrypoint(
        method_name="m14_strategy_conditioned_dpo",
        default_model_path="/scratch/craj/diy/outputs/7_finetuned_models/finetuned_ms-full-allstrategies-opinion-action-event-allversions",
        default_inference_mode="strategy",
        default_inference_strategy="all_strategies",
    )
