# Result Tables Summary

Main results use: Module 1 + Module 2B-2 + Module 3, checkpoint = 90974.

Gap-C is not used in the main results.
Gap-C is reported only as an optional visual refinement branch.

Generated files:
- main_experiment_metrics.csv / .md
- ablation_metrics.csv / .md
- candidate_generation_metrics.csv / .md
- supplementary_metrics.csv
- optional_gaploss_comparison.csv / .md
- module3_strict_top1_eval_metrics.csv

Gap-C configuration: lambda_gap = 0.005, gap_kernel_size = 3, finetune_steps = 5000.
Gap-C improves aspect-ratio validity and slightly reduces component aggregation, but it slightly decreases Dice, IoU, and Recall. Therefore, it is not used for the Ours row in main_experiment_metrics.csv.
