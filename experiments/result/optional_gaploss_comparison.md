# Optional Gap-C Comparison

Gap-C improves aspect-ratio validity and slightly reduces component aggregation, but it slightly decreases Dice, IoU, and Recall. Therefore, it is treated as an optional visual refinement branch and is not used in the main results. Local Gap-C data are available as single-sample evaluation only; Module 3 was not rerun for Gap-C because this stage forbids resampling.

| method | dice | iou | recall | precision | bcr_error | max_component_ratio | aspect_valid_ratio | small_fragment_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ours (main): M1 + M2B-2 + M3 strict top-1, checkpoint 90974 | 0.313034 | 0.192300 | 0.342123 | 0.320114 | 0.084151 | 0.410695 | 0.629338 | 0.000034 |
| Ours + Gap-C optional visual refinement (single-sample eval; not main) | 0.336332 | 0.210074 | 0.390868 | 0.325819 | 0.091786 | 0.591617 | 0.471877 | 0.000046 |
