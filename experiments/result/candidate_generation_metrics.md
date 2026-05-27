# Candidate Generation Metrics

Valid ratio uses strict geometry criteria: 0.10 <= Pred_BCR <= 0.35, 2 <= components <= 12, outside violation = 0, max component ratio <= 0.50, and aspect valid ratio >= 0.60. Diversity is the per-sample standard deviation of Pred_BCR across generated candidates.

| method | valid_ratio | diversity | max_component_ratio | aspect_valid_ratio |
| --- | --- | --- | --- | --- |
| Single sample | 0.043379 | 0.000000 | 0.678305 | 0.185312 |
| Multi-sample random | 0.136530 | 0.055999 | 0.607119 | 0.317017 |
| Ours + Validator top-1 | 0.547945 | 0.000000 | 0.410695 | 0.629338 |
| Ours + Validator top-5 average | 0.384018 | 0.000000 | 0.441302 | 0.537396 |
