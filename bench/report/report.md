# pg_acorn Benchmark Report

## Scenario A: Filter Selectivity Sweep

Target                        1%      5%     10%     40%     80%
----------------------------------------------------------------
pgvector                  1.000  1.000  1.000  1.000  1.000
pg_acorn_tier1_g1         1.000  1.000  1.000  1.000  1.000
pg_acorn_tier2_g1         0.035  0.195  0.395  0.990  0.995
pg_acorn_tier2_g2         0.035  0.190  0.395  0.990  0.995


## Scenario B: Post-Filter Recall (pgvector CTE workaround)

(not run)