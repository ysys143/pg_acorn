# pg_acorn Benchmark Report

## Scenario A: Filter Selectivity Sweep

Target                        1%      5%     10%     40%     80%
----------------------------------------------------------------
pgvector                  0.040  0.204  0.408  0.956  0.972
pg_acorn_tier1_g1         1.000  0.206  0.408  0.950  0.970
pg_acorn_tier2_g1         1.000  0.974  0.918  0.936  0.950
pg_acorn_tier2_g2         1.000  0.992  0.992  0.966  0.970


## Scenario B: Post-Filter Recall (pgvector CTE workaround)

(not run)