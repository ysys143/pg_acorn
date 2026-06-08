# pg_acorn Benchmark Report

## Scenario A: Filter Selectivity Sweep

Target                        1%      5%     10%     40%     80%
----------------------------------------------------------------
pgvector                  1.000  1.000  1.000  1.000  1.000
pg_acorn_tier1_g1         1.000  1.000  1.000  0.982  0.974
pg_acorn_tier2_g1         1.000  0.980  0.922  0.924  0.950
pg_acorn_tier2_g2         1.000  0.992  0.996  0.972  0.976


## Scenario A: Page Accesses per Query (shared hit + read)

Target                         1%       5%      10%      40%      80%
---------------------------------------------------------------------
pgvector                    27353    12679     7905     2541     1441
pg_acorn_tier1_g1           27349    12726     7919     1244     1230
pg_acorn_tier2_g1           63000    22820    13027     2403     2389
pg_acorn_tier2_g2             371    35922    20970     4186     4172


## Scenario B: Post-Filter Recall (pgvector CTE workaround)

(not run)