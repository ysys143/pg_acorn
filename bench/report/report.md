# pg_acorn Benchmark Report

## Scenario A: Filter Selectivity Sweep

Target                        1%      5%     10%     40%     80%
----------------------------------------------------------------
pgvector                  1.000  1.000  1.000  0.988  0.982
pg_acorn_tier1_g1         1.000  1.000  1.000  1.000  1.000
pg_acorn_tier2_g1         0.998  0.952  0.884  0.926  0.934
pg_acorn_tier2_g2         1.000  0.992  0.988  0.984  0.984


## Scenario A: Page Accesses per Query (shared hit + read)

Target                         1%       5%      10%      40%      80%
---------------------------------------------------------------------
pgvector                    27420    12791     7894     1254     1240
pg_acorn_tier1_g1           27348    12678     7905     2524     1386
pg_acorn_tier2_g1           19638     8715     5339     1695      933
pg_acorn_tier2_g2           21853    12649     8588     3025     1702


## Scenario B: Post-Filter Recall (pgvector CTE workaround)

(not run)