# 5-task continual learning -- end-of-run report

Sequence: Stack -> Pull -> Pick -> LiftPeg -> Push  (2.5M steps/task, 48-dim unified obs, 128 envs x 64 steps)

## Final success rate, 100 episodes per task

| Task | expert | ER | SeqFT | ER+MTD | ER+MTD+Cons |
|---|---|---|---|---|---|
| Stack | 87% | 25% | 0% | 76% | 85% |
| Pull | 100% | 100% | 0% | 100% | 100% |
| Pick | 98% | 57% | 0% | 99% | 94% |
| LiftPeg | 95% | 76% | 0% | 92% | 88% |
| Push | 100% | 100% | 100% | 100% | 100% |
| **mean SR** | 96% | 72% | 20% | 93% | 93% |

## Forgetting vs the single-task experts (100 episodes)

| Task | expert | ER | SeqFT | ER+MTD | ER+MTD+Cons |
|---|---|---|---|---|---|
| Stack | 87% | -62 pp | -87 pp | -11 pp | -2 pp |
| Pull | 100% | +0 pp | -100 pp | +0 pp | +0 pp |
| Pick | 98% | -41 pp | -98 pp | +1 pp | -4 pp |
| LiftPeg | 95% | -19 pp | -95 pp | -3 pp | -7 pp |
| Push | 100% | +0 pp | +0 pp | +0 pp | +0 pp |


## Training-time forgetting matrix -- ER

Rows: the last eval recorded while that task was trained (20 episodes). Columns: every task seen so far.

| after | steps | Stack | Pull | Pick | LiftPeg | Push |
|---|---|---|---|---|---|---|
| Stack | 2498560 | 85% | - | - | - | - |
| Pull | 2498560 | 85% | 100% | - | - | - |
| Pick | 2498560 | 70% | 100% | 100% | - | - |
| LiftPeg | 2498560 | 75% | 95% | 40% | 95% | - |
| Push | 2498560 | 65% | 100% | 70% | 70% | 100% |

## Training-time forgetting matrix -- SeqFT

Rows: the last eval recorded while that task was trained (20 episodes). Columns: every task seen so far.

| after | steps | Stack | Pull | Pick | LiftPeg | Push |
|---|---|---|---|---|---|---|
| Stack | 2498560 | 0% | - | - | - | - |
| Pull | 2498560 | 0% | 100% | - | - | - |
| Pick | 2498560 | 0% | 0% | 0% | - | - |
| LiftPeg | 2498560 | 0% | 5% | 0% | 100% | - |
| Push | 2498560 | 0% | 0% | 0% | 0% | 100% |

## Training-time forgetting matrix -- ER+MTD

Rows: the last eval recorded while that task was trained (20 episodes). Columns: every task seen so far.

| after | steps | Stack | Pull | Pick | LiftPeg | Push |
|---|---|---|---|---|---|---|
| Stack | 2498560 | 85% | - | - | - | - |
| Pull | 2498560 | 95% | 100% | - | - | - |
| Pick | 2498560 | 80% | 95% | 90% | - | - |
| LiftPeg | 2498560 | 85% | 100% | 90% | 85% | - |
| Push | 2498560 | 70% | 100% | 90% | 85% | 100% |

## Training-time forgetting matrix -- ER+MTD+Cons

Rows: the last eval recorded while that task was trained (20 episodes). Columns: every task seen so far.

| after | steps | Stack | Pull | Pick | LiftPeg | Push |
|---|---|---|---|---|---|---|
| Stack | 294912 | 85% | - | - | - | - |
| Pull | 294912 | 75% | 100% | - | - | - |
| Pick | 294912 | 75% | 100% | 95% | - | - |
| LiftPeg | 294912 | 70% | 100% | 90% | 90% | - |
| Push | 1499136 | 85% | 100% | 90% | 85% | 100% |
[exit=0]
