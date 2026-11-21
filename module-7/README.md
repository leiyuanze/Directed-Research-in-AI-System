# Module 7: Multi-Agent Systems

## Completed submission

This repository contains the **Short Assignment (5 points)** submission:

- [`code/ppo.py`](code/ppo.py) - PPO actor, critic, action sampling, and
  clipped surrogate update.
- [`code/rl_utils.py`](code/rl_utils.py) - replay-buffer sampling, rollout
  collection, and generalized advantage estimation.
- [`train_ppo.py`](train_ppo.py) - reproducible CartPole training without a
  render window.
- [`results.json`](results.json) - all 500 episode returns and measured
  aggregates.
- [`report/report.pdf`](report/report.pdf) - three-page LaTeX report.
- [`ML7_2510124204_submission.zip`](ML7_2510124204_submission.zip) - complete
  Gradescope package.

The fixed-seed run improves from a first-100 mean return of 140.26 to a
last-100 mean of 199.19. In the final 100 episodes, 96 reach the 200-step
CartPole-v0 limit and the minimum return is 173. The result is evidence for
finite-horizon control, not a proof of perpetual balance.

## Assignment choice

The handout defines two independent submission channels:

| Channel | Scope | This repository |
| --- | --- | --- |
| Full Assignment | Task 1 PPO and Task 2 MADDPG | Not submitted |
| Short Assignment | Task 1 PPO | Completed |

The Short package does not include `maddpg.py`, MADDPG figures, or
multi-agent analysis. The supplied `code/maddpg.py` starter remains outside the
submission archive.

## Reproduction

The original handout uses `gym`. Python 3.14 has no compatible binary wheel for
`gym==0.26.2` or `pygame`, so this machine ran the same five-value CartPole
interface through `gymnasium==1.2.1` and `pygame-ce==2.5.8`. The submitted
`code/ppo.py` still imports `gym`, matching the starter and grading interface.

```powershell
D:\miniconda3\envs\course_env\python.exe test_ppo.py
D:\miniconda3\envs\course_env\python.exe train_ppo.py --episodes 500
D:\miniconda3\envs\course_env\python.exe plot_results.py
```

The report's exact settings are stored in `results.json`. The documented
compatibility layer does not alter the learning update.
