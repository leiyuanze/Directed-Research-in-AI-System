# Module 7 Short Assignment

Submit one ZIP to the Gradescope **Short Assignment** channel:

`ML7_2510124204_submission.zip`

The package contains:

- `ppo.py`
- `rl_utils.py`
- `report.pdf`
- `report.tex`
- the figure and LaTeX dependencies referenced by the report

| Short-assignment check | Evidence |
| --- | --- |
| PPO actor and critic networks and optimizers | `ppo.py`, `PPO.__init__` |
| Policy-distribution action sampling | `ppo.py`, `PPO.take_action` |
| PPO clipped surrogate actor loss | `ppo.py`, `PPO.update` |
| CartPole training curve and result analysis | `results.json`, `report.pdf` |

The Short channel does not require `maddpg.py`, MADDPG training, or a
multi-agent report. None of those artifacts is included in the archive.

The report uses the measured 500-episode seed-0 run. It reports the
compatibility environment and finite-horizon limit explicitly rather than
claiming infinite-duration balance.
