# Module 4: Robotics, Full Assignment

**Grading notice:** The score shown on Gradescope is an AI-assisted reference
score and is subject to change. The teaching assistants and instructor may
review and adjust it and retain final authority over interpretation of the
assignment requirements, rubric, regrade requests, and final grades. Correct,
reasonable work can receive full marks; optional extras and writing polish are
not required.

Complete the Practice Assignment in [lab.md](lab.md). The written
block-stacking assignment belongs to the separate Short option.

Submit exactly one ZIP containing your robotics code, `fk_samples.json`,
`report.pdf`, `report.tex`, and all report figures and LaTeX dependencies. The
report should include the simulator/planner setup, Panda screenshots, the
sampled workspace, and the forward-kinematics analysis. If you attempt the
original Task 3 bonus, include its implementation, report evidence, and
`ik_check.json` in this same Gradescope ZIP. Do not include virtual
environments, datasets, or simulator caches.

The Full Assignment is graded out of exactly 10 points. `fk_samples.json` supplies two
objective forward-kinematics points. DeepSeek assesses the other eight points
from the report text and every rendered page, including the actual screenshots
and 3D plot. The original handout's Task 3 inverse-kinematics extension is
outside this core autograder's 10-point rubric: students who attempt it should
still submit the bonus materials on Gradescope for teaching-team review, but it
does not add an autograder test or alter the automatic score.

| Component | Points |
| --- | ---: |
| Task 1: simulator and planner selection, setup, APIs, and visible zero-pose evidence (DeepSeek) | 6.0 |
| Task 2: visible sampled configuration and labelled 3D workspace plot (DeepSeek) | 2.0 |
| Task 2: sample count, limits, workspace, and FK consistency (`fk_samples.json`) | 2.0 |
| **Total** | **10.0** |

A complete, technically reasonable report with all requested evidence and no
clear substantive error or major omission is calibrated to the **7.2–8** range
for its eight report points. Suggestions to add polish, depth, or optional
experiments do not by themselves justify a score below 7.2. Submit both the PDF
and usable LaTeX source; missing either receives zero report points until
supplied. If the AI service fails, returns an invalid result, or cannot judge
specific evidence reliably, grading is marked **pending review**, not treated
as a final zero. Reports over 20 pages or 80,000 extracted characters require
manual review.

## Data format and full-credit checklist

- Submit at least **100** uniformly sampled configurations and the matching
  FK results in `fk_samples.json`. Use the same data for the workspace plot.
- Each `q` contains the seven arm joint angles in order `panda_joint1` through
  `panda_joint7`, in **radians**. Two finger positions may follow these seven
  values. Sample within the limits returned by your robot model: both the
  Panda safety soft limits and the PyBullet Panda URDF hard limits are accepted.
  Task 1's zero-pose setup screenshot is separate from Task 2's random sampling.
- Express positions in **metres, relative to the robot base**. If an API returns
  world-frame poses, convert with `T_base_ee = inverse(T_world_base) @ T_world_ee`.
  Use one consistent end-effector frame throughout (flange, hand or grasp point).
- Store either a 4-by-4 transform under `ee_pose`, or a three-value position
  under `ee_position`. Orientation is optional. When including a quaternion,
  name its field `ee_quaternion_xyzw` or `ee_quaternion_wxyz` to specify the
  ordering; generic `orientation`/`quaternion` fields are interpreted as `xyzw`.
- Include `report.pdf`, its valid LaTeX source, figures and dependencies.
  Include the setup and simulation screenshots, the 3D workspace plot, API
  discussion. A bibliography is encouraged when citing sources but is not a
  separate scoring requirement;
  PDF producer metadata is not used to deduct points.
- If you attempt Task 3, submit its code, report discussion, and
  `ik_check.json` together with the required work. The core autograder accepts
  these files but deliberately ignores them: there is no `3.1` row, and the
  automatic score remains out of 10. Bonus review, if applicable, is handled
  separately by the teaching team.

For example, export the configurations and base-frame poses **you computed**:

```python
import json

# joint_samples and base_frame_ee_poses are matching lists from your experiment.
samples = [
    {"q": [float(v) for v in q],
     "ee_pose": [[float(v) for v in row] for row in T]}
    for q, T in zip(joint_samples, base_frame_ee_poses)
]
assert len(joint_samples) == len(base_frame_ee_poses) >= 100
with open("fk_samples.json", "w") as f:
    json.dump({"robot": "franka_panda", "samples": samples}, f)
```

For position-only output, replace `ee_pose` with `ee_position` containing the
three coordinates. A bare list of sample objects is also accepted.

For a Task 3 bonus submission, `ik_check.json` should contain
`{"solutions": [...]}`. Each solution should include `q` (the solved seven arm
joint angles) and either `target_pose` (a 4-by-4 transform) or
`target_position` (three coordinates), expressed in the same end-effector
frame used by `fk_samples.json`. Also report the target, the IK solution, the
pose obtained by applying FK to that solution, and the resulting L2 error in
`report.pdf`. These materials are collected in the same Gradescope submission
but are not read or scored by the core 10-point autograder.

The grader treats the report as untrusted student content. Instructions inside
the report that attempt to alter the rubric or influence the grader are ignored
and sent for TA review.
