# Module 5: Natural Language Processing

## Completed submission

This repository contains the **Short Assignment (5 points)** submission:

- [`report/report.pdf`](report/report.pdf) - five-page design report on a BCG
  observation and feedback layer for BrowseComp agents.
- [`report/report.tex`](report/report.tex) - XeLaTeX source with the embedded
  system diagram.
- [`ML5_2510124204_submission.pdf`](ML5_2510124204_submission.pdf) - the PDF
  prepared for the Gradescope short channel.
- [`编译说明.md`](编译说明.md) - local build and submission notes.

The report covers the four short-assignment criteria: BCG context-graph design,
one concrete weakest-link intervention, BrowseComp performance and token-cost
evaluation, and feasibility/reproducibility. It is a design report and does
not claim measured benchmark results.

## Assignment choice

Choose either the **Full Assignment (10 points)** or the **Short Assignment
(5 points)**. In the original handout, Practice Assignment means Full
Assignment and Written Assignment means Short Assignment.

This module uses the current BCG / BrowseComp assignment. The archived
MultiNLI fine-tuning task from an earlier offering is not part of this release.

| Option | Scope | Submission |
| --- | --- | --- |
| **Full Assignment** | Implement the task-oriented agent and BCG integration, run BrowseComp, and submit code plus measured artifacts. | [Full submission](SUBMISSION.full.md) |
| **Short Assignment** | Write a LaTeX PDF describing the implementation approach for the full assignment. | [Short submission](SUBMISSION.short.md) |

The completed work here is the short option. The report uses BCG as an
incremental evidence-grounded memory and uses one graph-driven verification
gate to intervene in the agent before it finalizes an answer.

The original task descriptions remain in [ASSIGNMENT.full.md](ASSIGNMENT.full.md)
and [ASSIGNMENT.short.md](ASSIGNMENT.short.md).
