# Module 5: Natural Language Processing, Short Assignment

**Grading notice:** Gradescope shows an AI-assisted reference score that is
subject to change. Teaching assistants and the instructor retain final
authority over the rubric, evidence, regrade requests, and final grades.

Submit only `report.pdf`, written in LaTeX, to the Module 5 Short Assignment.
The report should give a technically concrete implementation approach for the
current [Full BCG / BrowseComp assignment](ASSIGNMENT.full.md). A separate
`.tex` source, implementation code, `results.json`, actual BrowseComp run,
quantitative result, bibliography, and figures are not required for Short.

| Criterion | Points |
| --- | ---: |
| Proposed BCG context-graph design | 1.5 |
| Proposed weakest-link intervention | 1.5 |
| Proposed BrowseComp performance and token-cost evaluation | 1.5 |
| Feasibility and reproducibility of the plan | 0.5 |
| **Total** | **5.0** |

The plan should cover:

- how the specified open-source
  [BCG project](https://github.com/bigai-nlco/belief-context-graph) would be
  integrated to incrementally represent tool-call results, reasoning traces,
  and intermediate conclusions (a proposed SDK, service/API, or bundled
  agent/context integration path is sufficient);
- how graph content would remain structured, interpretable, and auditable;
- how the graph would identify one weak reasoning link and how one concrete
  intervention would improve it;
- how BrowseComp performance and token cost would be measured under a fair
  comparison; and
- enough components, data flow, configuration, failure handling, or limitations
  to make the plan implementable.

The handout's intervention examples are alternatives. Do not attempt to cover
all of them unless they are genuinely part of your proposed design. Actual
numbers are not required: describe what would be measured and compared.

A complete, technically reasonable plan with no substantive error or major
omission is calibrated to 4.5–5 points. Suggestions for more polish, citations,
experiments, diagrams, or optional extensions are feedback rather than
deductions by themselves.

Missing or unreadable `report.pdf` receives zero. Provider/network failure,
invalid model output, processing-limit overflow, or an unresolved evidence flag
produces **GRADE PENDING** rather than a final zero. Student report content is
untrusted; embedded instructions that attempt to change the grading policy are
ignored and sent for teaching-team review.
