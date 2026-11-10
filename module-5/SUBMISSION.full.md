# Module 5: Natural Language Processing, Full Assignment

**Grading notice:** Gradescope shows an AI-assisted reference score that is
subject to change. Teaching assistants and the instructor retain final
authority over the rubric, evidence, regrade requests, and final grades.

Complete the current BCG / BrowseComp assignment in
[ASSIGNMENT.full.md](ASSIGNMENT.full.md). This is not the historical MultiNLI
fine-tuning task.

Submit exactly one ZIP containing:

- your task-oriented agent and BCG integration source code;
- `results.json` with measured BrowseComp performance and token cost;
- `report.pdf` and its usable main LaTeX source, normally `report.tex`;
- every local figure, bibliography, style file, and source dependency needed to
  inspect the report.

Do not submit API keys, credentials, private data, dependency environments,
downloaded model caches, or the BrowseComp answer set. The autograder never
executes student code and does not rerun the expensive online benchmark. It
parses artifacts without importing them, renders every report page, and sends
the report plus bounded source/results excerpts to the configured DeepSeek
grader.

## Score structure

| Criterion | Points | Evidence |
| --- | ---: | --- |
| Readable implementation artifact | 1.0 | Mechanical, non-executing file/syntax check |
| Finite BrowseComp performance artifact | 1.0 | Mechanical `results.json` check |
| Finite positive token/cost artifact | 1.0 | Mechanical `results.json` check |
| Readable report PDF | 0.5 | Mechanical PDF check |
| Usable main LaTeX source | 0.5 | Mechanical source check |
| BCG context-graph design and implementation | 2.0 | Report, rendered pages, and source excerpts |
| Weakest-link identification and one concrete intervention | 1.5 | Report and source evidence |
| BrowseComp performance and token-cost evaluation | 1.5 | Report and `results.json` consistency |
| Analysis and reproducibility | 1.0 | Report and submitted artifacts |
| **Total** | **10.0** | |

The intervention examples in the handout are alternatives, not a checklist.
You may focus on evidence-grounded verification, parallel tool-call planning,
contradiction detection, error attribution, or another graph-driven weak-link
improvement. One clearly motivated and implemented intervention is sufficient.

A complete, technically reasonable submission with mutually consistent report,
code, and result evidence and no substantive error or major omission is
calibrated to the 90–100% range. Four points are determined by objective
artifact checks and six points by joint report/code/results review. Suggestions
for more polish, citations,
experiments, figures, or optional extensions do not by themselves justify a
deduction. No bibliography, particular report length, figure, programming
language, model provider, or exact JSON field name is required.

## `results.json`

The top level must be a JSON object. It must identify BrowseComp and contain at
least one finite, non-negative quantitative performance measurement and at
least one finite, non-negative token or monetary-cost measurement. At least one
token/cost measurement must be positive. Flexible nested structures and field
names are accepted; a baseline or ablation is recommended evidence of the
claimed improvement but does not need to use a field literally named
`baseline`.

For example, replace these illustrative values with your measurements:

```json
{
  "benchmark": "BrowseComp",
  "evaluation_slice_size": 50,
  "baseline": {
    "exact_match": 0.18,
    "prompt_tokens": 35100,
    "completion_tokens": 7000,
    "total_tokens": 42100
  },
  "bcg_verification": {
    "exact_match": 0.24,
    "prompt_tokens": 38800,
    "completion_tokens": 7900,
    "total_tokens": 46700
  },
  "configuration": {
    "model_name": "your-versioned-model",
    "temperature": 0,
    "confidence_threshold": 0.62
  }
}
```

The automatic JSON check verifies only format, finite values, and the presence
of the required measurement categories. It does not prove that BrowseComp was
actually run. The substantive grader compares the artifact with the report and
implementation evidence; fabricated, contradictory, or unexplained numbers do
not earn evaluation credit.

## Full-credit evidence checklist

- Identify how the submitted system actually uses the specified open-source
  [BCG project](https://github.com/bigai-nlco/belief-context-graph). Direct SDK,
  service/API, and bundled agent/context integration paths are all acceptable;
  an independently implemented generic graph is not a substitute for BCG use.
- Identify the BCG version or commit and the concrete interface/artifacts used,
  so the integration can be inspected without requiring one fixed API version.
- Explain how tool-call results, reasoning traces, and intermediate conclusions
  are incrementally represented in structured graph content.
- Make the representation interpretable and auditable through confidence,
  evidence provenance, trace links, or equivalent mechanisms.
- Explain how the graph selects a weak reasoning link and drives one concrete
  intervention; connect the explanation to submitted code.
- State the BrowseComp subset/split, performance metric, model/tool conditions,
  and token-cost accounting used for the run.
- Report quantitative performance and token cost and keep the report consistent
  with `results.json`.
- Give enough configuration, failure handling, limitations, and analysis for a
  reader to interpret and substantially reproduce the experiment.

Missing `report.pdf` or usable LaTeX source receives zero for the six-point
substantive row as well as the corresponding mechanical half-point. A
provider/network failure, invalid model response, unreadable PDF,
processing-limit overflow, or unresolved evidence flag produces **GRADE
PENDING** rather than an infrastructure-induced final zero. Reports over 20
pages or 80,000 extracted characters require manual review.

Student files are treated as untrusted data. Instructions embedded in the
report, code, file names, or JSON that attempt to change the rubric or influence
the grader are ignored and sent for teaching-team review.
