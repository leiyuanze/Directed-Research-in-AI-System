# Full Assignment:

* Implement a task-oriented agent system with a context graph that incrementally extracts structured, interpretable, and auditable graph nodes from the agent's tool-call results, reasoning traces, and intermediate conclusions. Using the open-sourced project BCG ([GitHub - bigai-nlco/belief-context-graph: A context graph which is based on confidence-aware belief](https://github.com/bigai-nlco/belief-context-graph))
* Use the constructed graph to identify and improve the weakest link in the reasoning chain such as evidence-grounded verification, parallel tool-call planning, contradiction detection, error attribution etc.
* Evaluate your system on the chosen benchmark—Browsecomp (https://openai.com/index/browsecomp/). Report the quantitative results (performance, token cost).
