# tt-symbiote-research-topics
Shared knowledge base for the TT-Symbiote agentic workflow. Acts as a cache between the Team prompt flow (Architect agent) and the Research prompt flow (guide generation agents).

- The Team Architect writes Pending topics here when it needs information it doesn't have (cache miss).
- The Research instance picks up Pending topics, generates a full A→B→C reviewed guide for each, and writes back Completed status with a pointer to the output guide directory.
- Neither flow waits on the other — the repo is the async handoff point.
