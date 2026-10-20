# ChangeRadar contributor instructions

- ChangeRadar is a local-first AI Codebase Impact Analyst.
- Implemented: Phase 1 source scanning and dependency analysis; Phase 2 lexical, semantic, and hybrid retrieval with evaluation; Phase 3 evidence-grounded RAG and bounded tool workflow; Phase 4 Streamlit UI.
- Do not start deployment work without an explicit request.
- Keep the application lightweight and suitable for an 8 GB RAM laptop.
- Never execute code from analyzed repositories.
- Never transmit repository contents externally unless explicitly enabled by the user.
- Never commit credentials, local virtual environments, caches, or generated model artifacts.
- Automated tests must not require network access, API credentials, or model downloads.
- Write tests for new functionality and run the relevant tests and quality checks.
- Report actual evaluation results; never invent metrics.
- Keep changes small and modular. Explain significant dependency additions and verify functionality before claiming it works.
