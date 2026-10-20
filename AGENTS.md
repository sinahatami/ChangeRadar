# ChangeRadar contributor instructions

- ChangeRadar is a local-first code change impact analyst. Current scope is Phase 1: safe Python source scanning, AST parsing, and initial dependency analysis. Do not start later phases without an explicit request.
- Keep the application lightweight and practical on an 8 GB RAM laptop. Explain significant dependency additions before introducing them.
- Never execute code from analyzed repositories. Do not transmit repository contents externally unless the user explicitly enables that behavior.
- Keep changes small and modular. Add tests for new functionality and run the relevant tests and quality checks after changes.
- Do not claim a feature works unless it has been verified.
