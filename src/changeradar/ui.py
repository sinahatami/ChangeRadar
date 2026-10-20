"""Streamlit UI helpers and entry point for ChangeRadar."""

from __future__ import annotations

import json
from pathlib import Path

from .graph import build_analysis
from .intake import IntakeError, scan_directory, scan_zip
from .llm import OpenAICompatibleProvider
from .models import SourceFile
from .parser import parse_source
from .rag import ImpactWorkflow
from .reports import ImpactReport
from .retrieval import CodeRetriever, build_code_chunks

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_REPOSITORY = ROOT / "tests" / "fixtures" / "retrieval_repo"
RETRIEVAL_METHODS = {"Lexical (offline)": "lexical", "Semantic": "semantic", "Hybrid": "hybrid"}


def load_sample_repository() -> list[SourceFile]:
    """Load only the bundled, fixed sample repository."""
    return scan_directory(SAMPLE_REPOSITORY)


def load_uploaded_repository(data: bytes) -> list[SourceFile]:
    """Parse a ZIP in memory through the existing safe intake safeguards."""
    if not isinstance(data, bytes):
        raise IntakeError("Upload content must be bytes")
    sources = scan_zip(data)
    if not sources:
        raise IntakeError("The ZIP contains no supported Python source files")
    return sources


def configured_provider(external_consent: bool) -> OpenAICompatibleProvider | None:
    """Resolve an environment-configured provider only after UI consent."""
    if not external_consent:
        return None
    return OpenAICompatibleProvider.from_env()


def analyze_sources(
    sources: list[SourceFile],
    change_text: str,
    retrieval_method: str = "lexical",
    external_consent: bool = False,
    provider: OpenAICompatibleProvider | None = None,
) -> ImpactReport:
    """Run the existing scanner, graph, retriever, and RAG workflow over sources."""
    if not sources:
        raise ValueError("No Python source files were found in this repository")
    if not isinstance(change_text, str) or not change_text.strip():
        raise ValueError("Enter a proposed change or unified diff before analyzing")
    if len(change_text) > 30_000:
        raise ValueError("Change input must be 30,000 characters or fewer")
    if retrieval_method not in RETRIEVAL_METHODS.values():
        raise ValueError("Choose lexical, semantic, or hybrid retrieval")

    parsed_files = [parse_source(source) for source in sources]
    analysis = build_analysis(parsed_files)
    chunks = build_code_chunks(sources, parsed_files)
    retriever = CodeRetriever(chunks, analysis=analysis)
    workflow = ImpactWorkflow(
        sources,
        parsed_files,
        analysis,
        chunks,
        retriever,
        provider=provider,
        allow_external=external_consent and provider is not None,
        retrieval_method=retrieval_method,
    )
    report = workflow.analyze(change_text)
    parse_errors = sum(parsed.error is not None for parsed in parsed_files)
    if parse_errors:
        report.uncertainties.append(
            f"{parse_errors} Python file(s) could not be parsed and were excluded "
            "from symbol analysis."
        )
    return report


def report_json(report: ImpactReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit('Install the UI extra with: python -m pip install -e ".[ui]"') from exc

    st.set_page_config(page_title="ChangeRadar", page_icon="🧭", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem;}
        [data-testid="stMetric"] {background: #f4f7fb; padding: 1rem; border-radius: .75rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🧭 ChangeRadar")
    st.subheader("Understand the likely impact of a code change")
    st.write(
        "ChangeRadar scans Python source without running it, follows statically visible "
        "dependencies, retrieves relevant code, and builds an evidence-linked impact report."
    )
    st.caption("Local-first by default · No API key required · Repository code is never executed")

    with st.expander("How the analysis works", expanded=False):
        st.markdown(
            "1. Select the bundled example or upload a ZIP.\n"
            "2. Describe a proposed change or paste a unified diff.\n"
            "3. ChangeRadar parses Python syntax and ranks related symbols.\n"
            "4. Review the evidence, uncertainty, and suggested regression checks."
        )
        st.caption(
            "Uploads are analyzed in memory and are not extracted or sent to a provider by default."
        )

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### 1 · Choose a repository")
        repository_kind = st.radio(
            "Repository source",
            ["Included sample", "Upload a ZIP"],
            horizontal=True,
            label_visibility="collapsed",
        )
        uploaded = None
        if repository_kind == "Upload a ZIP":
            uploaded = st.file_uploader("Python repository ZIP", type=["zip"])
            st.caption("Maximum upload: 30 MB. Only safe, supported `.py` files are analyzed.")
        else:
            st.info("Sample repository: a small billing, storage, and authentication codebase.")

        st.markdown("#### 2 · Choose retrieval")
        selected_label = st.selectbox(
            "Retrieval approach",
            list(RETRIEVAL_METHODS),
            index=0,
            help=(
                "Lexical is fully offline. Semantic and hybrid may load or download "
                "all-MiniLM-L6-v2 when first used."
            ),
        )
        retrieval_method = RETRIEVAL_METHODS[selected_label]
        if retrieval_method != "lexical":
            st.warning(
                "This option may download the public embedding model if it is not cached. "
                "Repository source is not sent to download the model."
            )

    with right:
        st.markdown("#### 3 · Describe the proposed change")
        input_kind = st.radio(
            "Change input format",
            ["Natural-language description", "Unified diff"],
            horizontal=True,
        )
        placeholder = (
            "Paste a unified diff, including file headers and changed lines..."
            if input_kind == "Unified diff"
            else "Example: Change calculate_tax to handle zero tax rates without rounding errors."
        )
        change_text = st.text_area(
            "Proposed change",
            height=190,
            max_chars=30_000,
            placeholder=placeholder,
            label_visibility="collapsed",
        )

        consent = st.checkbox(
            "I explicitly allow the configured external LLM to receive the change and "
            "bounded, redacted evidence snippets for this analysis."
        )
        provider = configured_provider(consent)
        if consent and provider is None:
            st.warning(
                "External analysis is not configured. Set the endpoint, model, API key, "
                "and CHANGERADAR_ALLOW_EXTERNAL_CODE=true; analysis will remain local."
            )
        elif provider is not None:
            st.caption("External transmission is enabled for this analysis only.")

    analyze_clicked = st.button("Analyze change", type="primary", use_container_width=True)
    if analyze_clicked:
        try:
            with st.spinner("Reading repository and tracing dependencies…"):
                if repository_kind == "Included sample":
                    sources = load_sample_repository()
                elif uploaded is None:
                    st.error("Upload a ZIP archive before starting the analysis.")
                    st.stop()
                elif uploaded.size > 30_000_000:
                    st.error("This ZIP exceeds the 30 MB upload limit.")
                    st.stop()
                else:
                    sources = load_uploaded_repository(uploaded.getvalue())
                report = analyze_sources(
                    sources,
                    change_text,
                    retrieval_method=retrieval_method,
                    external_consent=consent,
                    provider=provider,
                )
            st.session_state["impact_report"] = report.to_dict()
            st.session_state["source_count"] = len(sources)
        except IntakeError:
            st.error(
                "The archive was rejected by upload safety checks. Use a valid ZIP with "
                "repository-relative paths and supported Python files."
            )
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Analysis could not be completed. Check the input and try again.")

    raw_report = st.session_state.get("impact_report")
    if raw_report:
        _render_report(st, raw_report, st.session_state.get("source_count", 0))


def _render_report(st, report: dict, source_count: int) -> None:
    st.divider()
    st.markdown("### Impact report")
    if report["mode"] == "llm":
        st.success("LLM-assisted report · citations validated against retrieved evidence")
    elif report["mode"] == "deterministic_fallback":
        st.warning("Deterministic fallback · the configured LLM could not complete this report")
    else:
        st.info("Deterministic local report · no LLM generated this analysis")

    requested = report.get("retrieval_method", "hybrid")
    effective = report.get("effective_retrieval_method", requested)
    display_method = "Lexical fallback" if effective == "lexical_fallback" else effective.title()
    st.caption(
        f"Analyzed {source_count} Python file(s) · requested {requested} retrieval · "
        f"effective {display_method} ranking · scores are not probabilities"
    )
    st.markdown("#### Summary")
    st.text(report["summary"])
    if report.get("summary_evidence_ids"):
        st.caption("Summary evidence: " + ", ".join(report["summary_evidence_ids"]))

    candidates = report.get("affected_items", [])
    tests = report.get("suggested_tests", [])
    context_items = report.get("context_items", [])
    direct_count = sum(item.get("classification") == "direct_target" for item in candidates)
    dependency_count = sum(item.get("classification") == "dependency_linked" for item in candidates)
    metrics = st.columns(5)
    metrics[0].metric("Direct targets", direct_count)
    metrics[1].metric("Dependency candidates", dependency_count)
    metrics[2].metric("Context only", len(context_items))
    metrics[3].metric("Evidence references", len(report.get("evidence", [])))
    metrics[4].metric("Regression ideas", len(tests))

    st.markdown("#### Direct targets and dependency-linked candidates")
    if not candidates:
        st.info("No direct targets or evidence-supported dependency candidates were identified.")
    for rank, candidate in enumerate(candidates, start=1):
        with st.container(border=True):
            score = candidate.get("retrieval_score")
            score_label = f" · ranking score {score:.3f}" if score is not None else ""
            st.text(f"{rank}. {candidate['path']} · {candidate['symbol']}{score_label}")
            st.text(candidate["explanation"])
            if candidate.get("unsupported"):
                st.warning("This candidate contains an unsupported claim.")
            if candidate.get("location_evidence_ids"):
                st.caption(
                    "Source-location evidence: " + ", ".join(candidate["location_evidence_ids"])
                )
            if candidate.get("relationship_evidence_ids"):
                st.caption(
                    "Impact-relationship evidence: "
                    + ", ".join(candidate["relationship_evidence_ids"])
                )

    st.markdown("#### Retrieval-only context")
    st.caption(
        "These ranked source matches establish relevance only; no impact relationship was found."
    )
    if not context_items:
        st.caption("No retrieval-only context symbols were returned.")
    for item in context_items:
        with st.container(border=True):
            score = item.get("retrieval_score")
            score_label = f" · ranking score {score:.3f}" if score is not None else ""
            st.text(f"{item['path']} · {item['symbol']}{score_label}")
            st.text(item["explanation"])
            st.caption(
                "Source-location evidence: " + ", ".join(item.get("location_evidence_ids", []))
            )

    st.markdown("#### Suggested regression tests")
    if not tests:
        st.caption("No test suggestions were generated.")
    for suggestion in tests:
        with st.container(border=True):
            st.text(suggestion["title"])
            st.text(suggestion["behavior"])
            st.caption(
                f"{suggestion.get('classification', 'targeted')} · "
                f"{suggestion.get('status', 'suggested; not implemented or executed')}"
            )
            st.caption("Evidence: " + ", ".join(suggestion.get("evidence_ids", [])))

    st.markdown("#### Retrieved evidence")
    evidence = report.get("evidence", [])
    if not evidence:
        st.caption("No source snippets were selected within the analysis context limits.")
    for item in evidence:
        with st.expander(f"Evidence {item['evidence_id']}"):
            st.text(
                f"{item['path']}:{item['start_line']}-{item['end_line']} · "
                f"{item['symbol']} · {item['kind']}"
            )
            st.code(item["text"], language="python", line_numbers=True)

    st.markdown("#### Uncertainty and unsupported areas")
    uncertainties = report.get("uncertainties", [])
    unsupported = report.get("unsupported_claims", [])
    if not uncertainties and not unsupported:
        st.caption("No additional uncertainty was reported.")
    for message in uncertainties:
        st.text(f"• {message}")
    for message in unsupported:
        st.warning(message)

    st.download_button(
        "Download JSON report",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="changeradar-impact-report.json",
        mime="application/json",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
