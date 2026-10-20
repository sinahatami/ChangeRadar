import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from changeradar.intake import IntakeError
from changeradar.models import SourceFile
from changeradar.ui import (
    analyze_sources,
    configured_provider,
    load_sample_repository,
    load_uploaded_repository,
    report_json,
)


def test_sample_repository_and_local_analysis_are_connected_end_to_end():
    sources = load_sample_repository()
    assert sources

    report = analyze_sources(
        sources,
        "Change calculate_tax rounding for zero tax rates",
        retrieval_method="lexical",
    )

    assert report.mode == "deterministic"
    assert report.retrieval_method == "lexical"
    assert report.effective_retrieval_method == "lexical"
    assert report.affected_items
    assert any(item.path == "billing.py" for item in report.affected_items)
    assert report.evidence
    assert all(1 <= item.start_line <= item.end_line for item in report.evidence)
    assert '"mode": "deterministic"' in report_json(report)


def test_uploaded_zip_uses_existing_safe_intake_and_ignores_non_python_files():
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("src/example.py", "def update_total(total):\n    return total\n")
        zipped.writestr("README.md", "not source code")
        zipped.writestr(".venv/lib/ignored.py", "def ignored(): pass")

    sources = load_uploaded_repository(archive.getvalue())
    assert [(source.path, source.content) for source in sources] == [
        ("src/example.py", "def update_total(total):\n    return total\n")
    ]


def test_bad_zip_and_zip_without_python_sources_are_rejected():
    with pytest.raises(IntakeError):
        load_uploaded_repository(b"not a zip")

    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("README.md", "No Python source here")
    with pytest.raises(IntakeError, match="no supported Python"):
        load_uploaded_repository(archive.getvalue())


@pytest.mark.parametrize(
    ("sources", "change", "message"),
    [
        ([], "change something", "No Python source"),
        ([SourceFile("a.py", "pass\n")], "", "Enter a proposed"),
        ([SourceFile("a.py", "pass\n")], "x" * 30_001, "30,000"),
    ],
)
def test_analysis_rejects_empty_or_oversized_inputs(sources, change, message):
    with pytest.raises(ValueError, match=message):
        analyze_sources(sources, change)


def test_invalid_retrieval_mode_is_rejected():
    with pytest.raises(ValueError, match="Choose lexical"):
        analyze_sources([SourceFile("a.py", "def f(): pass\n")], "change f", "unknown")


def test_external_provider_requires_ui_consent_and_existing_environment_opt_in(monkeypatch):
    monkeypatch.setenv("CHANGERADAR_ALLOW_EXTERNAL_CODE", "true")
    monkeypatch.setenv("CHANGERADAR_LLM_ENDPOINT", "https://llm.example/v1/chat/completions")
    monkeypatch.setenv("CHANGERADAR_LLM_MODEL", "test-model")
    monkeypatch.setenv("CHANGERADAR_API_KEY", "test-secret")

    assert configured_provider(False) is None
    provider = configured_provider(True)
    assert provider is not None
    assert provider.model == "test-model"
    assert "test-secret" not in repr(provider.endpoint)

    monkeypatch.setenv("CHANGERADAR_ALLOW_EXTERNAL_CODE", "false")
    assert configured_provider(True) is None


def test_unconsented_provider_is_not_used_even_if_passed_to_helper():
    class UnwantedProvider:
        def complete(self, messages, tools):
            raise AssertionError("provider must not be called without consent")

    report = analyze_sources(
        [SourceFile("billing.py", "def calculate_tax(amount):\n    return amount\n")],
        "Change calculate_tax behavior",
        retrieval_method="lexical",
        external_consent=False,
        provider=UnwantedProvider(),
    )
    assert report.mode == "deterministic"


def test_streamlit_app_sample_flow_when_optional_ui_extra_is_installed():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = streamlit_testing.AppTest.from_file(app_path).run()
    app.text_area[0].set_value("Change calculate_tax rounding for zero tax rates")
    app.button[0].click().run()

    assert not app.exception
    assert any("Impact report" in str(element.value) for element in app.markdown)
    assert [button.label for button in app.get("download_button")] == ["Download JSON report"]
