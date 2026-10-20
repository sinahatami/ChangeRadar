from changeradar.graph import build_analysis
from changeradar.intake import scan_zip
from changeradar.models import SourceFile
from changeradar.parser import parse_source


def test_parser_extracts_functions_classes_calls_imports_and_lines():
    parsed = parse_source(
        SourceFile(
            "pkg/sample.py",
            "import os\nfrom pkg.util import helper\n\nclass Worker:\n"
            "    def run(self):\n        helper()\n\ndef top():\n    os.getcwd()\n",
        )
    )
    assert parsed.error is None
    assert parsed.imports == ["os", "pkg.util"]
    by_name = {symbol.qualified_name: symbol for symbol in parsed.symbols}
    assert by_name["Worker"].kind == "class"
    assert (by_name["Worker"].start_line, by_name["Worker"].end_line) == (4, 6)
    assert by_name["Worker.run"].calls == ("helper",)
    assert by_name["top"].calls == ("os.getcwd",)


def test_parser_reports_syntax_errors_without_raising():
    parsed = parse_source(SourceFile("bad.py", "def broken(:\n pass"))
    assert parsed.symbols == []
    assert parsed.error.startswith("SyntaxError:")


def test_dependency_graph_links_local_imports_and_unique_calls():
    sources = scan_zip(
        _zip(
            {
                "pkg/main.py": "from pkg.util import helper\ndef run():\n    helper()\n",
                "pkg/util.py": "def helper():\n    return 1\n",
            }
        )
    )
    analysis = build_analysis([parse_source(source) for source in sources])
    edges = {(edge.source, edge.target, edge.kind) for edge in analysis.dependencies}
    assert ("pkg.main", "pkg.util", "import") in edges
    assert ("pkg.main:run", "pkg.util:helper", "call") in edges


def test_ambiguous_call_names_are_not_linked():
    files = [
        parse_source(SourceFile("a.py", "def same(): pass\ndef caller(): same()")),
        parse_source(SourceFile("b.py", "def same(): pass")),
    ]
    analysis = build_analysis(files)
    assert not [edge for edge in analysis.dependencies if edge.kind == "call"]


def _zip(entries):
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()
