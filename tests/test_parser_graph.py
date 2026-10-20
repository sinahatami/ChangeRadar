import io
import zipfile

from changeradar.graph import build_analysis
from changeradar.intake import scan_zip
from changeradar.models import SourceFile
from changeradar.parser import parse_source


def test_parser_extracts_symbol_lines_calls_and_imports():
    source = SourceFile(
        "pkg/sample.py",
        "import os\nfrom pkg.util import helper\n\nclass Worker:\n"
        "    def run(self):\n        helper()\n\ndef top():\n    os.getcwd()\n",
    )
    parsed = parse_source(source)
    assert parsed.error is None
    assert parsed.imports == ["os", "pkg.util"]
    symbols = {symbol.qualified_name: symbol for symbol in parsed.symbols}
    assert (symbols["Worker"].start_line, symbols["Worker"].end_line) == (4, 6)
    assert symbols["Worker.run"].calls == ("helper",)
    assert symbols["top"].calls == ("os.getcwd",)


def test_parser_records_syntax_error():
    parsed = parse_source(SourceFile("broken.py", "def bad(:\n pass"))
    assert parsed.error and parsed.error.startswith("SyntaxError:")
    assert not parsed.symbols


def test_graph_links_local_import_and_unique_call():
    archive = _zip(
        {
            "pkg/main.py": "from pkg.util import helper\ndef run():\n    helper()\n",
            "pkg/util.py": "def helper():\n    return 1\n",
        }
    )
    analysis = build_analysis([parse_source(source) for source in scan_zip(archive)])
    edges = {(edge.source, edge.target, edge.kind) for edge in analysis.dependencies}
    assert ("pkg.main", "pkg.util", "import") in edges
    assert ("pkg.main:run", "pkg.util:helper", "call") in edges
    call_edge = next(edge for edge in analysis.dependencies if edge.kind == "call")
    assert call_edge.evidence_line == 3


def _zip(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return output.getvalue()
