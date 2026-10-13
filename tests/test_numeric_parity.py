import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical(node):
    # Stable across Python AST versions; generic type parameters are unused here.
    if isinstance(node, ast.AST):
        return {"type": type(node).__name__,
                "fields": {name: canonical(value) for name, value in ast.iter_fields(node)
                           if name != "type_params"}}
    if isinstance(node, list):
        return [canonical(value) for value in node]
    return node


def test_original_flow_sampler_and_auxiliary_numerics_unchanged():
    expected = json.loads((ROOT / "tests/original_numeric_ast.json").read_text())
    for filename, signatures in expected.items():
        tree = ast.parse((ROOT / "spf_ssp126" / filename).read_text())
        nodes = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                nodes[node.name] = node
            elif isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        nodes[node.name + "." + method.name] = method
        for name, signature in signatures.items():
            dumped = json.dumps(canonical(nodes[name]), sort_keys=True, separators=(",", ":"))
            assert hashlib.sha256(dumped.encode()).hexdigest() == signature
