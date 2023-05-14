import networkx as nx
import pytest
from glycowork.motif.graph import glycan_to_nxGraph, ensure_graph

from glycan_utils.utils.annotate import list_contains_glycan

GLYCAN_GRAPH = glycan_to_nxGraph("GlcNGc")


@pytest.mark.parametrize(
    "input_glycan",
    [
        "GlcNGc",
        GLYCAN_GRAPH
    ],
)
def test_ensure_graph(input_glycan: str):
    glycan_graph = ensure_graph(input_glycan)
    assert isinstance(glycan_graph, nx.Graph)


@pytest.mark.parametrize(
    "glycan,expected",
    [
        ("GlcNAc", True),
        ("Neu5Ac", False)
    ],
)
def test_list_contains_glycan(glycan, expected):
    glycan_list = ["GlcNAc", "Rha"]
    if expected:
        assert list_contains_glycan(glycan=glycan, glycan_list=glycan_list)
    else:
        assert not list_contains_glycan(glycan, glycan_list)