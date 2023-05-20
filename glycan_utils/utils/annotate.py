from collections import defaultdict
from typing import List, Iterable, Tuple
import networkx as nx
import pandas as pd
from glycowork.glycan_data.loader import lib
from glycowork.motif.graph import (
    glycan_to_nxGraph,
    compare_glycans,
    graph_to_string,
)
from networkx import Graph

from glycan_utils.utils.logger import get_logger
from glycan_utils.utils.types import Glycan

LOGGER = get_logger(__name__)


def ensure_graph(glycan: Glycan, libr: List[str] = lib, **kwargs) -> nx.Graph:
    """
    *** adapted from glycaowork function ensure_graph ***
    Ensures that the glycan being used is a networkx Graph object
    :param glycan: Glycan as either a string or nx.Graph
    :param kwargs: additional keyword arguments to pass to glycan_to_nxGraph
    :param libr: Library of monosaccharides to use. If not specified, will use the default glycowork library
    :return: the glycan as a network x Graph object
    """
    if isinstance(glycan, str):
        return glycan_to_nxGraph(glycan, libr=libr, **kwargs)
    else:
        return glycan


def list_contains_glycan(glycan: Glycan, glycan_list: List[Glycan]) -> bool:
    """
    Checks whether a glycan is in the list of glycans supplied.
    :param glycan: glycan string or network x graph obejct
    :param glycan_list: list of glycans to compare agains
    :return: bool, True if glycan is in list, else false
    """
    return any([compare_glycans(glycan, glycan_b) for glycan_b in glycan_list])


def glycan_from_subgraph(subgraph: nx.Graph) -> nx.Graph:
    """
    Helper function for _get_terminal_motifs - converts a subgraph to a standalone graph so that it can be
    used with other glycan graph functions such as graph_to_string.
    :param subgraph: glycan subgraph (network networkx.Graph object)
    :return: glycan graph
    """
    # Make a copy of the subgraph and sort the nodes so that they are in order (which is neceesary to work with string
    # to graph
    graph = nx.Graph()
    graph.add_nodes_from(sorted(subgraph.nodes(data=True)))
    graph.add_edges_from(subgraph.edges(data=True))
    # relabel nodes starting with 0
    return nx.relabel_nodes(graph, {node: idx for idx, node in enumerate(graph.nodes())})


def get_terminal_motifs(glycan: Glycan, size: int, libr: List[str] = lib) -> Tuple[List[str], List[List[str]]]:
    """
    *** Adapted from glycowork function get_k_saccharides ***
    Gets the terminal motifs containing the number of monosaccharides, specified by the size, from the non-reducing ends
    of the glycan.
    :param glycan: Instance of Glycan
    :param size: size of motif to find (number of monosaccarides
    :param libr: Library of monosaccharides to use. If not specified, will use the default glycowork library
    :return: List of terminal motifs in string format, list of position labels (internal or terminal) for each
    monosaccharide and bond in the motif.
    """
    # The size has to account for both saccharides and bonds since each bond is also a node in the graph
    size = size * 2 - 1
    ggraph = ensure_graph(glycan, libr=libr, termini="calc")
    # Ensures that the termini of the reducing end are not labeled as "terminal"
    fix_reducing_end_termini(ggraph)
    if len(ggraph.nodes()) < size:
        return [], []
    subgraphs = []
    reducing_end_node = max([node for node in ggraph.nodes()])

    # define a recursive function to traverse the graph
    def traverse(node: int, path: List[int]) -> None:
        """
        Function to recursively build a subgraph (motif) for a given node with the specified size
        :param node: Starting node
        :param path: path that has been traversed so far
        :return: None
        """
        # TODO: refactor this function
        # add the current node to the path
        path.append(node)

        # check if the path has the desired subgraph size
        if len(path) == size:
            # TODO: this could just return the path
            # add the path as a subgraph to the list of subgraphs
            subgraph = ggraph.subgraph(path).copy()
            # check to make sure the subgraph is not already in the list
            if not list_contains_glycan(subgraph, subgraphs):
                subgraph = glycan_from_subgraph(subgraph)
                subgraphs.append(subgraph)
        # First, make sure the node is not the reducing end because we don't want to get neighbors of the reducing end
        if node == reducing_end_node:
            LOGGER.debug(f"Found reducing end node: {node}, not traversing neighbors")
        elif ggraph.degree[node] > 2:  # noqa
            LOGGER.debug(
                f"Not traversing neighbors because node {node} {ggraph.nodes[node]['string_labels']} degree > "
                f"2: {ggraph.degree[node]}"  # noqa
            )
        else:
            # iterate over the neighbors of the current node
            for neighbor in ggraph[node]:
                # check if the neighbor is already in the path
                if neighbor not in path:
                    # traverse the neighbor
                    traverse(neighbor, path)

        # remove the current node from the path
        path.pop()
        return

    for node in ggraph.nodes:
        if ggraph.nodes[node]["termini"] == "terminal":
            traverse(node, [])

    position_labels = [[subgraph.nodes[node]["termini"] for node in subgraph.nodes] for subgraph in subgraphs]
    # Convert the subsgraphs to strings
    subgraphs = [graph_to_string(k) for k in subgraphs]
    # Make sure the subgraph does not start with a bond
    for s in subgraphs:
        if s[0] in ["a", "b", "?"]:
            raise ValueError(f"Motif should not start with a bond, found {s}")
    # return the list of subgraphs
    return subgraphs, position_labels


def fix_reducing_end_termini(glycan_graph: Graph) -> Graph:
    """
    Changes the "termini" of the monosaccharide of the reducing end of the glycan to be "internal"
    :param glycan_graph: network X graph object of the glycan
    :return: glycan graph
    """
    reducing_end_node = max(glycan_graph.nodes)
    reducing_end_termini = glycan_graph.nodes[reducing_end_node]["termini"]
    if len(glycan_graph.nodes) > 2:  # Need to have more than one monosaccharide (bonds also count as a node)
        if reducing_end_termini == "terminal":
            glycan_graph.nodes[reducing_end_node]["termini"] = "internal"
        else:
            LOGGER.debug(f"Reducing end termini set to {reducing_end_termini} - leaving as is")
    else:
        LOGGER.info(f"Only one node in glycan_graph. Leaving reducing end termini as is.")
    return glycan_graph


def get_terminal_motifs_dataframe(glycans: Iterable[str], max_size: int = 3) -> pd.DataFrame:
    """
    Generates a dataframe containing terminal motifs of the glycans supplied. The dataframe will contain three columns:
    motif, termini_spec and motif_name, the contents of which are identical to motif. The dataframe is formatted so that
    it can easily be passed to make_heatmap function.
    :param glycans: List of glycans to generate terminal motifs for
    :param max_size: The maximum size of motifs to generate. For example, if max_size = 2, will generate motifs
    containing 1 and two monosaccharides
    :return:
    """
    motifs_dict = defaultdict(list)
    for glycan in glycans:
        for idx in range(max_size):
            size = idx + 1
            motifs, positions_lists = get_terminal_motifs(glycan, size=size)
            positions_lists = [str(positions) for positions in positions_lists]
            if len(motifs) != len(positions_lists):
                raise ValueError(f"Mismatch in len between motifs: {motifs} and positions lists: {positions_lists}")
            motifs_dict["motif"] += motifs
            motifs_dict["termini_spec"] += positions_lists
    motifs_dict["motif_name"] = motifs_dict["motif"].copy()
    motifs_df = pd.DataFrame(motifs_dict)
    motifs_df = motifs_df.drop_duplicates(ignore_index=True)
    return motifs_df
