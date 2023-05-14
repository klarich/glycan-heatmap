from collections import defaultdict
from typing import Union, List, Iterable, Tuple
import networkx as nx
import pandas as pd
from glycowork.glycan_data.loader import lib, motif_list
from glycowork.motif.annotate import estimate_lower_bound, motif_matrix, get_molecular_properties
from glycowork.motif.graph import (
    glycan_to_nxGraph,
    subgraph_isomorphism,
    generate_graph_features,
    compare_glycans,
    graph_to_string,
)
from networkx import Graph

from glycan_utils.utils.logger import get_logger
from glycan_utils.utils.types import Glycan

LOGGER = get_logger(__name__)


def ensure_graph(glycan: Glycan, libr: List[str] = lib, **kwargs) -> nx.Graph:
    """
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


def _glycan_from_subgraph(subgraph: nx.Graph) -> nx.Graph:
    """
    Helper function for _get_terminal_motifs - updates the subgraph containing the terminal motifs so that it can be a
    standalone glycan and works with graph_to_string
    :param subgraph: glycan subgraph (network networkx.Graph object)
    :return: glycan subgraph
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
    Gets the terminal motifs containing the number of monosaccharides, specified by the size, from the non-reducing ends
    :param glycan: input glycan
    :param size: Size (i.e. number of monosaccharides) of motif to get
    :param libr: library of monosaccharides to use
    :return: a list of terminal structures (in string format) and a list of position labels (terminal / internal) for
    each node in the motif
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
                subgraph = _glycan_from_subgraph(subgraph)
                subgraphs.append(subgraph)
        # First, make sure the node is not the reducing end because we don't want to get neighbors of the reducing end
        if node == reducing_end_node:
            LOGGER.debug(f"Found reducing end node: {node}, not traversing neighbors")
        elif ggraph.degree[node] > 2:
            LOGGER.debug(
                f"Not traversing neighbors because node {node} {ggraph.nodes[node]['string_labels']} degree > "
                f"2: {ggraph.degree[node]}"
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


def annotate_glycan(
    glycan: Union[str, Graph],
    motifs: pd.DataFrame = motif_list,
    libr: List[str] = lib,
    extra="termini",
    wildcard_list=None,
    termini_list=None,
    set_reducing_end_termini_to_internal: bool = True,
) -> pd.DataFrame:
    """
    Searches for known motifs in glycan sequence. If a motifs dataframe is supplied, those motifs will be used.
    Otherwise the default motif_list dataframe containing a comprehensive set of known motifs will be used.
    :param glycan: Input glycan to find motifs in
    :param motifs: Dataframe of motifs. The columns should contain motif_name and motif and,optionally, termini_spec (to
    specify whether each monosaccharide is internal, terminal or flexible
    :param libr: library of monosaccharides to use
    :param extra: 'ignore' skips this, 'wildcards' allows for wildcard matching', and 'termini' allows for positional
    matching; default:'termini'
    :param wildcard_list: list of wildcard names (such as 'z1-z', 'Hex', 'HexNAc', 'Sia')
    :param termini_list: list of monosaccharide/linkage positions for the glycan (from 'terminal','internal', and
    'flexible')
    :param set_reducing_end_termini_to_internal: Sets the termini of the reducing end to "internal"
    :return: A dataframe with the count of each motif in the glycan
    """
    wildcard_list = [] if wildcard_list is None else wildcard_list
    termini_list = [] if termini_list is None else termini_list
    # check whether termini are specified
    if extra == "termini":
        if len(termini_list) < 1:
            termini_list = motifs.termini_spec.values.tolist()
            termini_list = [eval(k) for k in termini_list]
    # count the number of times each motif occurs in a glycan
    if extra == "termini":
        ggraph = ensure_graph(glycan=glycan, libr=libr, termini="calc")
        if set_reducing_end_termini_to_internal:
            ggraph = fix_reducing_end_termini(glycan_graph=ggraph)
        res = [
            subgraph_isomorphism(
                ggraph,
                motifs.motif.values.tolist()[k],
                libr=libr,
                extra=extra,
                wildcard_list=wildcard_list,
                termini_list=termini_list[k],
                count=True,
            )
            for k in range(len(motifs))
        ] * 1
    else:
        ggraph = ensure_graph(glycan=glycan, libr=libr, termini="ignore")
        res = [
            subgraph_isomorphism(
                ggraph,
                motifs.motif.values.tolist()[k],
                libr=libr,
                extra=extra,
                wildcard_list=wildcard_list,
                termini_list=termini_list,
                count=True,
            )
            for k in range(len(motifs))
        ] * 1

    out = pd.DataFrame(columns=motifs.motif_name.values.tolist())
    out.loc[0] = res
    out.loc[0] = out.loc[0].astype("int")
    out.index = [glycan]
    return out


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


# def annotate_dataset(
#     glycans,
#     motifs=None,
#     libr=None,
#     feature_set=["known"],
#     extra="termini",
#     wildcard_list=[],
#     termini_list=[],
#     condense=False,
#     estimate_speedup=False,
# ):
#     """wrapper function to annotate motifs in list of glycans\n
#     | Arguments:
#     | :-
#     | glycans (list): list of IUPAC-condensed glycan sequences as strings
#     | motifs (dataframe): dataframe of glycan motifs (name + sequence); default:motif_list
#     | libr (list): sorted list of unique glycoletters observed in the glycans of our data; default:lib
#     | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'known'; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), and 'chemical' (molecular properties of glycan)
#     | extra (string): 'ignore' skips this, 'wildcards' allows for wildcard matching', and 'termini' allows for positional matching; default:'termini'
#     | wildcard_list (list): list of wildcard names (such as '?1-?', 'Hex', 'HexNAc', 'Sia')
#     | termini_list (list): list of monosaccharide/linkage positions (from 'terminal','internal', and 'flexible')
#     | condense (bool): if True, throws away columns with only zeroes; default:False
#     | estimate_speedup (bool): if True, pre-selects motifs for those which are present in glycans, not 100% exact; default:False\n
#     | Returns:
#     | :-
#     | Returns dataframe of glycans (rows) and presence/absence of known motifs (columns)
#     """
#     if motifs is None:
#         motifs = motif_list
#     if libr is None:
#         libr = lib
#     # non-exhaustive speed-up that should only be used if necessary
#     if estimate_speedup:
#         motifs = estimate_lower_bound(glycans, motifs)
#     # checks whether termini information is provided
#     if extra == "termini":
#         if len(termini_list) < 1:
#             termini_list = motifs.termini_spec.values.tolist()
#             termini_list = [eval(k) for k in termini_list]
#     shopping_cart = []
#     if "known" in feature_set:
#         # counts literature-annotated motifs in each glycan
#         shopping_cart.append(
#             pd.concat(
#                 [
#                     annotate_glycan(
#                         k,
#                         motifs=motifs,
#                         libr=libr,
#                         extra=extra,
#                         wildcard_list=wildcard_list,
#                         termini_list=termini_list,
#                         set_reducing_end_termini_to_internal=True,
#                     )
#                     for k in glycans
#                 ],
#                 axis=0,
#             )
#         )
#     if "graph" in feature_set:
#         # calculates graph features of each glycan
#         shopping_cart.append(pd.concat([generate_graph_features(k, libr=libr) for k in glycans], axis=0))
#     if "exhaustive" in feature_set:
#         # counts disaccharides and glycoletters in each glycan
#         temp = motif_matrix(
#             pd.DataFrame({"glycans": glycans, "labels": range(len(glycans))}), "glycans", "labels", libr=libr
#         )
#         temp.index = glycans
#         temp.drop(["labels"], axis=1, inplace=True)
#         shopping_cart.append(temp)
#     if "chemical" in feature_set:
#         shopping_cart.append(get_molecular_properties(glycans, placeholder=True))
#     if condense:
#         # remove motifs that never occur
#         temp = pd.concat(shopping_cart, axis=1)
#         return temp.loc[:, (temp != 0).any(axis=0)]
#     else:
#         return pd.concat(shopping_cart, axis=1)


def get_terminal_motifs_dataframe(glycans: Iterable[str], max_size: int = 3) -> pd.DataFrame:
    """
    Generates a dataframe containing terminal motifs of the glycans supplied. The dataframe will contain three columns:
    motif, termini_spec and motif_name (identical to motif). The dataframe is formatted so that it can easily be passed
    to make_heatmap function.
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
