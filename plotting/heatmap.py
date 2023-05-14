import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from utils.logger import get_logger

from glycowork.glycan_data.loader import lib
from glycowork.motif.annotate import annotate_dataset

LOGGER = get_logger(__name__)


def make_heatmap(
        df,
        mode="sequence",
        libr=None,
        feature_set=["known"],
        extra="termini",
        wildcard_list=[],
        datatype="response",
        rarity_filter=0.05,
        filepath="",
        index_col="target",
        estimate_speedup=False,
        motifs: pd.DataFrame = None,
        csv_filepath: str = None,
        **kwargs,
):
    """clusters samples based on glycan data (for instance glycan binding etc.)\n
    | Arguments:
    | :-
    | df (dataframe): dataframe with glycan data, rows are samples and columns are glycans
    | mode (string): whether glycan 'sequence' or 'motif' should be used for clustering; default:sequence
    | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset
    | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'exhaustive'; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), and 'chemical' (molecular properties of glycan)
    | extra (string): 'ignore' skips this, 'wildcards' allows for wildcard matching', and 'termini' allows for positional matching; default:'termini'
    | wildcard_list (list): list of wildcard names (such as 'bond', 'Hex', 'HexNAc', 'Sia')
    | datatype (string): whether df comes from a dataset with quantitative variable ('response') or from presence_to_matrix ('presence')
    | rarity_filter (float): proportion of samples that need to have a non-zero value for a variable to be included; default:0.05
    | filepath (string): absolute path including full filename allows for saving the plot
    | index_col (string): default column to convert to dataframe index; default:'target'
    | estimate_speedup (bool): if True, pre-selects motifs for those which are present in glycans, not 100% exact; default:False
    | motifs: dataframe of motifs to use in the heatmap. Must contain the following columns: motif_name (should be
    | unique), motif (as a string), and optionally, termini_spec - a list detailing the positions of each node,
    | including bonds, in the glycan graph. See glycowork.glycan_data.loader.motif_list for an example of the dataframe
    | format
    | csv_filepath: filepath to save the dataframe that is used to generate the clustermap
    | **kwargs: keyword arguments that are directly passed on to seaborn clustermap\n
    | Returns:
    | :-
    | Prints clustermap
    """
    if libr is None:
        libr = lib
    if index_col in df.columns.values.tolist():
        df.index = df[index_col]
        df.drop([index_col], axis=1, inplace=True)
    df = df.fillna(0)
    if mode == "motif":
        # count glycan motifs and remove rare motifs from the result
        df_motif = annotate_dataset(
            df.columns.values.tolist(),
            libr=libr,
            feature_set=feature_set,
            extra=extra,
            wildcard_list=wildcard_list,
            estimate_speedup=estimate_speedup,
            motifs=motifs,
        )
        df_motif = df_motif.replace(0, np.nan).dropna(
            thresh=np.max([np.round(rarity_filter * df_motif.shape[0]), 1]), axis=1
        )
        collect_dic = {}
        # distinguish the case where the motif abundance is paired to a quantitative value or a qualitative variable
        if datatype == "response":
            for col in df_motif.columns.values.tolist():
                indices = [i for i, x in enumerate(df_motif[col].values.tolist()) if x >= 1]
                temp = np.mean(df.iloc[:, indices], axis=1)
                collect_dic[col] = temp
            df = pd.DataFrame(collect_dic)
        elif datatype == "presence":
            idx = df.index.values.tolist()
            collecty = [
                [
                    np.sum(df.iloc[row, [i for i, x in enumerate(df_motif[col].values.tolist()) if x >= 1]])
                    / df.iloc[row, :].values.sum()
                    for col in df_motif.columns.values.tolist()
                ]
                for row in range(df.shape[0])
            ]
            df = pd.DataFrame(collecty)
            df.columns = df_motif.columns.values.tolist()
            df.index = idx
    df.dropna(axis=1, inplace=True)
    # cluster the motif abundances
    cluster_map_df = df.T
    if csv_filepath is not None:
        cluster_map_df.to_csv(csv_filepath, index=False)
        LOGGER.info(f"Saved heatmap dataset to: {csv_filepath}")
    sns.clustermap(cluster_map_df, **kwargs)
    plt.xlabel("Samples")
    if mode == "sequence":
        plt.ylabel("Glycans")
    else:
        plt.ylabel("Motifs")
    plt.tight_layout()
    if len(filepath) > 1:
        plt.savefig(filepath, format=filepath.split(".")[-1], dpi=300, bbox_inches="tight")
        LOGGER.info(f"Saved heatmap to: {filepath}")
    plt.show()
