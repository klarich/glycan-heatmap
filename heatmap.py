from typing import Optional, List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from utils.logger import get_logger

from glycowork.glycan_data.loader import lib
from glycowork.motif.annotate import annotate_dataset

LOGGER = get_logger(__name__)


def make_clustermap(
    df: pd.DataFrame,
    feature_set: str = "known",
    rarity_filter: float = 0.05,
    filepath: Optional[str] = None,
    index_col: str = "target",
    motifs: pd.DataFrame = None,
    csv_filepath: str = None,
    **kwargs,
):
    """
    *** Adapted from glycowork function make_heatmap ***
    :param df: dataframe with glycan array data, each row is a sample (glycan array) and columns are glycans in IUPAC
    format
    :param feature_set: which feature set to use for annotations, add more to list to expand; default is 'exhaustive';
    options are: 'known' (use 'known' motifs'), 'exhaustive' (all mono- and disaccharide features in the dataset)
    :param rarity_filter: proportion of samples that must have a feature for it to be included in the dataset. For
    example, if the rarity filter is set to 0.02, 2% of glycans in the dataset will need to contain a motif for it to be
    included in the clustermap
    :param filepath: Filepath to save the figure to
    :param index_col: Column that will be converted to dataframe index and ultimately used to label the samples in the
    clustermap
    :param motifs: dataframe of motifs to use in the plotting. Must contain the following columns: motif_name (should be
    unique), motif (as a string), and optionally, termini_spec - a list detailing the positions of each node,
    including bonds, in the glycan graph. See glycowork.glycan_data.loader.motif_list for an example of the dataframe
    format.
    :param csv_filepath:
    :param kwargs: filepath to save the dataframe that is used to generate the clustermap
    :return: None
    """
    feature_set = validate_feature_set(feature_set)
    df = set_index(df, index_col)
    # Replace NA with 0
    df = df.fillna(0)
    df_motif = create_motif_df(
        glycans=df.columns.values.tolist(), feature_set=feature_set, motifs=motifs, rarity_filter=rarity_filter
    )
    df = calculate_mean_response(df, df_motif)
    cluster_map_df = df.T
    if csv_filepath is not None:
        cluster_map_df.to_csv(csv_filepath, index=False)
        LOGGER.info(f"Saved dataset to: {csv_filepath}")
    sns.clustermap(cluster_map_df, **kwargs)
    plt.xlabel("Samples")
    plt.ylabel("Motifs")
    plt.tight_layout()
    if filepath is not None:
        plt.savefig(filepath, format=filepath.split(".")[-1], dpi=300, bbox_inches="tight")
        LOGGER.info(f"Saved clustermap to: {filepath}")
    else:
        plt.show()


def calculate_mean_response(df: pd.DataFrame, df_motif: pd.DataFrame) -> pd.DataFrame:
    """
    For each motif in motif_df, calculate the mean response for all of the glycans containing that motif for each sample
    :param df: Dataframe of glycans and response data. Each column is a glycan, each row is a sample. Values are glycan
    array response data
    :param df_motif: dataframe of motifs where each column is a motif and each row is a glycan. Values are the count of
    the motif in each glycan.
    :return: Dataframe of the mean response for each motif
    """
    collect_dic = {}
    # Now iterate over the motifs and calculate the mean value of all the glycans containing the motif in the sample
    for col in df_motif.columns.values.tolist():
        indices = [i for i, x in enumerate(df_motif[col].values.tolist()) if x >= 1]
        mean_response = np.mean(df.iloc[:, indices], axis=1)
        collect_dic[col] = mean_response
    df = pd.DataFrame(collect_dic)
    df.dropna(axis=1, inplace=True)
    return df


def create_motif_df(
    glycans: List[str], feature_set: List[str], motifs: pd.DataFrame | None, rarity_filter: float
) -> pd.DataFrame:
    """
    Creates a motif dataframe, where each column is a motif, each row is a glycan, and the values are the counts of the
    motif if each glycan. Filters the motifs so that each motif must occur in at leat X% of the glycan (as specified by
    the rarity filter) for it to be included in the final dataframe.
    :param glycans: Input dataframe where each column is a clyan
    :param feature_set: which feature set to use for annotations, add more to list to expand; default is 'exhaustive';
    options are: 'known' (use 'known' motifs'), 'exhaustive' (all mono- and disaccharide features in the dataset)
    :param motifs: Motifs to use. If None, then feature_set should be 'exhaustive' and all mono- and di- saccharide
    glycan features will be considered
    :param rarity_filter: Minimum % of glycans a motif must be found in to be included in the final dataframe
    :return: motif dataframe where each column is a motif and each row is a glycan
    """
    # Create a dataframe of all the motifs in the dataset. Columns contain the motif, each row contains a glycan, the
    # value of the dataframe contains the count of each motif in the glycan.
    df_motif = annotate_dataset(
        glycans=glycans,
        libr=lib,
        feature_set=feature_set,
        extra="termini",
        estimate_speedup=False,
        motifs=motifs,
    )
    df_motif = df_motif.replace(0, np.nan)
    LOGGER.info(f"Found {len(df_motif.columns)} motifs in dataset")
    # Now remove motifs that do not meet the criteria for the rarity filter that are not in X% of the glycans,
    # as defined by the rarity filter. The "threshold" specifies the % of not na values that are needed to prevent the
    # motif in each column from being dropped.
    df_motif = df_motif.dropna(thresh=np.max([np.round(rarity_filter * df_motif.shape[0]), 1]), axis=1)
    LOGGER.info(f"{len(df_motif.columns)} motifs remain after applying rarity filter of " f"{rarity_filter * 100}%")
    return df_motif


def set_index(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    """
    Creates a new index in the provided dataframe using index_col
    :param df: dataframe
    :param index_col: col to create index from
    :return: dataframe with new index
    """
    if index_col in df.columns.values.tolist():
        # Create the index sith index_col
        df.index = df[index_col]
        df.drop([index_col], axis=1, inplace=True)
    else:
        raise ValueError(f"Did not find index col: {index_col} in df.columns")
    return df


def validate_feature_set(feature_set) -> List[str]:
    """
    Ensures that the feature set is a valid feature set, converts to format needed for annotate_dataset
    :param feature_set: feature set, should be one of 'known', 'exhaustive'
    :return: feature set in the format needed for annotate_dataset function (i.e. list of string)
    """
    allowed_feature_sets = ["known", "exhaustive"]
    if feature_set not in allowed_feature_sets:
        raise ValueError(f"Feature set should be one of {allowed_feature_sets}, found: {feature_set}")
    feature_set = [feature_set]
    return feature_set
