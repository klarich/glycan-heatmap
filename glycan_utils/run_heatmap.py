import random
from typing import Optional

import fire
import pandas as pd
from glycowork.glycan_data.loader import glycan_binding as gb_df

from glycan_utils.heatmap import make_clustermap
from glycan_utils.transformations import run_power_transformation
from glycan_utils.utils.annotate import get_terminal_motifs_dataframe
from glycan_utils.utils.logger import get_logger


LOGGER = get_logger(__name__)


def generate_heatmap(
    df: pd.DataFrame,
    output_png_filepath: str,
    output_csv_filepath: Optional[str] = None,
    feature_set: str = "known",
    max_size: int = 2,
    terminal_motifs_only: bool = True,
):
    """
    Function to generate a glycan binding plotting
    :param df: Dataframe to make plotting with. Should be in the format of the glycowork glycan_binding
    dataframe; columns are glycans in IUPAC format and their values are the binding interactions to be plotted.
    :param output_png_filepath: Filepath to save the plotting to
    :param output_csv_filepath: Filepath to save the csv data used to generate the plotting
    :param max_size: maximum size of the monosaccharide to use (only applies to the terminal motifs dataframe)
    :param feature_set: either "known" or "exhaustive". Known should be selected if trying to make a plotting of terminal
    motifs
    :param terminal_motifs_only: Whether to generate and use a dataframe of terminal motifs only. Otherwise the default
    motif_list dataframe from glycowork will be used, if the feature set is "known"
    :return: None
    """
    df['target_id'] = [f"seq_{idx}" for idx, _ in df.iterrows()]
    # Drops the protein and target columns if they exist
    target_df = df[['target_id', 'protein']]
    df = df.drop(columns=["target", 'protein'])
    df = df.set_index("target_id")
    df = df.T
    # Drop all columns with no data
    df = df.dropna(how='all', axis=1)
    df = run_power_transformation(df)
    glycans = set(df.columns)
    glycans.remove("target_id")
    LOGGER.info(f"Creating motif dataframe")
    motifs_df = get_terminal_motifs_dataframe(glycans, max_size=max_size) if terminal_motifs_only else None
    LOGGER.info(f"Making clustermap")
    df = df.merge(target_df, how='left', on='target_id')
    df = df.drop(columns='target_id')
    make_clustermap(
        df=df,
        index_col="protein",
        feature_set=feature_set,
        yticklabels=True,
        xticklabels=True,
        rarity_filter=0.02,
        filepath=output_png_filepath,
        figsize=(12, 12),
        motifs=motifs_df,
        csv_filepath=output_csv_filepath,
    )
    return


def main(
    output_png_filepath: str = None,
    proteins_csv_filepath: str = None,
    feature_set: str = "terminal",
    max_size: int = 2,
    num_proteins: int = 20
):
    """
    Creates an sns.clustermap using the glycan_binding dataset, with optional filtering by protein
    :param output_png_filepath: Optional path to output the heatmap png to
    :param proteins_csv_filepath: Optional path to csv file containing proteins to filter the dataframe by
    :param feature_set: which feature set to use for making the heatmap; options are: 'terminal' for terminal_motifs,
    'known' (hand-crafted glycan features from glycowork), 'exhaustive' (all mono- and disaccharide features)
    :param max_size: maximum size of the motifs
    :param num_proteins If not specifying proteins, the number of proteins to randomly select from the glycan_binding
    dataframe
    :return: None
    """
    allowed_features = {"terminal", "known", "exhaustive"}
    if feature_set not in allowed_features:
        raise ValueError(f"Feature set should be one of: {allowed_features}, found: {feature_set}")
    if proteins_csv_filepath is not None:
        LOGGER.info(f"Getting proteins from: {proteins_csv_filepath}")
        proteins_df = pd.read_csv(proteins_csv_filepath)
        proteins = proteins_df["protein"]
    else:
        LOGGER.info(f"Randomly selecting {num_proteins} proteins from glycan_binding dataframe")
        proteins = random.sample(list(set(gb_df["protein"])), num_proteins)
    df = gb_df[gb_df["protein"].isin(proteins)].copy()
    if df.empty:
        raise ValueError(f"Filtering glycan_binding by proteins led to an empty df. Proteins used = \n{proteins}")
    elif len(df) < 2:
        raise ValueError(f"Clustermap requires at least two glycan arrays to perform clustering. Please specify more "
                         f"proteins")
    LOGGER.info(f"Making clustermap with {len(df)} samples (glycan arrays) and feature set = {feature_set}")
    terminal_motifs_only = True if feature_set == "terminal" else False
    feature_set = "known" if feature_set == "terminal" else feature_set
    generate_heatmap(
        df=df,
        output_png_filepath=output_png_filepath,
        output_csv_filepath=None,
        feature_set=feature_set,
        max_size=max_size,
        terminal_motifs_only=terminal_motifs_only,
    )
    return


if __name__ == "__main__":
    fire.Fire(main)
