from functools import reduce
from typing import Optional, List
import os

import fire
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

from plotting.heatmap import make_heatmap
from utils.annotate import get_terminal_motifs_dataframe
from utils.logger import get_logger


LOGGER = get_logger(__name__)


def train_power_transformer(col: pd.Series) -> pd.DataFrame:
    """
    Trains the power transformer which converts the data into a normal space with unit variance
    :param col: data to transform
    :return: transformed df
    """
    col_no_na = col.dropna()
    y_train = np.array(col_no_na)
    y_train = np.array(y_train).reshape(-1, 1)
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    pt.fit(y_train)
    y_transform = pt.transform(y_train)
    new_col_name = f"{col.name}_transformed"
    transform_df = pd.DataFrame({col.name: col_no_na, new_col_name: y_transform.squeeze()})
    transform_df.reset_index(inplace=True)
    transform_df.drop(columns=[col.name], inplace=True)
    transform_df.rename(columns={"index": "glycan", new_col_name: col.name}, inplace=True)
    return transform_df


def generate_heatmap(
    output_png_filepath: str,
    output_csv_filepath: Optional[str] = None,
    glycan_binding_df_filepath: Optional[str] = None,
    blast_df_filepath: Optional[str] = None,
    feature_set: str | List[str] = "known",
    max_size: int = 2,
    terminal_motifs_only: bool = True,
    run_power_transformation: bool = True,
):
    """
    Function to generate a glycan binding heatmap
    :param output_png_filepath: Filepath to save the heatmap to
    :param output_csv_filepath: Filepath to save the csv data used to generate the heatmap
    :param glycan_binding_df_filepath: Filepath to the glycan binding dataset
    :param blast_df_filepath: Filepath to the bast dataframe containing genes of interest (should match
    glycan_binding_df on target_id)
    :param max_size: maximum size of the monosaccharide to use (only applies to the terminal motifs dataframe)
    :param feature_set: either "known" or "exhaustive".
    :param terminal_motifs_only: Whether to generate and use a dataframe of terminal motifs only. Otherwise the default
    motif_list dataframe from glycowork will be used, if the feature set is "known"
    :param run_power_transformation: Whether to run a power transformation to convert glycan binding values into a
    normal space with unit variance (recommended)
    :return: None
    """
    current_directory = os.path.dirname(os.path.realpath(__file__))
    if glycan_binding_df_filepath is None:
        glycan_binding_df_filepath = os.path.join(current_directory, "data/glycan_binding_glycowork_v0.6.0.csv")
    if blast_df_filepath is None:
        blast_df_filepath = os.path.join(current_directory, "data/membraine_genes_updated.csv")
    if isinstance(feature_set, str):
        feature_set = [feature_set]
    # Read in the relevant datasets
    glycan_binding = pd.read_csv(glycan_binding_df_filepath)
    blast_df = pd.read_csv(blast_df_filepath)
    # Sometimes we might have two near-identical genes, which lead to two rows with the same match from glycowork
    # glycan_binding dataset - need to drop duplicates so that there is only one glycowork row
    blast_df = blast_df.drop_duplicates(subset="target_id")
    blast_df = blast_df[blast_df["remove"] == "no"]
    blast_df = blast_df[["Gene_updated", "target_id"]].copy()
    blast_df = blast_df.rename(columns={"Gene_updated": "Gene"})
    heatmap_df = glycan_binding.merge(blast_df, how="inner", on="target_id")
    heatmap_df = heatmap_df.set_index("target_id")
    heatmap_df = heatmap_df.drop(columns=["target", "protein", "Gene"])
    heatmap_df = heatmap_df.T
    transformed_dfs = []
    if run_power_transformation:
        for col in heatmap_df.columns:
            # Transform glycan binding data into a normal space (see README.md about why this is necessary)
            series = heatmap_df[col]
            transformed_dfs.append(train_power_transformer(series))
        transformed_dfs_merged = reduce(
            lambda left, right: pd.merge(left, right, on="glycan", how="outer"), transformed_dfs
        )
    else:
        raise NotImplementedError(f"Not implemented to run this function without power transformation")
    transformed_dfs_merged.set_index("glycan", inplace=True)
    transformed_dfs_merged = transformed_dfs_merged.T
    transformed_dfs_merged.reset_index(inplace=True)
    transformed_dfs_merged.rename(columns={"index": "target_id"}, inplace=True)
    heatmap_df_transformed = transformed_dfs_merged.merge(blast_df, how="left", on="target_id")
    heatmap_df_transformed.drop(columns="target_id", inplace=True)
    glycans = set(heatmap_df_transformed.columns)
    glycans.remove("Gene")
    LOGGER.info(f"Creating motif dataframe")
    motifs_df = get_terminal_motifs_dataframe(glycans, max_size=max_size) if terminal_motifs_only else None
    LOGGER.info(f"Making heatmap")
    make_heatmap(
        df=heatmap_df_transformed,
        mode="motif",
        index_col="Gene",
        feature_set=feature_set,
        datatype="response",
        yticklabels=True,
        xticklabels=True,
        rarity_filter=0.02,
        filepath=output_png_filepath,
        figsize=(12, 12),
        motifs=motifs_df,
        csv_filepath=output_csv_filepath,
    )
    LOGGER.info("Done generating heatmap!")
    return


if __name__ == "__main__":
    fire.Fire(generate_heatmap)
