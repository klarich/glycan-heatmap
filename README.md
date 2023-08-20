# glycan-heatmap

This repository contains code for making a glycan clustermap using publicly available glycan array data, compiled by
[glycowork](https://bojarlab.github.io/glycowork/glycan_data.html). This builds upon the heatmap functionality in
glycowork, enabling users to create heatmaps based on terminal motifs, know motifs, or all mono- and di-saccharides in
the dataset. The cluster map clusters lectins based on their glycan binding profile, enabling users to identify lectins
which bind similar glycans, and therefore may have similar functionality.

### Motivations for creating this repository

The glycowork python package enables user to make heatmaps using raw glycan array data with all mono- and
di-saccharides, or a set of known motifs (created by glycowork). The motifvations for creating this repository are:

1. To enable the creation of heatmaps using terminal motifs of a specified size. Being able to create a heatmap with
   terminal motifs of _n_ monosaccharides is important since lectins often recognize terminal (rather than internal)
   motifs.
2. To enable the application of a power transformation to normalize the data before creating the heatmaps. The values of
   the heatmap are created by taking the mean of all glycans containing a given motif on the array. However, glycan
   array data is lognormally distruted, so taking the mean of the raw data means that the values are likely to be skewed
   by outliers. Therefore, this code applies a
   yeo-johnson [power transformation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
   to each array before the heatmap is created to normalize the data.

### How to run this code

Create a new conda environment and install the requirements into the environment:

1. `conda create -c conda-forge -n <conda-env-name> python=3.10`
2. `conda activate <conda-env-name>`
3. `conda install --file conda_reqs.txt`
4. `pip install -r pip_reqs.txt`

Then run the code (see arguments below for more detailed information):

`python run_heatmap.py --output_png_filepath heatmap.png`

### Optional command line arguments for run_heatmap.py

`output_png_filepath`: Optional path to output the heatmap png to. If not provided, the heatmap will be displayed with
plt.show().

`proteins_csv_filepath`: Optional path to csv file containing proteins to filter the dataframe by; this file can be used
to customize the proteins which are shown in the heatmap. The file must contain a column named `protein` and the protein
names must match those in the `proetin` column of the glycowork glycan_binding dataset. If not provided, a random set of
proteins will be selected (the number will be determined by the `num_proteins` argument).

`feature_set`: Which feature set to use for making the heatmap; options are: 'terminal' for terminal_motifs, 'known' (
hand-crafted glycan features from glycowork), 'exhaustive' (all mono- and disaccharide features). Default is 'terminal'.

`max_size`: Maximum size (number of monosaccharides) of the motifs. Default is 2.

`num_proteins`: If not specifying proteins, the number of proteins to randomly select from the glycan_binding
dataframe. Default is 20.


