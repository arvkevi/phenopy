import fire
import itertools
import sys
import os

from configparser import NoOptionError, NoSectionError
from multiprocessing import Pool
import pandas as pd

from phenopy import open_or_stdout, generate_annotated_hpo_network
from phenopy.config import config, logger, project_data_dir
from phenopy.score import Scorer
from phenopy.util import parse_input, half_product
from phenopy.cluster import process_kfile, prep_cluster_data, prep_feature_array, apply_umap, dbscan
from phenopy.plot import plot_basic_dbscan



def score(input_file, output_file='-', records_file=None, annotations_file=None, custom_disease_file=None, ages_distribution_file=None,
          self=False, summarization_method='BMWA', scoring_method='HRSS', threads=1):
    """
    Scores similarity of provided HPO annotated entries (see format below) against a set of HPO annotated dataset. By
    default scoring happens against diseases annotated by the HPO group. See https://hpo.jax.org/app/download/annotation.

    Phenopy also supports scoring the product of provided entries (see "--product") or scoring against a custom records
    dataset (see "--records-file).

    :param input_file: File with HPO annotated entries, one per line (see format below).
    :param output_file: File path where to store the results. [default: - (stdout)]
    :param records_file: An entity-to-phenotype annotation file in the same format as "input_file". This file, if
     provided, is used to score entries in the "input_file" against entries here. [default: None]
    :param annotations_file: An entity-to-phenotype annotation file in the same format as "input_file". This file, if
     provided, is used to add information content to the network. [default: None]
    :param custom_disease_file: entity Annotation for ranking diseases/genes
    :param ages_distribution_file: Phenotypes age summary stats file containing phenotype HPO id, mean_age, and std.
     [default: None]
    :param self: Score entries in the "input_file" against itself.
    :param summarization_method: The method used to summarize the HRSS matrix. Supported Values are best match average
    (BMA), best match weighted average (BMWA), and maximum (maximum). [default: BMWA]
    :param scoring_method: Either HRSS or Resnik
    :param threads: Number of parallel processes to use. [default: 1]
    """

    try:
        obo_file = config.get('hpo', 'obo_file')
    except (NoSectionError, NoOptionError):
        logger.critical(
            'No HPO OBO file found in the configuration file. See "hpo:obo_file" parameter.')
        exit(1)
    if custom_disease_file is None:
        try:
            disease_to_phenotype_file = config.get('hpo', 'disease_to_phenotype_file')
        except (NoSectionError, NoOptionError):
            logger.critical(
                'No HPO annotated dataset file found in the configuration file.'
                ' See "hpo:disease_to_phenotype_file" parameter.'
            )
            exit(1)
    else:
        logger.info(f"using custom disease annotation file: {custom_disease_file}")
        disease_to_phenotype_file = custom_disease_file

    logger.info(f'Loading HPO OBO file: {obo_file}')
    hpo_network, alt2prim, disease_records = \
        generate_annotated_hpo_network(obo_file,
                                       disease_to_phenotype_file,
                                       annotations_file=annotations_file,
                                       ages_distribution_file=ages_distribution_file
                                       )

    # parse input records
    input_records = parse_input(input_file, hpo_network, alt2prim)

    # create instance the scorer class
    try:
        scorer = Scorer(hpo_network, summarization_method=summarization_method,
                        scoring_method=scoring_method)
    except ValueError as e:
        logger.critical(f'Failed to initialize scoring class: {e}')
        sys.exit(1)

    if self:
        score_records = input_records

        scoring_pairs = list(half_product(len(score_records), len(score_records)))
    else:
        if records_file:
            score_records = parse_input(records_file, hpo_network, alt2prim)
        else:
            score_records = disease_records

        scoring_pairs = itertools.product(
            range(len(input_records)),
            range(len(score_records)),
        )

    # launch as many scoring process as requested
    with Pool(threads) as p:
        results = p.starmap(
            scorer.score_records,
            [
                (
                    input_records,  # a records
                    score_records,  # b records
                    scoring_pairs,  # pairs
                    i,  # thread_index
                    threads,  # threads
                ) for i in range(threads)
            ]
        )

    with open_or_stdout(output_file) as output_fh:
        output_fh.write('\t'.join(['#query', 'entity_id', 'score']))
        output_fh.write('\n')
        for r in results:
            for s in r:
                output_fh.write('\t'.join(s))
                output_fh.write('\n')


def cluster(input_file, kfile=None, k=1000, n_neighbors=30, n_components=2, min_dist=0.01, metric='euclidean', eps=0.40, min_samples=10):
    """
    :param input_file: file containing phenotypes encoded in HPO ids
    :param kfile: phenotype to group id conversion (optional)
    :param k: number of phenotype features
    :param n_neighbors: UMAP num neightbors
    :param n_components: UMAP num components
    :param min_dist: UMAP min distances
    :param metric: UMAP metric
    :param eps: DBSCAN eps
    :param min_samples: DBSCAN min_samples
    :return: None
    """

    if kfile is None:
            kfile = os.path.join(project_data_dir, "phenotype_groups.txt")

    feature_to_hps, hp_to_feature, n_features = process_kfile(kfile, k=k)
    records = parse_input(input_file)
    results_df = prep_cluster_data(pd.DataFrame.from_dict(records), hp_to_feature)
    logger.info(f"Loading: {input_file}")
    X_vect = prep_feature_array(results_df, n_features)
    logger.info("Performing UMAP dimensionality reduction")

    umap_result = apply_umap(X_vect,
                             n_neighbors=n_neighbors,
                             n_components=n_components,
                             min_dist=min_dist,
                             metric=metric)

    logger.info("Clustering using DBSCAN")

    labels, core_samples_mask, stats = dbscan(umap_result,
                                              eps=eps,
                                              min_samples=min_samples)

    logger.info(f"Num. Clusters: {stats['n_clusters']}")
    logger.info(f"Num. noise points: {stats['n_noise']}")
    logger.info(f"Silhouette_score: {stats['silhouette_score']}")

    output_df = results_df.copy()[['id']]
    output_df['cluster_id'] = labels
    dbscan_plot = plot_basic_dbscan(umap_result, core_samples_mask, labels)

    input_file_name, extension = os.path.splitext(input_file)
    output_data_name = input_file_name + ".clusters.tsv"
    output_plot_name = input_file_name + ".clusters.png"
    logger.info(f"Writing results to {output_data_name}")
    output_df.to_csv(output_data_name, index=None, header=None, sep="\t")
    logger.info(f"Saving plot {output_plot_name}")
    dbscan_plot.savefig(output_plot_name)


def main():
    fire.Fire({
        'score': score,
        'cluster': cluster
    })


if __name__ == '__main__':
    main()
