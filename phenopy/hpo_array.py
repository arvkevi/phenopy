import os

import numpy as np

from multiprocessing import Pool
from multiprocessing.sharedctypes import RawArray

from phenopy import generate_annotated_hpo_network
from phenopy.score import Scorer
from phenopy.util import half_product


def generate_hpo_array(threads):
    """
    Generates the numpy array of pairwise HPO HRSS scores.
    :param threads: Number of threads used to calculate the pairwise array.
    :return: A numpy array of pairwise HRSS scores for every HPO term and the mapping dictionary.
    """

    # data directory
    phenopy_data_directory = os.path.join(os.getenv('HOME'), '.phenopy/data')

    # files used in building the annotated HPO network
    obo_file = os.path.join(phenopy_data_directory, 'hp.obo')
    disease_to_phenotype_file = os.path.join(phenopy_data_directory, 'phenotype.hpoa')

    # if you have a custom ages_distribution_file, you can set it here.
    ages_distribution_file = os.path.join(phenopy_data_directory, 'xa_age_stats_oct052019.tsv')

    hpo, alt2prim, disease_records = \
        generate_annotated_hpo_network(obo_file,
                                       disease_to_phenotype_file,
                                       ages_distribution_file=ages_distribution_file
                                       )

    scorer = Scorer(hpo)

    def score_hpo_pair_hrss_wrapper(terms):
        term_a, term_b = int2hpo[terms[0]], int2hpo[terms[1]]
        tmp = np.ctypeslib.as_array(term_scores)
        tmp[terms[0], terms[1]] = scorer.score_hpo_pair_hrss(term_a, term_b)

    n_hpos = len(hpo.nodes())

    int2hpo = {i: hpoid for i, hpoid in enumerate(hpo.nodes())}
    c_arr = np.ctypeslib.as_ctypes(np.zeros((n_hpos, n_hpos)))
    term_scores = RawArray(c_arr._type_, c_arr)

    with Pool(threads) as p:
        p.map(score_hpo_pair_hrss_wrapper, half_product(n_hpos, n_hpos))

    # convert to numpy
    scores_arr = np.ctypeslib.as_array(term_scores)
    # copy scores from the upper half to the emtpy lower half
    lower = np.tril_indices(n_hpos, -1)
    scores_arr[lower] = scores_arr.T[lower]

    return scores_arr, int2hpo