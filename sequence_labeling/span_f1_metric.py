'''
Original code from the 2019 paper: Fine-Grained Analysis of Propaganda in News Articles
https://propaganda.qcri.org/fine-grained-propaganda-emnlp.html

Licensed under GPL license.

Computing of the adapted version F1 score for matching possibly overlapping spans with different labels.
'''

import logging
import sys
from typing import Tuple

logger = None
def init_logger():
    """
    Initialize the logger for the span F1 scorer.
    """

    global logger
    if logger is None:
        logger = logging.getLogger("span_f1_scorer")
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(ch)

def compute_score_pr(submission_annotations: dict, gold_annotations: dict, technique_names: list, 
                     prop_vs_non_propaganda: bool = False, per_article_evaluation: bool = False, 
                     disable_logger: bool = True) -> dict:
    """
    Compute precision, recall, and F1 score for span annotations.

    Args:
        submission_annotations (dict): Submitted annotations.
        gold_annotations (dict): Gold standard annotations.
        technique_names (list): List of technique names.
        prop_vs_non_propaganda (bool, optional): Flag for considering propaganda vs non-propaganda. Default is False.
        per_article_evaluation (bool, optional): Flag for per article evaluation. Default is False.
        disable_logger (bool, optional): Flag to disable logger. Default is True.

    Returns:
        dict: A dictionary containing precision, recall, and F1 score for each technique.
    """

    init_logger()
    if disable_logger: logger.setLevel(logging.CRITICAL)
    prec_denominator = sum([len(annotations) for annotations in submission_annotations.values()])
    rec_denominator = sum([len(annotations) for annotations in gold_annotations.values()])
    technique_Spr_prec = {propaganda_technique: 0 for propaganda_technique in technique_names}
    technique_Spr_rec = {propaganda_technique: 0 for propaganda_technique in technique_names}
    cumulative_Spr_prec, cumulative_Spr_rec = (0, 0)
    f1_articles = []
    result = {}
    for article_id in submission_annotations.keys():
        gold_data = gold_annotations[article_id]
        logger.debug("Computing contribution to the score of article id %s\nand tuples %s\n%s\n"
                     % (article_id, str(submission_annotations[article_id]), str(gold_data)))

        article_cumulative_Spr_prec, article_cumulative_Spr_rec = (0, 0)
        for j, sd in enumerate(submission_annotations[article_id]): #submission annotations for article article_id:
            s=""
            sd_annotation_length = len(sd[1])
            for i, gd in enumerate(gold_data):
                if prop_vs_non_propaganda or gd[0]==sd[0]:
                    #s += "\tmatch %s %s-%s - %s %s-%s"%(sd[0],sd[1], sd[2], gd[0], gd[1], gd[2])
                    intersection = len(sd[1].intersection(gd[1]))
                    gd_annotation_length = len(gd[1])
                    Spr_prec = intersection/sd_annotation_length
                    article_cumulative_Spr_prec += Spr_prec
                    cumulative_Spr_prec += Spr_prec
                    s += "\tmatch %s %s-%s - %s %s-%s: S(p,r)=|intersect(r, p)|/|p| = %d/%d = %f (cumulative S(p,r)=%f)\n"\
                         %(sd[0],min(sd[1]), max(sd[1]), gd[0], min(gd[1]), max(gd[1]), intersection, sd_annotation_length, Spr_prec, cumulative_Spr_prec)
                    technique_Spr_prec[gd[0]] += Spr_prec

                    Spr_rec = intersection/gd_annotation_length
                    article_cumulative_Spr_rec += Spr_rec
                    cumulative_Spr_rec += Spr_rec
                    s += "\tmatch %s %s-%s - %s %s-%s: S(p,r)=|intersect(r, p)|/|r| = %d/%d = %f (cumulative S(p,r)=%f)\n"\
                         %(sd[0],min(sd[1]), max(sd[1]), gd[0], min(gd[1]), max(gd[1]), intersection, gd_annotation_length, Spr_rec, cumulative_Spr_rec)
                    technique_Spr_rec[gd[0]] += Spr_rec
            logger.debug("\n%s"%(s))

        p_article, r_article, f1_article =compute_prec_rec_f1(article_cumulative_Spr_prec,
                                                              len(submission_annotations[article_id]),
                                                              article_cumulative_Spr_rec,
                                                              len(gold_annotations[article_id]), False)
        f1_articles.append(f1_article)

    p,r,f1 = compute_prec_rec_f1(cumulative_Spr_prec, prec_denominator, cumulative_Spr_rec, rec_denominator)
    result['P'] = p; result['R'] = r; result['F1'] = f1
    if not prop_vs_non_propaganda:
        for technique_name in technique_Spr_prec.keys():
            prec_tech, rec_tech, f1_tech = compute_prec_rec_f1(technique_Spr_prec[technique_name],
                                        compute_technique_frequency(submission_annotations.values(), technique_name),
                                        technique_Spr_prec[technique_name],
                                        compute_technique_frequency(gold_annotations.values(), technique_name), False)
            logger.info("%s: P=%f R=%f F1=%f" % (technique_name, prec_tech, rec_tech, f1_tech))
            result[f'{technique_name}-F1'] = f1_tech
            result[f'{technique_name}-P'] = prec_tech
            result[f'{technique_name}-R'] = rec_tech
    if per_article_evaluation:
        logger.info("Per article evaluation F1=%s"%(",".join([ str(f1_value) for f1_value in  f1_articles])))

    return result


def compute_prec_rec_f1(prec_numerator: float, prec_denominator: int, rec_numerator: float, 
                        rec_denominator: int, print_results: bool = True) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        prec_numerator (float): Numerator for precision.
        prec_denominator (int): Denominator for precision.
        rec_numerator (float): Numerator for recall.
        rec_denominator (int): Denominator for recall.
        print_results (bool, optional): Flag to print results. Default is True.

    Returns:
        Tuple[float, float, float]: Precision, recall, and F1 score.
    """

    logger.debug("P=%f/%d, R=%f/%d"%(prec_numerator, prec_denominator, rec_numerator, rec_denominator))
    p, r, f1 = (0, 0, 0)
    if prec_denominator > 0:
        p = prec_numerator / prec_denominator
    if rec_denominator > 0:
        r = rec_numerator / rec_denominator
    if print_results: logger.info("Precision=%f/%d=%f\tRecall=%f/%d=%f" % (prec_numerator, prec_denominator, p,
                                                                           rec_numerator, rec_denominator, r))
    if prec_denominator == 0 and rec_denominator == 0:
        f1 = 1.0
    if p > 0 and r > 0:
        f1 = 2 * (p * r / (p + r))
    if print_results:
        logger.info("F1=%f" % (f1))
    return p,r,f1

def compute_technique_frequency(annotations_list: list, technique_name: str) -> int:
    """
    Compute the frequency of a specific technique in the annotations.

    Args:
        annotations_list (list): List of annotations.
        technique_name (str): Name of the technique.

    Returns:
        int: Frequency of the technique.
    """

    return sum([ len([ example_annotation for example_annotation in x if example_annotation[0]==technique_name])
                 for x in annotations_list ])
