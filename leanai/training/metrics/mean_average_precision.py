"""doc
# leanai.training.metrics.mean_average_precision

> An implementation of the AP metric.
"""
def compute_PR_curve(tp_fp_list, num_gts):
    """
    Given a list of true positive and false positives the precision-recall curve is computed.

    :param tp_fp_list: A list having a True for TP and a False for FP. It is assumed, that the list is sorted by confidence.
    :param num_gts: The amount of total ground truth boxes (equivalent to TP+FN).
    """
    num_TP = 0
    PR_curve = {0.0: 1.0}
    for idx, isTP in enumerate(tp_fp_list):
        # Only in case of TP does recall increase
        # As additional FP only decrease precision, we get our max value here
        if isTP:
            num_TP += 1
            precision = num_TP / (idx + 1)
            # Update old recalls if current precision is better
            for recall in PR_curve.keys():
                if PR_curve[recall] < precision:
                    PR_curve[recall] = precision
            # Add for current recall value
            recall = num_TP / num_gts
            PR_curve[recall] = precision
    return PR_curve


def compute_AP(PR_curve, N=11):
    """
    Compute average precision, given a mapping from recall to precision defining the precision recall curve.

    :param PR_curve: A dictionary mapping recall values to precision. (It is assumed, that keys are sorted ascending.)
    :param N: Number of approximation points, typically 11 is used.
    """
    AP = 0
    AP_r = 0.0
    for recall, precision in PR_curve.items():
        while AP_r <= recall:
            AP += precision * 1.0/N
            AP_r += 1.0/N
    return AP
