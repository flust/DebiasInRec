import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) 
+ os.path.sep + os.path.pardir + os.path.sep + "lib")

import Metrics
import numpy


def load_ratings(filename):
    try:
        raw_matrix = numpy.loadtxt(filename)
        return numpy.ma.array(raw_matrix, dtype=numpy.int, copy=False,
                              mask=raw_matrix <= 0, fill_value=0, hard_mask=True)
    except:
        print "Error: Could not load rating file '%s'" % filename
        exit()


def load_completed(filename):
    try:
        return numpy.loadtxt(filename)
    except:
        print "Error: Could not load rating file '%s'" % filename
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Propensity-scored Matrix Factorization Evaluation.')
    parser.add_argument("--test", "-t", type=str,
                        help="test ratings (uniformly sampled) in ASCII format", required=True)
    parser.add_argument("--completed", "-c", type=str,
                        help="filename for completed matrix", required=True)

    args = parser.parse_args()

    completed = load_completed(args.completed)
    test = load_ratings(args.test)

    for metric in [Metrics.MSE, Metrics.MAE]:
        metricValue = metric(test, completed, None)[0]
        print metric.__name__ + ": " + str(metricValue)
