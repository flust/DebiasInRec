import argparse
import itertools
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) 
+ os.path.sep + os.path.pardir + os.path.sep + "lib")

import Metrics
import Expt3
import numpy

class Files(object):
    def __init__(self, train, propensities):
        self.train = train
        self.propensities = propensities


class Logger(object):
    def __init__(self, verbosity_level=1):
        self._verbosity_level = verbosity_level

    def log(self, message, level=1):
        if level <= self._verbosity_level:
            print message
            sys.stdout.flush()

def load_ratings(filename):
    try:
        raw_matrix = numpy.loadtxt(filename)
        return numpy.ma.array(raw_matrix, dtype=numpy.int, copy=False,
                              mask=raw_matrix <= 0, fill_value=0, hard_mask=True)
    except:
        print "Error: Could not load rating file '%s'" % filename
        exit()


def load_propensities(filename):
    try:
        return numpy.loadtxt(filename)
    except:
        print "Error: Could not load propensities."
        exit()


def check_writeable(filename):
    try:
        with open(filename, "wb") as f:
            pass
        os.remove(filename)
        return True, ""
    except IOError:
        print "Error: Could not open file '%s' for writing" % filename
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Propensity-scored Matrix Factorization.')
    parser.add_argument("--ratings", "-r", type=str, required=True,
                        help="ratings matrix in ASCII format")
    parser.add_argument("--propensities", "-p", type=str, default="",
                        help="propensities matrix in ASCII format (optional)")
    parser.add_argument("--completed", "-c", type=str, default="completed_ratings.ascii",
                        help="filename for completed matrix")
    parser.add_argument('--metric', '-m', metavar='M', type=str, choices=["MSE", "MAE"],
                        help='Metric to be optimized', default='MSE')
    parser.add_argument('--lambdas', '-l', metavar='L', type=str,
                        help='Lambda values', default='0.008,0.04,0.2,1,5,25,125')
    parser.add_argument('--numdims', '-n', metavar='N', type=str,
                        help='Dimension values', default='5,10,20,40')
    parser.add_argument('--seed', '-s', metavar='S', type=int,
                        help='Seed for numpy.random', default=387)
    parser.add_argument("--verbosity", "-v", type=int, choices=[0, 1, 2],
                        help="output verbosity (default = 2)", default=2)
    
    args = parser.parse_args()
    check_writeable(args.completed)
    my_logger = Logger(args.verbosity)
    
    lambdas = []
    tokens = args.lambdas.strip().split(',')
    for token in tokens:
        lambdas.append(float(token))

    numDims = []
    tokens = args.numdims.strip().split(',')
    for token in tokens:
        numDims.append(int(token))

    train = load_ratings(args.ratings)
    if args.propensities:
        propensities = load_propensities(args.propensities)
        propensities_desc = "IPS using " + args.propensities
    else:
        propensities = None
        propensities_desc = "naive (uniform)"
    data = Files(train, propensities)

    if args.metric == 'MSE':
        metric = Metrics.MSE
    elif args.metric == 'MAE':
        metric = Metrics.MAE
    
    Expt3.learn(data, my_logger, lambdas=lambdas, numDims=numDims, metric=metric, approach="IPS",
          seed=args.seed, raw_metric=args.metric, output_name=args.completed, propensities_desc=propensities_desc)
