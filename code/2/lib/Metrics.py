import numpy
import sys


class Jitter:
    def __init__(self, cut_off, num_users, num_items):
        self.jitter = 1e-7 * numpy.random.standard_normal((num_users, num_items))
        discountParams = 2.0 + numpy.array(range(num_items), dtype = numpy.longdouble)
        self.discountParams = numpy.reciprocal(numpy.log2(discountParams))
        self.cutOff = min(cut_off, num_items)
        self.discountParams[self.cutOff:] = 0.0

        print "Jitter.init: [DBG]\t (NumUsers, NumItems)", num_users, num_items, "\t Sum DiscountFactors",\
                self.discountParams.sum(dtype = numpy.longdouble), "\t [Requested/Set] Cut-off:", \
                cut_off, self.cutOff

    def rank(self, predicted_matrix):
        transformedPredictions = -numpy.ma.add(predicted_matrix, self.jitter)
        sortedPredictions = numpy.ma.argsort(transformedPredictions, axis = 1)
        return sortedPredictions

dcgJitter = None
       

def SET_PROPENSITIES(observed_ratings, inverse_propensities, verbose = False):
    # observed_ratings: rating 矩阵
    # inverse_prpensities: 倾向性
    numObservations = numpy.ma.count(observed_ratings) # 数量
    numUsers, numItems = numpy.shape(observed_ratings) # user, item 数量
    scale = numUsers * numItems # user * item
    inversePropensities = None 
    # 如果输入的倾向性为None，就设置为全1
    if inverse_propensities is None:
        inversePropensities = numpy.ones((numUsers, numItems), dtype = numpy.longdouble) * scale /\
                            numObservations
    else:
        inversePropensities = numpy.array(inverse_propensities, dtype = numpy.longdouble, copy = True)

    # 做mask
    inversePropensities = numpy.ma.array(inversePropensities, dtype = numpy.longdouble, copy = False, 
                            mask = numpy.ma.getmask(observed_ratings), fill_value = 0, hard_mask = True)
 
    if verbose:
        print "Metrics.SET_PROPENSITIES: [LOG]\t NumUsers, NumItems, NumObservations", \
            numUsers, numItems, numObservations
        print "Metrics.SET_PROPENSITIES: [DBG]\t Sum of observed inverse propensities ", \
            numpy.ma.sum(inversePropensities, dtype = numpy.longdouble), \
            "(=? NumUsers * NumItems)", numUsers * numItems
    # 返回 mask 后的 inversePropensity 矩阵
    return inversePropensities


def ITEMWISE_METRICS(observed_ratings, predicted_ratings, inverse_propensities, verbose, mode = 'MSE'):
    # 输入 真实值 预测值，倾向性评分
    delta = numpy.ma.subtract(predicted_ratings, observed_ratings)
    # delta: 差值
    rectifiedDelta = None
    if mode == 'MSE':
        rectifiedDelta = numpy.square(delta)
    elif mode == 'MAE':
        rectifiedDelta = numpy.ma.abs(delta)
    else:
        print "Metrics.ITEMWISE_METRICS: [ERR]\t Unrecognized itemwise metric ", mode
        sys.exit(0)

    inversePropensities = SET_PROPENSITIES(observed_ratings, inverse_propensities, verbose)

    numUsers, numItems = numpy.shape(observed_ratings)
    scale = numUsers * numItems

    # 
    observedError = numpy.ma.multiply(rectifiedDelta, inversePropensities)
    # MSE * IPS

    cumulativeError = numpy.ma.sum(observedError, dtype = numpy.longdouble)
    # totalloss = \sum{MSE * IPS}

    vanillaMetric = cumulativeError / scale
    # loss = totalloss / scale
    
    globalNormalizer = numpy.ma.sum(inversePropensities, dtype = numpy.longdouble)
    # sum(IPS)

    selfNormalizedMetric = cumulativeError / globalNormalizer
    # totalloss / sum(IPS)
    
    perUserNormalizer = numpy.ma.sum(inversePropensities, axis = 1, dtype = numpy.longdouble)
    # sum(IPS, axis = 1)

    perUserNormalizer = numpy.ma.masked_less_equal(perUserNormalizer, 0.0, copy = False)
    # mask the num that less than 0.0

    perUserError = numpy.ma.sum(observedError, axis = 1, dtype = numpy.longdouble)
    # axis = 1
    perUserEstimate = numpy.ma.divide(perUserError, perUserNormalizer)
    # only record the exist user

    userNormalizedMetric = numpy.ma.sum(perUserEstimate, dtype = numpy.longdouble) / numUsers

    perItemNormalizer = numpy.ma.sum(inversePropensities, axis = 0, dtype = numpy.longdouble)
    perItemNormalizer = numpy.ma.masked_less_equal(perItemNormalizer, 0.0, copy = False)

    perItemError = numpy.ma.sum(observedError, axis = 0, dtype = numpy.longdouble)
    perItemEstimate = numpy.ma.divide(perItemError, perItemNormalizer)
    itemNormalizedMetric = numpy.ma.sum(perItemEstimate, dtype = numpy.longdouble) / numItems
   
    if verbose:
        print "Metrics.ITEMWISE_METRICS: [LOG]\t Vanilla, SelfNormalized, UserNormalized, ItemNormalized", \
            vanillaMetric, selfNormalizedMetric, userNormalizedMetric, itemNormalizedMetric

    return vanillaMetric, selfNormalizedMetric, userNormalizedMetric, itemNormalizedMetric
    

def MSE(observed_ratings, predicted_ratings, inverse_propensities, verbose = False):
    return ITEMWISE_METRICS(observed_ratings, predicted_ratings, inverse_propensities, verbose, mode = 'MSE')


def MAE(observed_ratings, predicted_ratings, inverse_propensities, verbose = False):
    return ITEMWISE_METRICS(observed_ratings, predicted_ratings, inverse_propensities, verbose, mode = 'MAE')

    
def DCG(observed_ratings, predicted_ratings, inverse_propensities, cut_off = 50, verbose = False):
    global dcgJitter
    numUsers, numItems = numpy.shape(observed_ratings)
    scale = numUsers * numItems

    if dcgJitter is None or dcgJitter.cutOff != cut_off:
        dcgJitter = Jitter(cut_off, numUsers, numItems)
 
    inversePropensities = SET_PROPENSITIES(observed_ratings, inverse_propensities, verbose)
    
    predictedRankings = dcgJitter.rank(predicted_ratings)
    weightedGain = numpy.ma.multiply(observed_ratings, inversePropensities)
 
    perUserNormalizer = numpy.ma.sum(inversePropensities, axis = 1, dtype = numpy.longdouble)
    perUserNormalizer = numpy.ma.masked_less_equal(perUserNormalizer, 0.0, copy = False)

    staticIndices = numpy.ogrid[0:numUsers, 0:numItems]
    rankedGains = weightedGain[staticIndices[0], predictedRankings]
    perUserDCG = numpy.ma.dot(rankedGains, dcgJitter.discountParams)

    dcgValue = numpy.ma.sum(perUserDCG, dtype = numpy.longdouble) / numUsers
    snDCGValue = dcgValue * scale / numpy.ma.sum(inversePropensities, dtype = numpy.longdouble)

    perUserNormalizedEstimates = numpy.ma.divide(perUserDCG, perUserNormalizer)
    uDCGValue = numItems * numpy.ma.sum(perUserNormalizedEstimates, dtype = numpy.longdouble) / numUsers
    
    if verbose:
        print "Metrics.DCG: [LOG]\t DCG, SN-DCG, UN-DCG, IN-DCG", dcgValue, snDCGValue, uDCGValue, 0.0
    return dcgValue, snDCGValue, uDCGValue, 0.0
    
    
def CG(observed_ratings, selected_items, inverse_propensities, verbose = False):
    inversePropensities = SET_PROPENSITIES(observed_ratings, inverse_propensities, verbose)

    clippedSelections = numpy.clip(selected_items, 0, 1)
    weightedGain = numpy.ma.multiply(observed_ratings, inversePropensities)
    cumulativeGain = numpy.ma.multiply(weightedGain, clippedSelections)
    
    numUsers, numItems = numpy.shape(observed_ratings)
    scale = numUsers * numItems

    globalGain = numpy.ma.sum(cumulativeGain, dtype = numpy.longdouble)
    globalNormalizer = numpy.ma.sum(inversePropensities, dtype = numpy.longdouble)

    cg = globalGain / numUsers
    snCG = numItems * globalGain / globalNormalizer

    perUserNormalizer = numpy.ma.sum(inversePropensities, axis = 1, dtype = numpy.longdouble)
    perUserNormalizer = numpy.ma.masked_less_equal(perUserNormalizer, 0.0, copy = False)

    perUserGain = numpy.ma.sum(cumulativeGain, axis = 1, dtype = numpy.longdouble)
    perUserEstimate = numpy.ma.divide(perUserGain, perUserNormalizer)
    unCG = numItems * numpy.ma.sum(perUserEstimate, dtype = numpy.longdouble) / numUsers

    perItemNormalizer = numpy.ma.sum(inversePropensities, axis = 0, dtype = numpy.longdouble)
    perItemNormalizer = numpy.ma.masked_less_equal(perItemNormalizer, 0.0, copy = False)

    perItemGain = numpy.ma.sum(cumulativeGain, axis = 0, dtype = numpy.longdouble)
    perItemEstimate = numpy.ma.divide(perItemGain, perItemNormalizer)
    inCG = numpy.ma.sum(perItemEstimate, dtype = numpy.longdouble)
       
    if verbose:
        print "Metrics.CG: [LOG]\t CG, SN-CG, UN-CG, IN-CG", cg, snCG, unCG, inCG
    return cg, snCG, unCG, inCG

    
if __name__ == "__main__":
    shape = (5,3)
    a = numpy.random.randint(0,5, size=shape)
    b = numpy.random.randint(0,5, size=shape)
    
    print "[MAIN]\t True ratings:"
    print a
    print "[MAIN]\t Predicted ratings:"
    print b
    
    inversePropensities = numpy.random.random(shape)
    print "[MAIN]\t Propensities:"
    print inversePropensities
    obs = numpy.random.random(shape)
    obs = obs < inversePropensities
    inversePropensities = numpy.reciprocal(inversePropensities)
    print "[MAIN]\t Inverse Propensities:"
    print inversePropensities

    print "[MAIN]\t Observations:"
    print obs
    
    observed_a = numpy.ma.array(a, dtype = numpy.longdouble, copy = True, 
                            mask = numpy.logical_not(obs), fill_value = 0, hard_mask = True)
     
    print "[MAIN]\t MSE: Vanilla, SN, UN, IN:",
    MSE(observed_a, b, inversePropensities, verbose = True)
    print "[MAIN]\t MAE: Vanilla, SN, UN, IN:"
    MAE(observed_a, b, inversePropensities, verbose = True)
    print "[MAIN]\t DCG: Vanilla, SN, UN, IN:"
    DCG(observed_a, b, inversePropensities, cut_off = 50, verbose = True)
    
    print "[MAIN]\t CG: Vanilla, SN, UN, IN:"
    CG(observed_a, b, inversePropensities, verbose = True)
    
