import numpy
import scipy.optimize
import sys
import Metrics


def PREDICTED_SCORES(user_vectors, item_vectors, user_biases, item_biases, global_bias, use_bias = True):
    rawScores = numpy.dot(user_vectors, item_vectors.T)
    if use_bias:
        biasedScores = rawScores + user_biases[:,None] + item_biases[None,:] + global_bias
        return biasedScores
    else:
        return rawScores

    
def GENERATE_MATRIX(observed_ratings, inverse_propensities, l2_regularization, num_dimensions, normalization,
        bias_mode = 'Regularized', mode = 'MSE', start_vec = None, verbose = False):

    metricMode = None
    if mode == 'MSE':
        metricMode = 1
    elif mode == 'MAE':
        metricMode = 2
    else:
        print "MF.GENERATE_MATRIX: [ERR]\t Metric not supported:", mode
        sys.exit(0)

    inversePropensities = Metrics.SET_PROPENSITIES(observed_ratings, inverse_propensities, False)

    numUsers, numItems = numpy.shape(observed_ratings)
    scale = numUsers * numItems
    numObservations = numpy.ma.count(observed_ratings)

    perUserNormalizer = numpy.ma.sum(inversePropensities, axis = 1, dtype = numpy.longdouble)
    perUserNormalizer = numpy.ma.masked_less_equal(perUserNormalizer, 0.0, copy = False)

    perItemNormalizer = numpy.ma.sum(inversePropensities, axis = 0, dtype = numpy.longdouble)
    perItemNormalizer = numpy.ma.masked_less_equal(perItemNormalizer, 0.0, copy = False)

    globalNormalizer = numpy.ma.sum(inversePropensities, dtype = numpy.longdouble)

    normalizedPropensities = None
    if normalization == 'Vanilla':
        normalizedPropensities = inversePropensities
    elif normalization == 'SelfNormalized':
        normalizedPropensities = scale * numpy.ma.divide(inversePropensities, globalNormalizer)
    elif normalization == 'UserNormalized':
        normalizedPropensities = numItems * numpy.ma.divide(inversePropensities, perUserNormalizer[:, None])
    elif normalization == 'ItemNormalized':
        normalizedPropensities = numUsers * numpy.ma.divide(inversePropensities, perItemNormalizer[None, :])
    else:
        print "MF.GENERATE_MATRIX: [ERR]\t Normalization not supported:", normalization
        sys.exit(0)
    
    useBias = None
    regularizeBias = None
    if bias_mode == 'None':
        useBias = False
        regularizeBias = False
    elif bias_mode == 'Regularized':
        useBias = True
        regularizeBias = True
    elif bias_mode == 'Free':
        useBias = True
        regularizeBias = False
    else:
        print "MF.GENERATE_MATRIX: [ERR]\t Bias mode not supported:", bias_mode
        sys.exit(0)

    if verbose:
        print "MF.GENERATE_MATRIX: [LOG]\t Lamda:", l2_regularization, "\t NumDims:", num_dimensions,\
            "\t Normalization:", normalization, "\t Metric:", mode, "\t BiasMode:", bias_mode

    normalizedPropensities = numpy.ma.filled(normalizedPropensities, 0.0)
    observedRatings = numpy.ma.filled(observed_ratings, 0)
    
    def Mat2Vec(user_vectors, item_vectors, user_biases, item_biases, global_bias):
        allUserParams = numpy.concatenate((user_vectors, user_biases[:,None]), axis = 1)
        allItemParams = numpy.concatenate((item_vectors, item_biases[:,None]), axis = 1)
        
        allParams = numpy.concatenate((allUserParams, allItemParams), axis = 0)
        paramVector = numpy.reshape(allParams, (numUsers + numItems)*(num_dimensions + 1))
        paramVector = numpy.concatenate((paramVector, [global_bias]))
        return paramVector.astype(numpy.float)
        
    def Vec2Mat(paramVector):
        globalBias = paramVector[-1]
        remainingParams = paramVector[:-1]
        allParams = numpy.reshape(remainingParams, (numUsers + numItems, num_dimensions + 1))
        allUserParams = allParams[0:numUsers,:]
        allItemParams = allParams[numUsers:, :]
        
        userVectors = (allUserParams[:,0:-1]).astype(numpy.longdouble)
        userBiases = (allUserParams[:,-1]).astype(numpy.longdouble)
        
        itemVectors = (allItemParams[:,0:-1]).astype(numpy.longdouble)
        itemBiases = (allItemParams[:,-1]).astype(numpy.longdouble)
        return userVectors, itemVectors, userBiases, itemBiases, globalBias
    
    def Objective(paramVector):
        userVectors, itemVectors, userBiases, itemBiases, globalBias = Vec2Mat(paramVector)
        biasedScores = PREDICTED_SCORES(userVectors, itemVectors, userBiases, itemBiases, globalBias, useBias)

        delta = numpy.subtract(biasedScores, observedRatings)
        loss = None
        if metricMode == 1:
            loss = numpy.square(delta)
        elif metricMode == 2:
            loss = numpy.abs(delta)
        else:
            sys.exit(0)

        weightedLoss = numpy.multiply(loss, normalizedPropensities)
        objective = numpy.sum(weightedLoss, dtype = numpy.longdouble)

        gradientMultiplier = None
        if metricMode == 1:
            gradientMultiplier = numpy.multiply(normalizedPropensities, 2 * delta)
        elif metricMode == 2:
            gradientMultiplier = numpy.zeros(numpy.shape(delta), dtype = numpy.int)
            gradientMultiplier[delta > 0] = 1
            gradientMultiplier[delta < 0] = -1
            gradientMultiplier = numpy.multiply(normalizedPropensities, gradientMultiplier)
        else:
            sys.exit(0)

        userVGradient = numpy.dot(gradientMultiplier, itemVectors)
        itemVGradient = numpy.dot(gradientMultiplier.T, userVectors)

        userBGradient = None
        itemBGradient = None
        globalBGradient = None
        if useBias:
            userBGradient = numpy.sum(gradientMultiplier, axis = 1, dtype = numpy.longdouble)
            itemBGradient = numpy.sum(gradientMultiplier, axis = 0, dtype = numpy.longdouble)
            globalBGradient = numpy.sum(gradientMultiplier, dtype = numpy.longdouble)
        else:
            userBGradient = numpy.zeros(numpy.shape(userBiases), dtype = numpy.longdouble)
            itemBGradient = numpy.zeros(numpy.shape(itemBiases), dtype = numpy.longdouble)
            globalBGradient = 0.0

        if l2_regularization > 0:
            scaledPenalty = 1.0 * l2_regularization * scale / (numUsers + numItems)
            if regularizeBias:
                scaledPenalty /= (num_dimensions + 1)
            else:
                scaledPenalty /= num_dimensions

            userVGradient += 2 * scaledPenalty * userVectors
            itemVGradient += 2 * scaledPenalty * itemVectors
          
            objective += scaledPenalty * numpy.sum(numpy.square(userVectors), dtype = numpy.longdouble)
            objective += scaledPenalty * numpy.sum(numpy.square(itemVectors), dtype = numpy.longdouble)
 
            if regularizeBias:
                userBGradient += 2 * scaledPenalty * userBiases
                itemBGradient += 2 * scaledPenalty * itemBiases
                globalBGradient += 2 * scaledPenalty * globalBias
                objective += scaledPenalty * numpy.sum(numpy.square(userBiases), dtype = numpy.longdouble)
                objective += scaledPenalty * numpy.sum(numpy.square(itemBiases), dtype = numpy.longdouble)
                objective += scaledPenalty * globalBias * globalBias
            
        gradient = Mat2Vec(userVGradient, itemVGradient, userBGradient, itemBGradient, globalBGradient)

        if verbose:
            print ".",
            sys.stdout.flush()
        
        return objective, gradient
    
    def ObjectiveOnly(paramVector):
        objective, gradient = Objective(paramVector)
        return objective
    def GradientOnly(paramVector):
        objective, gradient = Objective(paramVector)
        return gradient
    
    userVectorsInit = None
    itemVectorsInit = None
    userBiasesInit = None
    itemBiasesInit = None
    globalBiasInit = None
    if start_vec is None:
        userVectorsInit = numpy.random.standard_normal((numUsers, num_dimensions))
        itemVectorsInit = numpy.random.standard_normal((numItems, num_dimensions))
        userBiasesInit = numpy.zeros(numUsers, dtype = numpy.float)
        itemBiasesInit = numpy.zeros(numItems, dtype = numpy.float)
        globalBiasInit = 0
    else:
        userVectorsInit = start_vec[0]
        itemVectorsInit = start_vec[1]
        userBiasesInit = start_vec[2]
        itemBiasesInit = start_vec[3]
        globalBiasInit = start_vec[4]
    
    startVector = Mat2Vec(userVectorsInit, itemVectorsInit, userBiasesInit, itemBiasesInit, globalBiasInit)

    if verbose:
        print "MF.GENERATE_MATRIX: [DBG]\t Checking gradients"
        print scipy.optimize.check_grad(ObjectiveOnly, GradientOnly, startVector)

    ops = {'maxiter': 2000, 'disp': False, 'gtol': 1e-5,\
            'ftol': 1e-5, 'maxcor': 50}

    result = scipy.optimize.minimize(fun = Objective, x0 = startVector,
                    method = 'L-BFGS-B', jac = True, tol = 1e-5, options = ops)
    
    if verbose:
        print ""
        print "MF.GENERATE_MATRIX: [DBG]\t Optimization result:", result['message']
        sys.stdout.flush()

    return Vec2Mat(result['x'])
    
    
if __name__ == "__main__":
    import scipy.sparse
    
    rows = [2,1,4,3,0,4,3]
    cols = [0,2,1,1,0,0,0]
    vals = [1,2,3,4,5,4,5]
    checkY = scipy.sparse.coo_matrix((vals, (rows,cols)), dtype = numpy.int)
    checkY = checkY.toarray()
    checkY = numpy.ma.array(checkY, dtype = numpy.int, mask = checkY <= 0, hard_mask = True, copy = False)
    print "[MAIN]\t Partially observed ratings matrix"
    print checkY

    randomPropensities = numpy.random.random(size = numpy.shape(checkY))
    randomInvPropensities = numpy.reciprocal(randomPropensities)

    userVectors, itemVectors, userBiases, itemBiases, globalBias = GENERATE_MATRIX(checkY, None, 1.0, 5, 'Vanilla',
                                                'Regularized', 'MSE', None, verbose = True)

    userVectors, itemVectors, userBiases, itemBiases, globalBias = GENERATE_MATRIX(checkY, randomInvPropensities, 1.0, 5, 'Vanilla',
                                                'Regularized', 'MSE', None, verbose = True)

    userVectors, itemVectors, userBiases, itemBiases, globalBias = GENERATE_MATRIX(checkY, randomInvPropensities, 1.0, 5, 'Vanilla',
                                                'Regularized', 'MAE', None, verbose = True)

    userVectors, itemVectors, userBiases, itemBiases, globalBias = GENERATE_MATRIX(checkY, randomInvPropensities, 1.0, 5, 'SelfNormalized',
                                                'Regularized', 'MSE', None, verbose = True)

    userVectors, itemVectors, userBiases, itemBiases, globalBias = GENERATE_MATRIX(checkY, None, 1.0, 5, 'Vanilla',
                                                'Free', 'MSE', None, verbose = True)

    userVectors, itemVectors, userBiases, itemBiases, globalBias = GENERATE_MATRIX(checkY, randomInvPropensities, 1.0, 5, 'SelfNormalized',
                                                'None', 'MSE', None, verbose = True)


    print "[MAIN]\t User vectors"
    print userVectors
    print "[MAIN]\t Item vectors"
    print itemVectors
    print "[MAIN]\t User biases"
    print userBiases
    print "[MAIN]\t Item biases"
    print itemBiases
    print "[MAIN]\t Global bias"
    print globalBias
    
    completeScores = PREDICTED_SCORES(userVectors, itemVectors, userBiases, itemBiases, globalBias, True)
    print "[MAIN]\t Predicted scores"
    print completeScores
    
