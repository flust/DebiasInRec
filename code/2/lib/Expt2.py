import MF
import numpy
import scipy.sparse.linalg
import sys


def MF_TRAIN(params, train_observations, inv_propensities, normalization, metric, start_vector):
    retVal = None
    actualStart = None
    if start_vector is not None:
        actualStart = (start_vector[0][:,0:params[1]], start_vector[1][:,0:params[1]],
                        start_vector[2], start_vector[3], start_vector[4])

    tempInvPropensities = None
    if inv_propensities is not None:
        tempInvPropensities = (4.0 / 3.0) * inv_propensities
        if params[2] >= 0:
            tempInvPropensities = numpy.clip(tempInvPropensities, a_min = 0, a_max = params[2])

    retVal = MF.GENERATE_MATRIX(train_observations, tempInvPropensities, params[0], 
                                params[1], normalization, bias_mode = params[3], mode = metric, 
                                start_vec = actualStart, verbose = False)

    return retVal

  
def FINAL_TRAIN(approach_tuple, metric, observations, start_vector):
    invP = approach_tuple[1]
    normN = approach_tuple[2]
    bestLambda = approach_tuple[3][0]
    bestDims = approach_tuple[3][1]
    bestClip = approach_tuple[3][2]
    bestBias = approach_tuple[3][3]
    actualStart = None
    if start_vector is not None:
        actualStart = (start_vector[0][:,0:bestDims], start_vector[1][:,0:bestDims],
                        start_vector[2], start_vector[3], start_vector[4])

    tempInvP = None
    if bestClip < 0 or invP is None:
        tempInvP = invP
    else:
        tempInvP = numpy.clip(invP, a_min = 0, a_max = bestClip)

    retVal = MF.GENERATE_MATRIX(observations, tempInvP, bestLambda, bestDims, normN, bias_mode = bestBias,
                        mode = metric, start_vec = actualStart)

    return retVal
    

def INIT_PARAMS(partial_observations, num_dimensions):
    averageObservedRating = numpy.ma.mean(partial_observations, dtype = numpy.longdouble)
    completeRatings = numpy.ma.filled(partial_observations.astype(numpy.float), averageObservedRating)
    numUsers, numItems = numpy.shape(partial_observations)

    u,s,vt = scipy.sparse.linalg.svds(completeRatings, k = num_dimensions, ncv = 50, tol = 1e-7, which = 'LM', 
                        v0 = None, maxiter = 2000, return_singular_vectors = True)
            
    startTuple = (u, numpy.transpose(numpy.multiply(vt, s[:,None])), 
                     numpy.zeros(numUsers, dtype = numpy.longdouble), 
                     numpy.zeros(numItems, dtype = numpy.longdouble), 
                     averageObservedRating)
    return startTuple 


def TRAIN_HELPER(approach, gold_inv_propensities, nb_inv_propensities):
    invP = None
    if approach == 'Naive':
        invP = None
    elif approach.startswith('Gold'):
        invP = gold_inv_propensities
    elif approach.startswith('NB'):
        invP = nb_inv_propensities
    else:
        print "TRAIN_HELPER: [ERR] Unrecognized approach", approach
        sys.exit(0)

    normN = None
    if approach == 'Naive' or approach.endswith('-IPS'):
        normN = 'Vanilla'
    elif approach.endswith('-SNIPS'):
        normN = 'SelfNormalized'
    elif approach.endswith('-UNIPS'):
        normN = 'UserNormalized'
    elif approach.endswith('-INIPS'):
        normN = 'ItemNormalized'
    else:
        print "TRAIN_HELPER: [ERR] Unrecognized approach", approach
        sys.exit(0)
        
    return invP, normN    
    
    
if __name__ == "__main__":
    import argparse
    import Datasets
    import Metrics
    import Propensity
    import pickle
    import os
    import itertools
    from joblib import Parallel, delayed
    
    parser = argparse.ArgumentParser(description='Semi-Synthetic Learning on ML100K.')
    parser.add_argument('--seed', '-s', metavar='S', type=int, 
                        help='Seed for numpy.random', default=387)
    parser.add_argument('--trial', '-t', metavar='T', type=int, 
                        help='Trial ID', default=1)
    parser.add_argument('--alphas', '-a', metavar='A', type=str, 
                        help='Alpha values', default='1,0.5,0.25,0.125,0.0625,0.03125')
    parser.add_argument('--lambdas', '-l', metavar='L', type=str, 
                        help='Lambda values', default='0.008,0.04,0.2,1,5,25,125')
    parser.add_argument('--numdims', '-n', metavar='N', type=str, 
                        help='Dimension values', default='20')
    parser.add_argument('--clips', '-c', metavar='C', type=str, 
                        help='Clip values', default='-1')
    parser.add_argument('--estimators', '-e', metavar='E', type=str, 
                        help='Learning methods', default='Naive,Gold-IPS,Gold-SNIPS')
    parser.add_argument('--metric', '-m', metavar='M', type=str, 
                        help='Metrics', default='MSE')
                        
    args = parser.parse_args()
    numpy.random.seed(args.seed)
    
    approaches = args.estimators.strip().split(',')
    
    approachDict = {}
    for approach in approaches:
        approachDict[(approach, args.metric)] = len(approachDict)
        
    alphas = []
    tokens = args.alphas.strip().split(',')
    for token in tokens:
        alphas.append(float(token))
    
    lambdas = []
    tokens = args.lambdas.strip().split(',')
    for token in tokens:
        lambdas.append(float(token))
        
    numDims = []
    tokens = args.numdims.strip().split(',')
    for token in tokens:
        numDims.append(int(token))
    
    clipVals = []
    tokens = args.clips.strip().split(',')
    for token in tokens:
        clipVals.append(int(token))

    numAlphas = len(alphas)
    numApproaches = len(approachDict)
    
    biasModes = ['Free']
    numDimSettings = len(numDims)
    numClipSettings = len(clipVals)
    numBiasModes = len(biasModes)
 
    numLambdas = len(lambdas)
    numParamSettings = numLambdas * numDimSettings * numClipSettings * numBiasModes
    
    paramSettings = list(itertools.product(lambdas, numDims, clipVals, biasModes))
       
    ML100KCompleteTest = Datasets.ML100K('../')
    
    allEstimates = numpy.zeros((numApproaches, numAlphas), dtype = numpy.longdouble)
    
    currMetric = None
    if args.metric == 'MSE':
        currMetric = Metrics.MSE
    elif args.metric == 'MAE':
        currMetric = Metrics.MAE
    else:
        print "Expt2: [ERR] Unrecognized metric", args.metric
        sys.exit(0)
        
    print "Expt2: [LOG] Starting metric", args.metric
        
    def updateResults(val, approach, ind):
        approachTuple = (approach, args.metric)
        approachIndex = approachDict[approachTuple]
        allEstimates[approachIndex, ind] = val
            
    print "Expt2: [LOG] Trial", args.trial
    numpy.random.seed(args.seed + args.trial)
    
    outputFile = '../logs/expt2/'+str(args.seed)+'_'+str(args.trial)+'_'+args.metric+'_'
    
    for ind, alpha in enumerate(alphas):
        print "Expt2: [LOG] Alpha:", alpha

        partialObservations, goldPropensities = Propensity.PARTIAL_OBSERVE(ML100KCompleteTest, alpha, 0.05, verbose = False)
       
        flatObservations = numpy.ma.compressed(partialObservations) 
        observedHistogram = numpy.bincount(flatObservations, minlength = 6)[1:]
        observedHistogram = observedHistogram.astype(numpy.longdouble) / \
                                observedHistogram.sum(dtype = numpy.longdouble)
        print "Expt2: [LOG] Observed Marginals: ", observedHistogram
       
        goldInvPropensities = numpy.reciprocal(goldPropensities)
        
        foldScores = numpy.zeros((numApproaches, 4, numParamSettings), dtype = numpy.longdouble)
        foldTestScores = numpy.zeros((numApproaches, 4, numParamSettings), dtype = numpy.longdouble)
        
        observationIndices = numpy.ma.nonzero(partialObservations)
        numObservations = numpy.ma.count(partialObservations)
 
        shuffleIndices = numpy.random.permutation(numObservations)
        fractionObservations = int(numObservations/4)
        firstFold = shuffleIndices[:fractionObservations]
        secondFold = shuffleIndices[fractionObservations:2*fractionObservations]
        thirdFold = shuffleIndices[2*fractionObservations:3*fractionObservations]
        fourthFold = shuffleIndices[3*fractionObservations:]
        print "Expt2: [LOG] Split %d observations into folds. Fold sizes:" % len(shuffleIndices),\
                        len(firstFold), len(secondFold), len(thirdFold), len(fourthFold)
        
        for fold in xrange(4):
            print "Expt2: [LOG] Fold:", fold
            trainObservations = numpy.ma.copy(partialObservations)
            testObservations = numpy.ma.copy(partialObservations)

            if fold == 0:
                trainObservations[observationIndices[0][firstFold], observationIndices[1][firstFold]] = \
                                        numpy.ma.masked

                testObservations[observationIndices[0][secondFold], observationIndices[1][secondFold]] = \
                                        numpy.ma.masked
                testObservations[observationIndices[0][thirdFold], observationIndices[1][thirdFold]] = \
                                        numpy.ma.masked
                testObservations[observationIndices[0][fourthFold], observationIndices[1][fourthFold]] = \
                                        numpy.ma.masked
            elif fold == 1:
                trainObservations[observationIndices[0][secondFold], observationIndices[1][secondFold]] = \
                                        numpy.ma.masked

                testObservations[observationIndices[0][firstFold], observationIndices[1][firstFold]] = \
                                        numpy.ma.masked
                testObservations[observationIndices[0][thirdFold], observationIndices[1][thirdFold]] = \
                                        numpy.ma.masked
                testObservations[observationIndices[0][fourthFold], observationIndices[1][fourthFold]] = \
                                        numpy.ma.masked
            elif fold == 2:
                trainObservations[observationIndices[0][thirdFold], observationIndices[1][thirdFold]] = \
                                        numpy.ma.masked

                testObservations[observationIndices[0][firstFold], observationIndices[1][firstFold]] = \
                                        numpy.ma.masked
                testObservations[observationIndices[0][secondFold], observationIndices[1][secondFold]] = \
                                        numpy.ma.masked
                testObservations[observationIndices[0][fourthFold], observationIndices[1][fourthFold]] = \
                                        numpy.ma.masked
            elif fold == 3:
                trainObservations[observationIndices[0][fourthFold], observationIndices[1][fourthFold]] = \
                                        numpy.ma.masked

                testObservations[observationIndices[0][firstFold], observationIndices[1][firstFold]] = \
                                        numpy.ma.masked
                testObservations[observationIndices[0][secondFold], observationIndices[1][secondFold]] = \
                                        numpy.ma.masked
                testObservations[observationIndices[0][thirdFold], observationIndices[1][thirdFold]] = \
                                        numpy.ma.masked
            else:
                print "Expt2: [ERR] #Folds not supported ", fold
                sys.exit(0)
                   
            #Get starting params by SVD
            startFileName = outputFile + 'fold' + str(fold) + '_' + str(alpha) +'_init.pkl'
            startTuple = None
            if os.path.exists(startFileName):
                g = open(startFileName, 'rb')
                startTuple = pickle.load(g)
                g.close()
            else:
                startTuple = INIT_PARAMS(trainObservations, 20)
                g = open(startFileName, 'wb')
                pickle.dump(startTuple, g, -1)
                g.close()
            
            for approach in approaches:
                print "Starting approach ", approach
                invP, normN = TRAIN_HELPER(approach, goldInvPropensities, None)
                approachTuple = (approach, args.metric)
                approachIndex = approachDict[approachTuple]
                modelFileName = outputFile + 'fold' + str(fold) + '_' + str(alpha) +'_'+approach+'.pkl'
                modelsPerLambda = None
                if os.path.exists(modelFileName):
                    g = open(modelFileName, 'rb')
                    modelsPerLambda = pickle.load(g)
                    g.close()
                    print "Expt2: [LOG]\t Loaded trained models for each lambda from ", modelFileName
                else:
                    modelsPerLambda = Parallel(n_jobs = -1, verbose = 0)(delayed(MF_TRAIN)(l2Lambda, 
                                            trainObservations, invP, normN, args.metric, startTuple)
                                            for l2Lambda in paramSettings)
                    g = open(modelFileName, 'wb')
                    pickle.dump(modelsPerLambda, g, -1)
                    g.close()
                    print "Expt2: [LOG]\t Saved trained models for each lambda to ", modelFileName

                for lambdaIndex, eachModel in enumerate(modelsPerLambda):
                    selectedBiasMode = paramSettings[lambdaIndex][3]
                    selectedBias = True
                    if selectedBiasMode == 'None':
                        selectedBias = False

                    predictedY = MF.PREDICTED_SCORES(eachModel[0], eachModel[1], 
                                                eachModel[2], eachModel[3], eachModel[4], use_bias = selectedBias)
                    score = None
                    if invP is not None:
                        score = currMetric(testObservations, predictedY, 4.0*invP)
                    else:
                        score = currMetric(testObservations, predictedY, None)
                        
                    if normN == 'Vanilla':
                        score = score[0]
                    elif normN == 'SelfNormalized':
                        score = score[1]
                    elif normN == 'UserNormalized':
                        score = score[2]
                    elif normN == 'ItemNormalized':
                        score = score[3]
                    else:
                        print "Expt2: [ERR] Normalization not supported for metric ", normN, args.metric
                        sys.exit(0)
                    
                    foldScores[approachIndex, fold, lambdaIndex] = score
                    
                    foldTestScore = currMetric(ML100KCompleteTest, predictedY, None)[0]
                    foldTestScores[approachIndex, fold, lambdaIndex] = foldTestScore
                    print "Expt2: [LOG] Lambda/NumDims: ", paramSettings[lambdaIndex], \
                                "\t Test Fold Score: ", score, "\t Test Set Score: ", foldTestScore
                
                #Save foldScores and foldTestScores after each approach in each fold. Overwrite if needed.
                scoresFile = outputFile + 'Alpha'+ str(alpha)+'_foldScores.pkl'
                scoresData = (foldScores, foldTestScores)
                g = open(scoresFile, 'wb')
                pickle.dump(scoresData, g, -1)
                g.close()
                sys.stdout.flush()        
        
        eventualApproachParams = []
        for approach in approaches:
            invP, normN = TRAIN_HELPER(approach, goldInvPropensities, None)
            approachTuple = (approach, args.metric)
            approachIndex = approachDict[approachTuple]
            approachScores = foldScores[approachIndex,:,:]
            allFoldScores = approachScores.sum(axis = 0, dtype = numpy.longdouble)
            bestLambdaIndex = numpy.argmin(allFoldScores)
            bestLambda = paramSettings[bestLambdaIndex]
            print "FINAL_TRAIN: [LOG] Retraining ", approach, " Best lambda/numDims", bestLambda    
            for everyLambdaIndex, everyLambda in enumerate(paramSettings):
                print "FINAL_TRAIN: [DBG] AllFoldScores: ", approach, everyLambda, allFoldScores[everyLambdaIndex]
            eventualApproachParams.append((approach,invP,normN,bestLambda))
            
        finalModels = None
        finalModelFileName = outputFile + str(alpha) +'_finalmodels.pkl'
        if os.path.exists(finalModelFileName):
            g = open(finalModelFileName, 'rb')
            finalModels = pickle.load(g)
            g.close()
            print "Expt2: [LOG]\t Loaded trained final models from ", finalModelFileName
        else:
            finalModels = Parallel(n_jobs = -1, verbose = 0)(delayed(FINAL_TRAIN)(approachTup, args.metric, 
                                        partialObservations, startTuple)
                                        for approachTup in eventualApproachParams)
            g = open(finalModelFileName, 'wb')
            pickle.dump(finalModels, g, -1)
            g.close()
            print "Expt2: [LOG]\t Saved trained final models to ", finalModelFileName
        
        for approachID, approachTuple in enumerate(eventualApproachParams):
            resultTuple = finalModels[approachID]
            finalBiasMode = approachTuple[3][3]
            finalBias = True
            if finalBiasMode == 'None':
                finalBias = False
            predictedY = MF.PREDICTED_SCORES(resultTuple[0], resultTuple[1], \
                                resultTuple[2], resultTuple[3], resultTuple[4], use_bias = finalBias)

            metricValue = currMetric(ML100KCompleteTest, predictedY, None)[0]
            print "Expt2: [LOG] ", approachTuple[0], "\t Eventual result:", metricValue
            sys.stdout.flush()
            updateResults(metricValue, approachTuple[0], ind)    
        
        #Dump results after each alpha. Overwrite if needed.
        outputData = (approaches, approachDict, alphas, allEstimates)
        g = open(outputFile + args.estimators +'.pkl', 'wb')
        pickle.dump(outputData, g, -1)
        g.close()
