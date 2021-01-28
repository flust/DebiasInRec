import argparse
import itertools
import os
import sys

import Expt2
import MF
import Metrics
import numpy
from joblib import Parallel, delayed

def learn(data, logger, lambdas=None, seed=None, numDims=None, approach=None, metric=None, raw_metric=None,
          output_name=None, propensities_desc=None):
    clipVals = [-1]
    biasModes = ['Free', 'Regularized']
    numpy.random.seed(seed)

    numBiasModes = len(biasModes)
    numLambdas = len(lambdas)
    numDimSettings = len(numDims)
    numClipSettings = len(clipVals)
    numParamSettings = numLambdas * numDimSettings * numClipSettings * numBiasModes

    paramSettings = list(itertools.product(lambdas, numDims, clipVals, biasModes))
    numApproaches = 1

    selfMatrix = data.train

    logger.log("Starting learning...")
    logger.log("\t-metric: " + raw_metric, 2)
    logger.log("\t-lambda values: " + str(lambdas), 2)
    logger.log("\t-dimension values: " + str(numDims), 2)
    logger.log("\t-propensity scoring method: " + propensities_desc)
    if data.propensities is not None:
        invP = numpy.reciprocal(data.propensities)
        invP = numpy.ma.array(invP, dtype=numpy.longdouble, copy=False,
                              mask=numpy.ma.getmask(selfMatrix), fill_value=0, hard_mask=True)
    else:
        invP = None

    foldScores = numpy.zeros((numApproaches, 4, numParamSettings), dtype=numpy.float)

    observationIndices = numpy.ma.nonzero(selfMatrix)
    numObservations = numpy.ma.count(selfMatrix)

    shuffleIndices = numpy.random.permutation(numObservations)
    # 5 fold
    fractionObservations = int(numObservations / 4)
    firstFold = shuffleIndices[:fractionObservations]
    secondFold = shuffleIndices[fractionObservations:2 * fractionObservations]
    thirdFold = shuffleIndices[2 * fractionObservations:3 * fractionObservations]
    fourthFold = shuffleIndices[3 * fractionObservations:]

    logger.log("Split %d observations into folds. Fold sizes: %s" %
               (len(shuffleIndices), str([len(firstFold), len(secondFold), len(thirdFold), len(fourthFold)])),
               2)

    for fold in xrange(4):
        # 5 fold
        logger.log("Learning on fold %d " % fold)
        trainObservations = numpy.ma.copy(selfMatrix)
        testObservations = numpy.ma.copy(selfMatrix)

        if fold == 0:
            trainObservations[observationIndices[0][firstFold], observationIndices[1][firstFold]] = \
                numpy.ma.masked
            ## masked the first fold in train set

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

        # Get starting params by SVD
        startTuple = Expt2.INIT_PARAMS(trainObservations, 40)
        normN = "Vanilla"
        approachIndex = 0

        # train MF
        modelsPerLambda = Parallel(n_jobs=-1, verbose=0)(delayed(Expt2.MF_TRAIN)(param,
                                                                                 trainObservations, invP, normN,
                                                                                 raw_metric, startTuple)
                                                         for param in paramSettings)

        for lambdaIndex, eachModel in enumerate(modelsPerLambda):
            selectedBiasMode = paramSettings[lambdaIndex][3]
            selectedBias = True
            if selectedBiasMode == 'None':
                selectedBias = False
            predictedY = MF.PREDICTED_SCORES(eachModel[0], eachModel[1],
                                             eachModel[2], eachModel[3], eachModel[4], use_bias=selectedBias)

            score = None
            if invP is not None:
                score = metric(testObservations, predictedY, 4.0 * invP)
            else:
                score = metric(testObservations, predictedY, invP)
            score = score[0]
            foldScores[approachIndex, fold, lambdaIndex] = score

            logger.log("\tLambda/NumDims: " + str(paramSettings[lambdaIndex]) +
                       ", Test Fold Score: " + str(score), 2)

    eventualApproachParams = []

    normN = "Vanilla"
    approachIndex = 0
    approachScores = foldScores[approachIndex, :, :]
    allFoldScores = approachScores.sum(axis=0, dtype=numpy.float)
    bestLambdaIndex = numpy.argmin(allFoldScores)
    bestLambda = paramSettings[bestLambdaIndex]
    logger.log("Retraining with best hyperparameter values: " + str(bestLambda))
    logger.log("Chosen from average cross-validation performance:", 2)
    for everyLambdaIndex, everyLambda in enumerate(paramSettings):
        logger.log("\t" + str(everyLambda) + ": " +  str(allFoldScores[everyLambdaIndex]), 2)
    eventualApproachParams.append((approach, invP, normN, bestLambda))

    finalModels = Parallel(n_jobs=-1, verbose=0)(delayed(Expt2.FINAL_TRAIN)(approachTup,
                                                                            raw_metric, selfMatrix, startTuple)
                                                 for approachTup in eventualApproachParams)

    for approachID, approachTuple in enumerate(eventualApproachParams):
        resultTuple = finalModels[approachID]
        finalBiasMode = approachTuple[3][3]
        finalBias = True
        if finalBiasMode == 'None':
            finalBias = False

        predictedY = MF.PREDICTED_SCORES(resultTuple[0], resultTuple[1],
                                         resultTuple[2], resultTuple[3], resultTuple[4], use_bias=finalBias)
        numpy.savetxt(output_name, predictedY)
        logger.log("Done.")
