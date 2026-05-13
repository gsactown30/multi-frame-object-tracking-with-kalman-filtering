import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from kalman_tracker import KalmanTracker

class KalmanManager:
    def __init__(self, objVariance, measureVariance, distThreshold, coastThreshold):
        self.trackers = []
        self.objVariance = objVariance
        self.measureVariance = measureVariance
        self.distThreshold = distThreshold
        self.coastThreshold = coastThreshold
        self.IDcounter = 0

    def updateFrame(self, detections):
        if len(self.trackers) == 0:
            for d in detections:
                newTracker = KalmanTracker(d, self.objVariance, self.measureVariance, self.IDcounter)
                self.trackers.append(newTracker)
                self.IDcounter += 1
        elif len(detections) == 0:
            for t in self.trackers:
                t.predict()
            self.trackers = [t for t in self.trackers if t.numPredicts <= self.coastThreshold]
        else:
            # 1. predict
            for t in self.trackers:
                t.predict()
            # 2. build cost matrix
            tPositions = np.array([t.getCoord() for t in self.trackers])

            dPositions = np.array([d for d in detections])

            costMatrix = cdist(tPositions, dPositions, 'euclidean')


            # 3. hungarian match
            row, col = linear_sum_assignment(costMatrix)

            # 4. threshold check all
            unmatchDet = []
            for r, c in zip(row, col):
                if costMatrix[r, c].sum() < self.distThreshold:
                    self.trackers[r].update(detections[c])
                else:
                    unmatchDet.append(c)
            #5. set difference to find unmatched detections → create new trackers
            matchDet = set(col)
            allDet = set(range(len(detections)))
            noMatchDet = allDet - matchDet
            noMatchDet = noMatchDet | set(unmatchDet)
            for d in noMatchDet:
                newTracker = KalmanTracker(detections[d], self.objVariance, self.measureVariance, self.IDcounter)
                self.trackers.append(newTracker)
                self.IDcounter += 1
            #6. filter self.trackers by coastThreshold
            self.trackers = [t for t in self.trackers if t.numPredicts <= self.coastThreshold]