import json
import math
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta  # $ pip install python-dateutil




M = 1000000
K = 1000

"""
aut has to be 3, 4, or 5
"""


def simulate(oldPrice, oldPerf, oldSz, oldMtbf, oldCap, oldAut, startAgeYears, price, perf, sz, mtbf, cap, aut,
             segment):
    totalLifeYears = 2
    rndYears = rndTimeYears(oldAut)
    for month in range(1, totalLifeYears * 12 + 1):
        curAge = currentAgeYears(month, rndYears, startAgeYears, oldPerf == perf and oldSz == sz)
        tperf, tsz, tmtbf = oldPerf, oldSz, oldMtbf
        tcap, taut = oldCap, oldAut
        if month / 12 > rndYears:
            tperf, tsz, tmtbf = perf, sz, mtbf
        if month / 12 > 1:
            tcap, taut = cap, aut
        # TODO


def testSimulate2():
    x = testSalesEat()
    # perfX, szY, rndYears = zip(*x)
    # print(perfX)

products = {
    "eat":{
        "price": 28,
        "perf": 5.5,
        "sz": 14.5,
        "mtbf": 17.5*K,
        "ageYears": 3.1,
        "awareness": .55,
        "accessibility":.37,
        "aut": 4,
        "capacity": 1800*K,
        "overProduction": 0,
        "score": {
            "traditional": 0.136,
        },
    }, "ebb": {
        "price": 21,
        "perf": 3,
        "sz": 17,
        "mtbf": 14*K,
        "ageYears": 4.6,
        "awareness": .52,
        "accessibility":.4,
        "aut": 5,
        "capacity": 1400*K,
        "overProduction": .3,
        "score": {
            "traditional": .0303,
            "low":1/6
        },
    }, "echo": {
        "price": 38,
        "perf": 8,
        "sz": 12,
        "mtbf": 23*K,
        "ageYears": 1.7,
        "awareness": .49,
        "accessibility":.48,
        "aut": 3,
        "capacity": 900,
        "overProduction": 0,
        "score": {
            "high": 1/6,
        },
    }, "edge": {
        "price": 33,
        "perf": 9.4,
        "sz": 15.5,
        "mtbf": 25*K,
        "ageYears": 2.5,
        "awareness": .46,
        "accessibility":.37,
        "aut": 3,
        "capacity": 600*K,
        "overProduction": 0,
        "score": {
            "performance": 1/6,
        },
    }, "egg": {
        "price": 33,
        "perf": 4,
        "sz": 11,
        "mtbf": 19*K,
        "ageYears": 2.6,
        "awareness": .46,
        "accessibility":.42,
        "aut": 3,
        "capacity": 600*K,
        "overProduction": 0,
        "score": {
            "size": .155,
        },
    }
}
def main():
    testSalesEat()

def simulate2():
    segment = "performance"
    seg = segments[segment]
    product = products["edge"]
    estimatedSalesPerYear = seg["demand"]
    baseScoreEmpiricalDenominator = score(seg, product["ageYears"], product["price"], product["perf"], product["sz"], product["mtbf"], product["awareness"], product["accessibility"])
    baseScoreReal = product["score"][segment]
    aut = product["aut"]
    overProduction = product["overProduction"]

    ebbPerf, ebbSz, ebbMtbf = product["perf"], product["sz"], product["mtbf"]

    perff = list(np.linspace(ebbPerf-5, ebbPerf, 30)) + list(np.linspace(ebbPerf, ebbPerf+5, 30))[1:]
    szz = list(np.linspace(ebbSz-5, ebbSz, 20)) + list(np.linspace(ebbSz, ebbSz+5, 20))[1:]
    pricee = list(np.linspace(seg["price"][0], product["price"], 20)) + list(np.linspace(product["price"], seg["price"][1], 20))[1:]
    ll = []
    dummy = 0
    for perf in perff:
        # if abs(perf - ebbPerf) > .01:
        #     continue
        for sz in szz:
            if perf <= 0 or sz <= 0:
                continue
            # if abs(sz - ebbSz) > .01:
            #     continue
            for mtbf in range(seg["mtbf"][0], seg["mtbf"][1] + 1000, 1000):
                # if abs(mtbf - ebbMtbf) > 1500:
                #     continue
                d = dist(ebbPerf, ebbSz, perf, sz)
                rndYear = rndTimeYears(aut, d, abs(ebbMtbf - mtbf), 3)
                if rndYear < 1:
                    dummy += 1
                    for price in pricee:
                        # if abs(price - product["price"]) > .01:
                        #     continue
                        estimatedRevenue = 0
                        estimatedCost = 0
                        for month in range(1, 12):
                            estimatedCostDelta, estimatedRevenueDelta = sales(baseScoreEmpiricalDenominator, baseScoreReal,                                                                    ebbMtbf, ebbPerf, ebbSz, estimatedSalesPerYear, month,                                                                    mtbf, perf, sz, price, rndYear, seg, aut, overProduction)
                            estimatedCost += estimatedCostDelta
                            estimatedRevenue += estimatedRevenueDelta

                        estimatedCost += rndYear*M
                        if estimatedCost == 0:
                            continue
                        totalScore = estimatedRevenue/estimatedCost
                        revenueGrowth = estimatedRevenue/(price *estimatedSalesPerYear*baseScoreReal)
                        # print("c:{} s:{}, s/c:{}".format(estimatedCost, estimatedRevenue, estimatedRevenue/estimatedCost))
                        # if totalScore > 1.4:
                        # ll.append(totalScore)
                        if totalScore > 1.35 and revenueGrowth>1.8:
                            # ll.append(revenueGrowth)
                            ll.append((perf, sz, rndYear*12))
                            print("dist:{} perf:{} sz:{} mtbf:{} price:{} rndTimeMonths:{} totalScore:{} revenue:{} sales {} growthInRev:{}".format(d, perf, sz, mtbf, price, rndYear*12, totalScore, estimatedRevenue, estimatedRevenue/price, revenueGrowth))
    return ll

def testSalesEat():
    segment = "size"
    seg = segments[segment]
    product = products["egg"]
    baseScoreEmpiricalDenominator = score(seg, product["ageYears"], product["price"], product["perf"], product["sz"], product["mtbf"], product["awareness"], product["accessibility"])
    baseScoreReal = product["score"][segment]
    estimatedSalesPerYear = seg["demand"]

    aut = product["aut"]
    overProduction = product["overProduction"]
    price = product["price"]
    ebbPerf, ebbSz, ebbMtbf = product["perf"], product["sz"], product["mtbf"]
    aPerf, aSz, aMtbf, aPrice = 4.1, 11, 19000, 31 #24 mar
    sale = 0
    s = 0
    c = 0
    print("baseMul:{} baseSales:{} BaseadjSales:{} baseScore:{}".format(baseScoreReal, estimatedSalesPerYear, baseScoreReal*estimatedSalesPerYear, baseScoreEmpiricalDenominator))
    print("=============")
    for month in range(1, 13):

        estimatedCost, estimatedRevenue = sales(baseScoreEmpiricalDenominator, baseScoreReal, ebbMtbf, ebbPerf, ebbSz, estimatedSalesPerYear, month, aMtbf, aPerf, aSz, aPrice, .24/12, seg, aut, overProduction)
        s+=estimatedRevenue
        c += estimatedCost
        sale += estimatedRevenue/price
    d = dist(ebbPerf, ebbSz, aPerf, aSz)
    rndYear = rndTimeYears(aut, d, abs(ebbMtbf - aMtbf), aut)
    c+=rndCost(rndYear)

    profitMargin = s/c
    growthInSales = s/(price *estimatedSalesPerYear*baseScoreReal)
    print("Totalrevenue:{} Totalcost:{} Totalsales:{} profitMargin:{} growthInSales:{}, rndYear:{} ".format(s, c, sale, profitMargin, growthInSales, rndYear))


def sales(baseScoreEmpiricalDenominator, baseScoreReal, ebbMtbf, ebbPerf, ebbSz,
          estimatedSalesPerYear, month, mtbf, perf, sz, price, rndYear, seg, aut, overProduction):
    seg2 = segmentState(month, seg)
    age = currentAgeYears(month, rndYear, 4.6)
    aware = awareness(month, rndYear)
    tperf, tsz, tmtbf = ebbPerf, ebbSz, ebbMtbf
    if month >= rndYear * 12:
        tperf, tsz, tmtbf = perf, sz, mtbf
    sc = score(seg2, age, price, tperf, tsz, tmtbf, aware, .5)
    baseScoreEmpirical = sc / baseScoreEmpiricalDenominator
    scoreMultiplier = baseScoreEmpirical * baseScoreReal / (1 + baseScoreEmpirical * baseScoreReal - baseScoreReal)
    estimatedSalesAdjMonth = scoreMultiplier * estimatedSalesPerYear * (seg2["growth"] ** int((month + 11) / 12)) / 12
    cost = totalVariableCostExistingProduct(tperf, tsz, tmtbf, aut, overProduction, month)
    estimatedRevenue = estimatedSalesAdjMonth * price
    estimatedCost = estimatedSalesAdjMonth * cost
    # print("month:{}, mul: {}, adjSales:{} varCost:{} totalCost:{} sc:{}".format(month, scoreMultiplier, estimatedSalesAdjMonth*12, cost, estimatedCost, sc))
    return estimatedCost, estimatedRevenue


# tested
def testRndEbb():
    start = date(2021, 12, 31)
    perf, sz, mtbf = 4, 17, 14 * K
    deltaYears = rndTimeYears(5, dist(perf, sz, 3, 17), abs(mtbf - 14 * K), 1)
    print(deltaYears)
    print("revision date is ", start + relativedelta(days=int(deltaYears * 365)))
    print("cost is ", rndCost(deltaYears))

def testSegmentState():
    seg = segmentState(1, segments["low"])

    print(json.dumps(seg, indent=2))


def segmentState(month, segment):
    seg = segment.copy()
    year = (month)/ 12
    yearInt = int((month + 11)/12)
    p = seg["pos"]
    driftPerYear = seg["driftPerYear"]
    seg["pos"] = [p[0] + year * driftPerYear[0], p[1] + year * driftPerYear[1], \
                  p[2] + year * driftPerYear[0], p[3] + year * driftPerYear[1]]
    if yearInt >= 1:
        seg["demand"] *= (seg["growth"] ** yearInt)
        seg["price"] = [seg["price"][0] - 0.5 * yearInt, seg["price"][1] - 0.5 * yearInt]
    return seg


# tested
def testCurrentAge():
    for month in range(1, 12):
        print(month, currentAgeYears(month, .59, 4.6))


"""
skipRevision = True if the device is not really revised due to no change in position
"""


def currentAgeYears(month, rndYears, startAgeYears, skipRevision=False):
    if month / 12 < rndYears or skipRevision or rndYears == 0:
        return startAgeYears + month / 12.0
    else:
        ageAtRevision = (startAgeYears + rndYears) / 2
        ageAfterRevision = month / 12 - rndYears
        return ageAtRevision + ageAfterRevision


segments = {
    "traditional": {
        "age": [[-1, 0, 0, 0.1], [0, 2, 0.45, 0.1], [2, 4, -.45, 1.9], [4, 11, 0, 0.1]],
        "price": [20, 30],
        "pos": [5, 15, 5, 15],
        "mtbf": [14000, 19000],
        "weights": [.47, .23, .21, .09],
        "growth": 1.092,
        "driftPerYear": [.7, -.7],
        "demand": 7387 * K
    },
    "low": {
        "age": [[-1, 0, 0, 0.1], [0, 2, 0.02, 0.08], [2, 7, 0.176, -0.232], [7, 11, -0.176, 2.232]],
        "price": [15, 25],
        "pos": [2.5, 17.5, 1.7, 18.3],
        "mtbf": [12000, 17000],
        "weights": [.47, .23, .21, .09],
        "growth": 1.117,
        "driftPerYear": [.5, -.5],
        "demand": 8960 * K

    },
    "high": {
        "age": [[0, 0.7, -4 / 25.0, 1], [.7, 2.4, -359 / 1000, 11393 / 10000], [2.4, 3.4, -1117 / 10000, 27289 / 50000],
                [3.4, 11, 0, 1.5 / 9]],
        "price": [30, 40],
        "pos": [7.5, 12.5, 8.9, 11.1],
        "mtbf": [20000, 25000],
        "weights": [.47, .23, .21, .09],
        "growth": 1.162,
        "driftPerYear": [.9, -.9],
        "demand": 2554 * K,
    },
    "performance": {
        "age": [[0, 1, 0.2, 0.8], [1, 3.5, -0.36, 1.36], [3.5, 11, 0, 0.1]],
        "price": [25, 35],
        "pos": [8.0, 17.0, 9.4, 16.0],
        "mtbf": [22000, 27000],
        "weights": [.47, .23, .21, .09],
        "growth": 1.198,
        "driftPerYear": [1, -.7],
        "demand": 1915 * K

    },
    "size": {
        "age": [[0, 1.5, 0.33, 0.5], [1.5, 4, -0.36, 1.54], [4, 11, 0, 0.1]],
        "price": [25, 35],
        "pos": [3.0, 12.0, 4.0, 10.6],
        "mtbf": [16000, 21000],
        "weights": [.47, .23, .21, .09],
        "growth": 1.183,
        "driftPerYear": [.7, -1.0],
        "demand": 1984 * K
    }
}


# tested
def ageScore(segment, x):
    return y(segment["age"], x)


# tested
def priceScore(segment, x):
    l, u, = segment["price"]
    if x > u:
        return 0
    return (x - u) / (l - u)


"""
decision variables
m:       related to MTBF. It is the lower end percentage
pc:      position constant: the decrease in score per unit distance inside the fine cut circle
         pc has to be less than 1/2.5 = 0.4
         
awr:     awareness is a fraction between 0 and 1. it costs $ to buy awareness
acs:     accessibility is a fraction between 0 and 1. it costs $ to buy accesibility. 
         It is set per product, but it accumulates to the segment.
"""

m = 0.7
pc = 0.75 / 2.5
awr = 0.8
acs = .5

awarenessCurve = [[0, .52], [1, .52 - .33 / 12 + .4 / 12]]


def awareness(month, revisionTimeYears):
    baseAwareNess = yy(awarenessCurve, month, 0, "extrapolate")
    newsWorthy = 0
    if month > revisionTimeYears * 12 and revisionTimeYears > 0:
        baseAwareNess += .25
    baseAwareNess = min(1, baseAwareNess)
    return baseAwareNess


def mtbfScore(segment, x):
    global m
    l, u = segment["mtbf"]
    segmentsLocal = [[l - 5000, 0], [l, m], [u, 1]]
    return yy(segmentsLocal, x, defaultRight=1, defaultLeft=0)


def testPositionScore():
    # segment = segmentState(12, "traditional")
    print(positionScore(segments["traditional"], 3, 17))


def positionScore(segment, perf, sz):
    cx, cy, ix, iy = segment["pos"]
    d = dist(cx, cy, perf, sz)
    if d <= 2.5:
        actualD = dist(ix, iy, perf, sz)
        return 1 - pc * actualD
    elif perf == cx:
        x, y = cx, cy + 2.5
    else:
        tperf, tsz = perf - cx, sz - cy
        # p = 2.5 / math.sqrt(1 + tsz * tsz / (tperf * tperf))
        # q = tsz * p / tperf
        # x = p + cx
        # y = q + cy
        p = 2.5 / math.sqrt(1 + tsz * tsz / (tperf * tperf))
        if not (min(perf, cx) <= p + cx <= max(perf, cx)):
            p = -p
        q = tsz * p / tperf
        x = p + cx
        y = q + cy

    distanceFromIdeal = dist(ix, iy, x, y)
    residualDistance = dist(x, y, perf, sz)
    scoreAtBoundary = 1 - pc * distanceFromIdeal
    finalScore = yy([[0, scoreAtBoundary], [1.5, 0]], residualDistance, 0, 0)
    return finalScore


def testScore():
    s = score("traditional", 0, 4.6, 21, 3, 17, 14000, .52, .54)
    print("final score is ", s)
    return s


def score(segment, agee, pricee, perf, sz, mtbff, awareness, accessibility):
    ageS = ageScore(segment, agee)
    priceS = priceScore(segment, pricee)
    positionS = positionScore(segment, perf, sz)
    mtbfS = mtbfScore(segment, mtbff)
    if ageS <= 0 or priceS <= 0 or positionS <= 0 or mtbfS <= 0:
        return 0
    w1, w2, w3, w4 = segment["weights"]
    # print("raw scores age:{} price:{} pos:{}, mtbf:{}".format(ageS, priceS, positionS, mtbfS))
    baseScore = w1 * ageS + w2 * priceS + w3 * positionS + w4 * mtbfS
    # print("weighted scores age score:{} priceScore:{} posScore:{} mtbfScore:{}".format(w1 * ageS * 100, w2 * priceS * 100, w3 * positionS * 100, w4 * mtbfS * 100))
    # print("basecore is ", baseScore)
    accountsRecievable = 0.993
    finalScore = baseScore * accountsRecievable * awareness * accessibility
    return finalScore * 100


def testTotalVariableCost():
    print(totalVariableCostExistingProduct(3, 17, 14000, 5, 0.3, 12))

# the fixed cost only the RnDCost
def totalVariableCostExistingProduct(perf, sz, mtbfLocal, aut, overProduction, month):
    pCost = positionMaterialCost(perf, sz)
    mCost = mtbfMaterialCost(mtbfLocal)
    lCost = labourCost(aut, overProduction, month)
    # print(pCost + mCost, lCost)
    return pCost + mCost + lCost


def dist(a, b, perf, sz):
    return np.sqrt((sz - b) * (sz - b) + (perf - a) * (perf - a))


def testPositionMaterialCost():
    print(positionMaterialCost(5.5, 14.5))
    print(mtbfMaterialCost(17500))


def positionMaterialCost(perf, sz):
    d = dist(0.732, 19.268, perf, sz)
    slope = .7456
    return 1 + slope * d


def mtbfMaterialCost(mtbfLocal):
    return mtbfLocal * 0.3 / 1000.0


def rndCost(timeYears):
    return M * timeYears


def testLaborCost():
    print(labourCost(5, 1))


labourBaseCost = 10.7


def labourCost(aut, overProduction, month):
    year = int((month+11)/12)
    labour = labourBaseCost * (1.05**year)
    # currently hardcoded to 0.99. In future versions, we will give correct values
    apLagPenalty = .99
    cost = 0.1 * (11 - aut) * labour * (1 + 1.5 * overProduction) / (1 + overProduction) / apLagPenalty
    return cost


segmentsForAutToYears = [[1, 0.47], [2, 0.47], [3, 0.48], [4, 0.5], [5, 0.51], [6, 0.69], [7, 0.81], [8, 0.99],
                         [9, 1.2],
                         [10, 1.48]]

aut3 = [[0, 0], [.1, .17], [.2, .150], [.4, .209], [.6, .295], [1, .513], [2, 1.124], [4, 1.993], [8, 2.532]]
aut4 = [[0, 0], [.1, .231], [.2, .187], [.4, .246], [.6, .332], [1, .55], [2, 1.161], [4, 2.03], [8, 2.569]]
aut5 = [[0, 0], [.1, .133], [.2, .248], [.4, .307], [.6, .393], [1, .611], [2, 1.222], [4, 2.091], [8, 2.630]]

timePer40KDollar = 40 * K / M
timePer90KDollar = 90 * K / M


def rndTimeYears(aut, dist, deltaMtbf, numProjects=1):
    autSeg = [aut3, aut4, aut5][aut - 3]
    distBasedTime = yy(autSeg, dist, defaultRight=2.6)
    deltaMtbfBasedTime = deltaMtbf * timePer40KDollar / 1000
    totalTime = (distBasedTime + deltaMtbfBasedTime)
    return totalTime


awarenessSegments = [[0, 0.02, 15000000, 0], [0.02, 0.05, 5000000 / 3.0, 800000 / 3.0],
                     [0.05, 0.4, 4000000, 150000], [0.4, 0.46, 25000000 / 3.0, -4750000 / 3.0],
                     [.46, 5, 18750000, -6375000], [5, 10000000, 100000000000, 0]]


def awarenessCost(x):
    return y(awarenessSegments, x)


accessibilitySegmentMultiple = [[0, 0.05, 15000000, 0], [0.05, 0.22, 125000000 / 17.0, 6500000 / 17.0],
                                [0.22, 0.275, 100000000 / 11, 0], [0.275, 0.31, 100000000 / 7.0, -10000000 / 7.0],
                                [0.31, 0.35, 25 * M, -4.7 * M]]
accessibilitySegmentSingle = [[0, 0.05, 15000000, 0], [0.05, 0.22, 125000000 / 17.0, 6500000 / 17.0],
                              [0.22, 0.275, 100000000 / 11, 0], [0.275, 0.31, 100000000 / 7.0, -10000000 / 7.0],
                              [0.31, M, M * M, 0]]


def accessibilityCost(x, numProducts):
    if numProducts == 1:
        return y(accessibilitySegmentSingle, x)
    else:
        return y(accessibilitySegmentMultiple, x)


def testCapacity():
    print( capacityAndAutomationChangeCost(1800, 2300, 3, 3))
# oc: old capacity, nc: new capacity, oa: old automation, na: new automation
def capacityAndAutomationChangeCost(oc, nc, oa, na):
    cost = 0
    # capacity changes
    if nc > oc:
        cost += 6 * (nc - oc)
    else:
        cost -= 0.65 * (oc - nc)

    cost += 4 * abs(oa - na) * nc
    return cost


"""
segments looks like this
[[l, u, m, c], [l, u, m, c]
"""


def y(segmentsLocal, x):
    for segment in segmentsLocal:
        l, u, m, c = segment
        if l <= x < u:
            return line(m, c, x)

def testYy():
    month = np.arange(0, 24, 1)
    print("month", month)
    segment = "low"
    cx, cy, ix, iy = segments[segment]["pos"]
    dx, dy = segments[segment]["driftPerYear"]
    print(ix, iy)
    print(dx, dy)
    segmentX = [[0, ix], [1, ix+dx/12]]
    segmentY = [[0, iy], [1, iy+dy/12]]
    print("point0", yy(segmentX, 2, "extrapolate", "extrapolate"))
    # x = list(map(lambda x: yy(segmentX, x, "extrapolate"), month))
    # y = list(map(lambda x: yy(segmentY, x, "extrapolate"), month))


# segments is a list of points. the function is assumed to be piecewise linear continous function
# between the points [[x1, y1], [x2, y2], ...]
# segments [[0, 0], [1, 1], ...]
def yy(segmentsLocal, x, defaultLeft=0, defaultRight=0):
    if x < segmentsLocal[0][0]:
        if defaultLeft == "extrapolate":
            x0, y0 = segmentsLocal[0]
            x1, y1 = segmentsLocal[0]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return defaultLeft
    if x == segmentsLocal[0][0]:
        return segmentsLocal[0][1]
    x1, y1 = segmentsLocal[0]
    x0, y0 = x1, y1
    for segment in segmentsLocal[1:]:
        x2, y2 = segment
        if x <= x2:
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        x0, y0 = x1, y1
        x1, y1 = x2, y2
    if defaultRight == "extrapolate":
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return defaultRight


def line(slope, c, x):
    return slope * x + c


def gen():
    ys = [.48, .49, .5, .52, .6, .69, .81, .99, 1.2, 1.48]
    xs = range(1, 11)
    points = list(zip(xs, ys))
    print(json.dumps(points))


def findPositionCost():
    low = segments["low"]["pos"][:2]
    high = segments["high"]["pos"][:2]
    dy = 2.5*math.sin(math.pi/4)
    dx = 2.5*math.cos(math.pi/4)
    x1, y1, x2, y2 = low[0] - dx, low[1] + dy, high[0] + dx, high[1] - dy
    d = dist(x1, y1, x2, y2)
    costPerD = 9 / d
    print(dx, dy)
    # (0.732 19.268)
    print(x1, y1)

    # 9.268 10.732
    print(x2, y2)

    print(costPerD)
    # 0.7456


if __name__ == '__main__':
    main()
