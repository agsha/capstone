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
        "sz": 14.6,
        "mtbf": 19*K,
        "ageYears": 2.4,
        "awareness": .77,
        "accessibility":.71,
        "aut": 4,
        "capacity": 1900*K,
        "score": {
            "traditional": 0.20,
        },
        "inventory": 0
    }, "ebb": {
        "price": 28,
        "perf": 2.5,
        "sz": 17.5,
        "mtbf": 14*K,
        "ageYears": 3.1,
        "awareness": .75,
        "accessibility":.32,
        "aut": 5,
        "capacity": 1700*K,
        "score": {
            "traditional": .02,
            "low":1/6
        },
        "inventory": 1623*K
    }, "echo": {
        "price": 34,
        "perf": 8.9,
        "sz": 11.1,
        "mtbf": 23*K,
        "ageYears": 1.5,
        "awareness": .73,
        "accessibility":.65,
        "aut": 6,
        "capacity": 900,
        "score": {
            "high": .23,
        },
        "inventory": 92*K
    }, "edge": {
        "price": 34.5,
        "perf": 9.5,
        "sz": 15.5,
        "mtbf": 25*K,
        "ageYears": 2.2,
        "awareness": .71,
        "accessibility":.53,
        "aut": 3,
        "capacity": 700*K,
        "score": {
            "performance": .136,
        },
        "inventory": 172*K
    }, "egg": {
        "price": 31,
        "perf": 4.1,
        "sz": 11,
        "mtbf": 19*K,
        "ageYears": 2.2,
        "awareness": .71,
        "accessibility":.6,
        "aut": 3,
        "capacity": 701*K,
        "score": {
            "size": .128,
        },
        "inventory": 31*K
    }
}
def main():
    testSalesEat()

def simulate2():
    segment = "low"
    seg = segments[segment]
    product = products["ebb"]
    estimatedSalesPerYear = seg["demand"]
    baseScoreEmpiricalDenominator = score(seg, product["ageYears"], product["price"], product["perf"], product["sz"], product["mtbf"], product["awareness"], product["accessibility"])
    baseScoreReal = product["score"][segment]
    aut = product["aut"]
    capacity = product["capacity"]
    inventory = product["inventory"]
    oldPrice = product["price"]
    ebbPerf, ebbSz, ebbMtbf = product["perf"], product["sz"], product["mtbf"]

    perff = list(np.linspace(ebbPerf-10, ebbPerf, 30)) + list(np.linspace(ebbPerf, ebbPerf+1, 30))[1:]
    szz = list(np.linspace(ebbSz-5, ebbSz, 20)) + list(np.linspace(ebbSz, ebbSz+5, 20))[1:]
    pricee = list(np.linspace(seg["price"][0], product["price"], 20)) + list(np.linspace(product["price"], seg["price"][1], 20))[1:]
    ll = []
    dummy = 0
    for perf in perff:
        if abs(perf - ebbPerf) > .01:
            continue
        for sz in szz:
            if perf <= 0 or sz <= 0:
                continue
            if abs(sz - ebbSz) > .01:
                continue
            for mtbf in range(seg["mtbf"][0], seg["mtbf"][1] + 1000, 1000):
                if abs(mtbf - ebbMtbf) > 1500:
                    continue
                d = dist(ebbPerf, ebbSz, perf, sz)
                rndYear = rndTimeYears(aut, d, abs(ebbMtbf - mtbf), 3)
                if rndYear < 1:
                    dummy += 1
                    for price in pricee:
                        if abs(price - product["price"]) > .01:
                            continue
                        estimatedRevenue = 0
                        estimatedCost = 0
                        for month in range(1, 13):
                            estimatedCostDelta, estimatedRevenueDelta, inventory = sales(baseScoreEmpiricalDenominator, baseScoreReal,
                                                                              ebbMtbf, ebbPerf, ebbSz, estimatedSalesPerYear, month,
                                                                              mtbf, perf, sz, price, rndYear, seg, aut, inventory, capacity)
                            estimatedCost += estimatedCostDelta
                            estimatedRevenue += estimatedRevenueDelta

                        estimatedCost += rndYear*M
                        profitMargin = 1000
                        if estimatedCost != 0:
                            profitMargin = estimatedRevenue/estimatedCost
                        revenueGrowth = estimatedRevenue/(oldPrice *estimatedSalesPerYear*baseScoreReal)
                        # print("c:{} s:{}, s/c:{}".format(estimatedCost, estimatedRevenue, estimatedRevenue/estimatedCost))
                        # if totalScore > 1.4:
                        # ll.append(revenueGrowth)
                        if  revenueGrowth>1.39:
                            # ll.append(profitMargin)
                            ll.append((perf, sz, rndYear*12))
                            print("perf:{} sz:{} mtbf:{} price:{} rndTimeMonths:{} profitMargin:{} revenue:{} sales {} growthInRev:{}".format( perf, sz, mtbf, price, rndYear*12, profitMargin, estimatedRevenue, estimatedRevenue/price, revenueGrowth))
    return ll

def testSalesEat():
    segment = "low"
    seg = segments[segment]
    product = products["ebb"]
    baseScoreEmpiricalDenominator = score(seg, product["ageYears"], product["price"], product["perf"], product["sz"], product["mtbf"], product["awareness"], product["accessibility"])
    baseScoreReal = product["score"][segment]
    estimatedSalesPerYear = seg["demand"]
    capacity = product["capacity"]
    inventory = product["inventory"]


    aut = product["aut"]
    ebbPerf, ebbSz, ebbMtbf, ebbPrice = product["perf"], product["sz"], product["mtbf"], product["price"]
    aPerf, aSz, aMtbf, aPrice = 2.2, 17.8, 17000,
    sale = 0
    s = 0
    c = 0
    print("baseMul:{} baseSales:{} BaseadjSales:{} baseScore:{}".format(baseScoreReal, estimatedSalesPerYear, baseScoreReal*estimatedSalesPerYear, baseScoreEmpiricalDenominator))
    print("=============")
    d = dist(ebbPerf, ebbSz, aPerf, aSz)
    rndYear = rndTimeYears(aut, d, abs(ebbMtbf - aMtbf), aut)

    for month in range(1, 13):

        estimatedCost, estimatedRevenue, inventory = sales(baseScoreEmpiricalDenominator, baseScoreReal, ebbMtbf,
                                                ebbPerf, ebbSz, estimatedSalesPerYear, month, aMtbf,
                                                aPerf, aSz, aPrice, rndYear, seg, aut, inventory, capacity)
        s+=estimatedRevenue
        c += estimatedCost
        sale += estimatedRevenue/aPrice
    c+=rndCost(rndYear)
    profitMargin = 1000
    if c != 0:
        profitMargin = s/c
    growthInSales = s/(ebbPrice *estimatedSalesPerYear*baseScoreReal)
    print("Totalrevenue:{} Totalcost:{} Totalsales:{} profitMargin:{} growthInSales:{}, rndYear:{} ".format(s, c, sale, profitMargin, growthInSales, rndYear))


def sales(baseScoreEmpiricalDenominator, baseScoreReal, ebbMtbf, ebbPerf, ebbSz,
          estimatedSalesPerYear, month, mtbf, perf, sz, price, rndYear, seg, aut, inventory, capacity):
    adjustmentForOtherTeams = .85
    seg2 = segmentState(month, seg)
    age = currentAgeYears(month, rndYear, 4.6)
    aware = awareness(month, rndYear)
    tperf, tsz, tmtbf = ebbPerf, ebbSz, ebbMtbf
    if month >= rndYear * 12:
        tperf, tsz, tmtbf = perf, sz, mtbf
    sc = score(seg2, age, price, tperf, tsz, tmtbf, aware, .5)
    baseScoreEmpirical = sc / baseScoreEmpiricalDenominator
    scoreMultiplier = baseScoreEmpirical * baseScoreReal / (1 + baseScoreEmpirical * baseScoreReal - baseScoreReal)
    estimatedSalesAdjMonth = adjustmentForOtherTeams * scoreMultiplier * estimatedSalesPerYear * (seg2["growth"] ** int((month + 11) / 12)) / 12
    estimatedRevenue = estimatedSalesAdjMonth * price
    inventoryUsed = min(estimatedSalesAdjMonth, inventory)
    estimatedProductionPerMonth = estimatedSalesAdjMonth - inventoryUsed
    capacityPerMonth = capacity/12
    overProduction = max(0, (estimatedProductionPerMonth - capacityPerMonth)/capacityPerMonth)
    cost = totalVariableCostExistingProduct(tperf, tsz, tmtbf, aut, overProduction, month)

    estimatedCost = max(0, estimatedProductionPerMonth) * cost
    print("month:{}, mul: {}, adjSales:{} salesLessInventory:{} overProduction: {}, varCost:{} totalCost:{} score:{}".format(month,        scoreMultiplier, estimatedSalesAdjMonth*12, estimatedProductionPerMonth, overProduction, cost, estimatedCost, sc))
    return estimatedCost, estimatedRevenue, inventory - inventoryUsed


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
        "price": [19.5, 29.5],
        "pos": [5.7, 14.3, 5.7, 14.3],
        "mtbf": [14000, 19000],
        "weights": [.47, .23, .21, .09],
        "growth": 1.092,
        "driftPerYear": [.7, -.7],
        "demand": 8067 * K
    },
    "low": {
        "age": [[-1, 0, 0, 0.1], [0, 2, 0.02, 0.08], [2, 7, 0.176, -0.232], [7, 11, -0.176, 2.232]],
        "price": [14.5, 24.5],
        "pos": [3, 17, 2.2, 17.8],
        "mtbf": [12000, 17000],
        "weights": [.24, .53, .16, .07],
        "growth": 1.117,
        "driftPerYear": [.5, -.5],
        "demand": 10009*K

    },
    "high": {
        "age": [[0, 0.7, -4 / 25.0, 1], [.7, 2.4, -359 / 1000, 11393 / 10000], [2.4, 3.4, -1117 / 10000, 27289 / 50000],
                [3.4, 11, 0, 1.5 / 9]],
        "price": [30, 40],
        "pos": [7.5, 12.5, 8.9, 11.1],
        "mtbf": [20000, 25000],
        "weights": [.29, .09, .43, .19],
        "growth": 1.162,
        "driftPerYear": [.9, -.9],
        "demand": 2554 * K,
    },
    "performance": {
        "age": [[0, 1, 0.2, 0.8], [1, 3.5, -0.36, 1.36], [3.5, 11, 0, 0.1]],
        "price": [25, 35],
        "pos": [8.0, 17.0, 9.4, 16.0],
        "mtbf": [22000, 27000],
        "weights": [.09, .19, .29, .43],
        "growth": 1.198,
        "driftPerYear": [1, -.7],
        "demand": 1915 * K

    },
    "size": {
        "age": [[0, 1.5, 0.33, 0.5], [1.5, 4, -0.36, 1.54], [4, 11, 0, 0.1]],
        "price": [25, 35],
        "pos": [3.0, 12.0, 4.0, 10.6],
        "mtbf": [16000, 21000],
        "weights": [.29, .09, .43, .19],
        "growth": 1.183,
        "driftPerYear": [.7, -1.0],
        "demand": 1984 * K
    }
}


# tested
def ageScore(segment, x):
    return y(segment["age"], x)


# tested
priceSegment = [[0, 0], [5, 1], [5.251798561151079, 0.9952153110047847], [7.158273381294964, 0.6100478468899522],
                [10.575539568345324, 0.4043062200956938], [15.251798561151078, 0.3133971291866029],
                [20, 0]]


def testPriceScore():
    print(priceScore(segments["low"], 20))

def priceScore(segment, x):
    x = x - (segment["price"][0] - 5)
    return yy(priceSegment, x, defaultLeft=0, defaultRight=0)


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
    print("raw scores age:{} price:{} pos:{}, mtbf:{}".format(ageS, priceS, positionS, mtbfS))

    if ageS <= 0 or priceS <= 0 or positionS <= 0 or mtbfS <= 0:
        return 0
    w1, w2, w3, w4 = segment["weights"]
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

def gimp2segments():
    ox, oy = 174, 631
    pixelsPerX = (592 - ox)
    pixelsPerY = (oy-492)/5
    #price range 0, 5 is obvious
    #price range 5 - 15 is the following
    points = [[590, 485], [429, 432], [343, 337], [305, 207]]
    #[p[0]/pixelsPerX, (oy-p[1]])/pixelsPerY]

    points = [[(oy-p[1])/pixelsPerY, (p[0]-ox)/pixelsPerX] for p in points]
    print(points)



    pass

if __name__ == '__main__':
    main()
