import json
import math
import numpy as np

def main():
    for segment in segments:
        cx, cy = segments[segment]["pos"]
        x = np.outer(np.linspace(cx-3.54, cx+3.54, 30), np.ones(30))
        y = x.copy().T
        zipped = zip(np.reshape(x, -1), np.reshape(y, -1))

        def mapper(p):
            x, y = p
            return position(segment, x, y)
        z = np.reshape(np.fromiter(map(mapper, zipped), float), (30, 30))
    # a = np.linspace(0, 10, 10)
    # b = a.copy()
    # c = zip(a, b)
    print("hi")
        # z = position(segment, x, y)



segments = {
    "traditional": {
        "age": [[-1, 0, 0, 0.1], [0, 2, 0.45, 0.1], [2, 4, -.45, 1.9], [4, 11, 0, 0.1]],
        "price": [20, 30],
        "pos": [5, 15],
        "mtbf": [12000, 17000],
        "weights": [.47, .23, .21, .09]
    },
    "low": {
        "age": [[-1, 0, 0, 0.1], [0, 2, 0.02, 0.08], [2, 7, 0.176, -0.232], [7, 11, -0.176, 2.232]],
        "price": [15, 25],
        "pos": [1.7, 18.3],
        "mtbf": [12000, 17000],
        "weights": [.47, .23, .21, .09]

    },
    "high": {
        "age": [[0, 0.7, -4 / 25.0, 1], [.7, 2.4, -359 / 1000, 11393 / 10000], [2.4, 3.4, -1117 / 10000, 27289 / 50000],
                [3.4, 11, 0, 1.5 / 9]],
        "price": [30, 40],
        "pos": [8.9, 11.1],
        "mtbf": [20000, 25000],
        "weights": [.47, .23, .21, .09]

    },
    "performance": {
        "age": [[0, 1, 0.2, 0.8], [1, 3.5, -0.36, 1.36], [3.5, 11, 0, 0.1]],
        "price": [25, 35],
        "pos": [9.4, 16.0],
        "mtbf": [22000, 27000],
        "weights": [.47, .23, .21, .09]

    },
    "size": {
        "age": [[0, 1.5, 0.33, 0.5], [1.5, 4, -0.36, 1.54], [4, 11, 0, 0.1]],
        "price": [25, 35],
        "pos": [4.0, 10.6],
        "mtbf": [16000, 21000],
        "weights": [.47, .23, .21, .09]

    }
}


def age(p, x):
    return y(segments[p]["age"], x)


def price(p, x):
    l, u, = segments[p]["price"]
    return (x - u) / (l - u)


"""
decision variables
m:       related to MTBF. It is the lower end percentage
pc:      position constant: the decrease in score per unit distance inside the fine cut circle
         pc has to be less than 1/2.5 
awr:     awareness is a fraction between 0 and 1. it costs $ to buy awareness
acs:     accessibility is a fraction between 0 and 1. it costs $ to buy accesibility. 
         It is set per product, but it accumulates to the segment.
"""

m = 0.8
pc = 0.08
awr = 0.8
acs = .5


def mtbf(p, x):
    global m
    l, u = segments[p]["mtbf"]
    segmentsLocal = [[l-5000, 0], [l, m], [u, 1]]
    return yy(segmentsLocal, x)


def position(p, perf, sz):
    a, b = segments[p]["pos"]
    d = dist(a, b, perf, sz)
    if d > 2.5:
        slope = (2.5 * pc - 1) / 1.5
        c = 2.66 - 6.66 * m
        return line(slope, c, d)
    else:
        return 1 - pc * d


def score(p, agee, pricee, perf, sz, mtbff):
    ageScore = age(p, agee)
    priceScore = price(p, pricee)
    posScore = position(p, perf, sz)
    mtbfScore = mtbf(p, mtbff)
    w1, w2, w3, w4 = segments[p]["weights"]
    return w1 * ageScore + w2 * priceScore + w3 * posScore + w4 * mtbfScore


# the fixed cost only the RnDCost
def totalVariableCostExistingProduct(perf, sz, mtbfLocal, aut, overProduction):
    positionMaterialCost(perf, sz) \
    + mtbfMaterialCost(mtbfLocal) \
    + labourCost(aut, overProduction)


def dist(a, b, perf, sz):
    return np.sqrt((sz - b) * (sz - b) + (perf - a) * (perf - a))


positionCostLine = [[0, 11, 1500.0 / 2357.0, 1]]


def positionMaterialCost(perf, sz):
    d = dist(1, 1, perf, sz)
    return y(positionCostLine, d)


def mtbfMaterialCost(mtbfLocal):
    return mtbfLocal * 0.3 / 1000.0


def rndCost(aut):
    return 1000000 * rndTimeYears(aut)


labourBaseCost = 10.7 * 1.05


def labourCost(aut, overProduction):
    labor = 0.1 * (11 - aut) * labourBaseCost * (1 + 1.5 * overProduction) / (1 + overProduction)
    return labor


segmentsForAutToYears = [[1, 0.48], [2, 0.49], [3, 0.5], [4, 0.52], [5, 0.6], [6, 0.69], [7, 0.81], [8, 0.99], [9, 1.2],
                         [10, 1.48]]


def rndTimeYears(aut):
    return yy(segmentsForAutToYears, aut)


awarenessSegments = [[0, 0.02, 15000000, 0], [0.02, 0.05, 5000000 / 3.0, 800000 / 3.0],
                     [0.05, 0.4, 4000000, 150000], [0.4, 0.46, 25000000 / 3.0, -4750000 / 3.0],
                     [.46, 5, 18750000, -6375000], [5, 10000000, 100000000000, 0]]


def awarenessCost(x):
    return y(awarenessSegments, x)


M = 1000000
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


# segments is a list of points. the function is assumed to be piecewise linear continous function
# between the points [[x1, y1], [x2, y1], ...]
# segments [[0, 0], [1, 1], ...]
def yy(segmentsLocal, x):
    if x < segmentsLocal[0][0]:
        raise Exception("out of bound")
    if x == segmentsLocal[0][0]:
        return segmentsLocal[0][1]
    x1, y1 = segmentsLocal[0]
    for segment in segmentsLocal[1:]:
        x2, y2 = segment
        if x <= x2:
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        x1, y1 = x2, y2


def line(slope, c, x):
    return slope * x + c


def gen():
    ys = [.48, .49, .5, .52, .6, .69, .81, .99, 1.2, 1.48]
    xs = range(1, 11)
    points = list(zip(xs, ys))
    print(json.dumps(points))


if __name__ == '__main__':
    main()
