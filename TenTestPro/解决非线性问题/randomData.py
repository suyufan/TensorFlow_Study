import random
# 产生随机数种子
random.seed()
xData = [int(random.random() * 41 + 60), int(random.random() * 41 + 60), int(random.random() * 41 + 60)]
xAll = xData[0] * 0.6 + xData[1] * 0.3 + xData[2] * 0.1

if xAll >= 95:
    yTrainData = 1
else:
    yTrainData = 0

print("xData: %s" % xData)
print("yTrainData: %s" % yTrainData)