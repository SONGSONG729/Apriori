from numpy import *
from time import sleep
from votesmart import votesmart

def loadDataSet():
    '''
    加载数据集
    :return:
    '''
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    '''
    创建集合 C1.即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
    :param dataSet:
    :return:
    '''
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                # 遍历所有的元素，如果不在 C1 出现过，就append
                C1.append([item])
    C1.sort()
    # 对C1中每个项构建一个不变集合frozenset表示冻结的set集合，元素无改变；可以把它当字典的key来使用
    return list(map(frozenset, C1))

def scanD(D, Ck, minSupport):
    '''
    计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于最小支持度（minSupport）的数据
    :param D:
    :param Ck:
    :param minSupport:
    :return:
    '''
    # ssCnt 临时存放选数据集 Ck 的频率. 例如: a->10, b->5, c->8
    ssCnt = {}
    for tid in D:
        for can in Ck:
            # 测试是否can中的每一个元素都在tid中
            if can.issubset(tid):
                #if not ssCnt.has_key(can):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))  # 数据集 D 的数量
    retList = []
    supportData = {}
    for key in ssCnt:
        # 支持度 = 候选项（key）出现的次数 / 所有数据集的数量
        support = ssCnt[key] / numItems
        if support >= minSupport:
            # 在 retList 的首位插入元素，只存储支持度满足频繁项集的值
            retList.insert(0, key)
        # 存储所有的候选项（key）和对应的支持度（support）
        supportData[key] = support
    return retList, supportData

# 输入频繁项集列表 Lk 与返回的元素个数 k，然后输出所有可能的候选项集 Ck
def aprioriGen(Lk, k):
    """aprioriGen（输入频繁项集列表 Lk 与返回的元素个数 k，然后输出候选项集 Ck。
       例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
       仅需要计算一次，不需要将所有的结果计算出来，然后进行去重操作
       这是一个更高效的算法）

    Args:
        Lk 频繁项集列表
        k 返回的项集元素个数（若元素的前 k-2 相同，就进行合并）
    Returns:
        retList 元素两两合并的数据集
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):  # 前k-2项相同时，将两个集合合并
            L1 = list(Lk[i])[: k - 2]
            L2 = list(Lk[j])[: k - 2]
            L1.sort()
            L2.sort()
            # 第一次 L1,L2 为空，元素直接进行合并，返回元素两两合并的数据集
            if L1 == L2:
                # set union
                # print 'union=', Lk[i] | Lk[j], Lk[i], Lk[j]
                retList.append(Lk[i] | Lk[j])  # 集合合并操作
    return retList

# 找出数据集 dataSet 中支持度 >= 最小支持度的候选项集以及它们的支持度。即我们的频繁项集。
def apriori(dataSet, minSupport=0.5):
    '''
    首先构建集合 C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求。
    那么满足最小支持度要求的项集构成集合 L1。然后 L1 中的元素相互组合成 C2，C2 再进一步过滤变成 L2，
    然后以此类推，知道 CN 的长度为 0 时结束，即可找出所有频繁项集的支持度。
    :param dataSet: 原始数据集
    :param minSupport: 支持度的阈值
    :return:
        L 频繁项集的全集
        supportData 所有元素和支持度的全集
    '''
    # C1 即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
    C1 = createC1(dataSet)
    # 对每一行进行 set 转换，然后存放到集合中
    D = list(map(set, dataSet))
    # 计算候选数据集 C1 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
    L1, supportData = scanD(D, C1, minSupport)
    # L 加了一层 list, L 一共 2 层 list
    L = [L1]
    k = 2
    # 判断 L 的第 k-2 项的数据长度是否 > 0。
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        # 计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
        Lk, supK = scanD(D, Ck, minSupport)
        # 保存所有候选项集的支持度，如果字典没有，就追加元素，如果有，就更新元素
        supportData.update(supK)
        if len(Lk) == 0:
            break
        # Lk 表示满足频繁子项的集合，L 元素在增加
        L.append(Lk)
        k += 1
    return L, supportData

# 计算可信度（confidence）
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    '''
    对两个元素的频繁项，计算可信度，例如： {1,2}/{1} 或者 {1,2}/{2} 看是否满足条件
    :param freqSet: 频繁项集中的元素，例如: frozenset([1, 3])
    :param H: 频繁项集中的元素的集合，例如: [frozenset([1]), frozenset([3])]
    :param supportData: 所有元素的支持度的字典
    :param brl: 关联规则列表的空数组
    :param minConf: 最小可信度
    :return:
        prunedH 记录 可信度大于阈值的集合
    '''
    # 记录可信度大于最小可信度（minConf）的集合
    prunedH = []
    for conseq in H:
        '''
        假设 freqSet = frozenset([1, 3]), H = [frozenset([1]), frozenset([3])]，
        那么现在需要求出frozenset([1]) -> frozenset([3])的可信度和frozenset([3]) -> frozenset([1])的可信度
        '''
        '''
        支持度定义: a -> b = support(a | b) / support(a). 
        假设  freqSet = frozenset([1, 3]), conseq = [frozenset([1])]，
        那么 frozenset([1]) 至 frozenset([3]) 的可信度为 = support(a | b) / support(a) 
        = supportData[freqSet]/supportData[freqSet-conseq] = 
        supportData[frozenset([1, 3])] / supportData[frozenset([1])]
        '''
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            # 只要买了 freqSet-conseq 集合，一定会买 conseq 集合（freqSet-conseq 集合和 conseq集合 是全集）
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

# 递归计算频繁项集的规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    '''

    :param freqSet: 频繁项集中的元素，例如: frozenset([2, 3, 5])
    :param H: 频繁项集中的元素的集合，例如: [frozenset([2]), frozenset([3]), frozenset([5])]
    :param supportData: 所有元素的支持度的字典
    :param brl: 关联规则列表的数组
    :param minConf: 最小可信度
    :return:
    '''
    '''
    # H[0] 是 freqSet 的元素组合的第一个元素，并且 H 中所有元素的长度都一样，长度由 aprioriGen(H, m+1) 这里的 m + 1 来控制
    # 该函数递归时，H[0] 的长度从 1 开始增长 1 2 3 ...
    # 假设 freqSet = frozenset([2, 3, 5]), H = [frozenset([2]), frozenset([3]), frozenset([5])]
    # 那么 m = len(H[0]) 的递归的值依次为 1 2
    # 在 m = 2 时, 跳出该递归。假设再递归一次，那么 H[0] = frozenset([2, 3, 5])，freqSet = frozenset([2, 3, 5]) ，
    # 没必要再计算 freqSet 与 H[0] 的关联规则了。
    '''
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        '''
        # print 'freqSet******************', len(freqSet), m + 1, freqSet, H, H[0]
        # 生成 m+1 个长度的所有可能的 H 中的组合，假设 H = [frozenset([2]), frozenset([3]), frozenset([5])]
        # 第一次递归调用时生成 [frozenset([2, 3]), frozenset([2, 5]), frozenset([3, 5])]
        # 第二次 。。。没有第二次，递归条件判断时已经退出了
        '''
        Hmp1 = aprioriGen(H, m + 1)
        # 返回可信度大于最小可信度的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        # 计算可信度后，还有数据大于最小可信度的话，那么继续递归调用，否则跳出递归
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

# 主函数，调用前两个函数，生成关联规则
def generateRules(L, supportData, minConf=0.7):
    '''

    :param L: 频繁项集列表
    :param supportData: 频繁项集支持度的字典
    :param minConf: 最小置信度
    :return:
        bigRuleList 可信度规则列表（关于 (A->B+置信度) 3个字段的组合）
    '''
    bigRuleList = []
    # 假设 L = [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])], [frozenset([1, 3]), frozenset([2, 5]), frozenset([2, 3]), frozenset([3, 5])], [frozenset([2, 3, 5])]]
    for i in range(1, len(L)):
        # 获取频繁项集中每个组合的所有元素
        for freqSet in L[i]:
            # 假设：freqSet= frozenset([1, 3]), H1=[frozenset([1]), frozenset([3])]
            # 组合总的元素并遍历子元素，并转化为 frozenset 集合，再存放到 list 列表中
            H1 = [frozenset([item]) for item in freqSet]
            # 2 个的组合，走 else, 2 个以上的组合，走 if
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

# 收集美国国会议案中actionId的函数
def getActionIds():
    votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
    actionIdList = []
    billTitleList = []
    fr = open('recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum)  # api call
            for action in billDetail.actions:
                if action.level == 'House' and (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print("problem getting bill %d" % billNum)
        sleep(1)  # delay to be polite
    return actionIdList, billTitleList

# 基于投票数据的事务列表填充函数
def getTransList(actionIdList, billTitleList):
    itemMeaning = ['Republican', 'Democratic']
    for billTitle in billTitleList:  # 填充itemMeaning函数
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName):
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except:
            print("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning


def main():
    # #程序清单11-1
    # # 导入数据集
    # dataSet = loadDataSet()
    # print(dataSet)
    # # 构建第一个选项集集合C1
    # C1 = createC1(dataSet)
    # print(C1)
    # # 构建集合表示的数据集D
    # D = list(map(set, dataSet))
    # print(D)
    # # 去掉不满足最小支持度的项集
    # L1, suppData0 = scanD(D, C1, 0.5)
    # print(L1)

    # # 测试程序清单11-2
    # dataSet = loadDataSet()
    # L, suppData = apriori(dataSet)
    # print('L:', L)
    # print('L[0]:', L[0])
    # print('L[1]:', L[1])
    # print('L[2]:', L[2])

    # # 测试程序清单11-3
    # dataSet = loadDataSet()
    # # 生成一个最小支持度为0.5的频繁项集的集合
    # L, suppData = apriori(dataSet, minSupport=0.5)
    # rules = generateRules(L, suppData, minConf=0.7)
    # print(rules)

    # 测试程序清单11-4
    # actionIdList, billTitles = getActionIds()

    # 测试程序清单11-5
    actionIdList, billTitles = getActionIds()
    # transDict, itemMeaning = getTransList(actionIdList[:2], billTitles[:2])
    # print(transDict.keys()[6])
    # for item in transDict[' ']:
    #     print(itemMeaning[item])

    # # 毒蘑菇相似特征
    # dataSet = [line.split() for line in open("mushroom.dat").readlines()]
    # L, supportData = apriori(dataSet, minSupport=0.3)
    # print('L[1]:')
    # for item in L[1]:
    #     if item.intersection('2'):
    #         print(item)
    # print('L[2]:')
    # for item in L[2]:
    #     if item.intersection('2'):
    #         print(item)
    #


if __name__ == "__main__":
    main()