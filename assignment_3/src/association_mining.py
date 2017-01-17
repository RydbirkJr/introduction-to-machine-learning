from import_data import *
from writeapriorifile import *
import numpy as np
import re


data = Data()
data.binarize_in_you_face()

# For test if it works
# data1 = Data()
# for row in range(10):
#     print ''
#     for elem in range(data1.M):
#         print data1.X[row, elem], data.X[row,elem]

medians = np.median(np.matrix(Data().X), 0)

for i in range(data.M):
    print i, data.attribute_names[i], medians[0, i]

#WriteAprioriFile(data.X, filename='apriorifile.txt')
#exit(0)
filename = 'apriorifile.txt'
minSup = 10
minConf = 10
maxRule = 10

#apriori.exe -f"," -s10 -v"[Sup. %S]" apriorifile.txt apriori_temp1.txt

#apriori.exe -tr -f"," -n10 -c10 -s10 -v"[Conf. %C,Sup. %S]" apriorifile.txt apriori_temp2.txt

# Run Apriori Algorithm
# print('Mining for frequent itemsets by the Apriori algorithm')
# status1 = call('apriori.exe -f"," -s{0} -v"[Sup. %S]" {1} apriori_temp1.txt'.format(minSup, filename))
# if status1 != 0:
#     print(
#     'An error occured while calling apriori, a likely cause is that minSup was set to high such that no frequent itemsets were generated or spaces are included in the path to the apriori files.')
#     exit()
# if minConf > 0:
#     print('Mining for associations by the Apriori algorithm')
#     status2 = call(
#         'apriori.exe -tr -f"," -n{0} -c{1} -s{2} -v"[Conf. %C,Sup. %S]" {3} apriori_temp2.txt'.format(maxRule,
#                                                                                                                minConf,
#                                                                                                                minSup,
#                                                                                                                filename))
#     if status2 != 0:
#         print('An error occured while calling apriori')
#         exit()
# print('Apriori analysis done, extracting results')

# Extract information from stored files apriori_temp1.txt and apriori_temp2.txt
f = open('apriori_temp1.txt', 'r')
lines = f.readlines()
f.close()
# Extract Frequent Itemsets
FrequentItemsets = [''] * len(lines)
sup = np.zeros((len(lines), 1))
for i, line in enumerate(lines):
    FrequentItemsets[i] = line[0:-1]
    gg = re.findall(' [-+]?\d*.\d+|\d+]', line)
    sup[i] = gg[-1][1:]

#os.remove('apriori_temp1.txt')

# Read the file
f = open('apriori_temp2.txt', 'r')
lines = f.readlines()
f.close()
# Extract Association rules
AssocRules = [''] * len(lines)
conf = np.zeros((len(lines), 1))
for i, line in enumerate(lines):
    AssocRules[i] = line[0:-1]
    gg = re.findall(' [-+]?\d*\.\d+|\d+,', line)
    #print gg
    conf[i] = gg[0][0:-1]
    #print 'conf[i]:', conf[i], 'Line: ', line

#os.remove('apriori_temp2.txt')

# sort (FrequentItemsets by support value, AssocRules by confidence value)
AssocRulesSorted = [AssocRules[item] for item in np.argsort(conf, axis=0).ravel()]
AssocRulesSorted.reverse()
FrequentItemsetsSorted = [FrequentItemsets[item] for item in np.argsort(sup, axis=0).ravel()]
FrequentItemsetsSorted.reverse()

# Print the results
import time;

time.sleep(.5)
print('\n')
print('RESULTS:\n')
print('Frequent itemsets:')
for i, item in enumerate(FrequentItemsetsSorted):
    print('Item: {0}'.format(item))
print('\n')
print('Association rules:')
for i, item in enumerate(AssocRulesSorted):
    print('Rule: {0}'.format(item))
