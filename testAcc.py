import numpy as np
import pandas as pd
#path = './MAVI_clean_eng/gt_cropped_eng_1.txt'

def read_GT(path):
    #print('Reading GT...')
    df = pd.read_csv(path, sep='\t')
    df.columns = ['path', 'word']
    return df['path'], df['word']

def check(tests, gts):
    trueVal, falseVal = 0, 0
    for i in range(len(tests)):
        if tests[i] != gts[i].lower():
            falseVal += 1
            print('False: {}, True: {}'.format(tests[i], gts[i]))

        else:
            trueVal += 1
    return trueVal, falseVal

def accuracy():
    #print('Calculating accuracy...')
    #GT = read_GT('./MAVI_clean_eng/gt_cropped_eng_1.txt')
    GT = read_GT('./mavi_hindi/gt_1.txt')
    Out = read_GT('./output_FP32_Hindi.txt')
    trueVal, falseVal = check(Out[1], GT[1])
    total = trueVal + falseVal
    acc = trueVal/total
    print('True: {}, False: {}'.format(trueVal, falseVal))
    print('Accuracy: {}'.format(acc))
    print('Total: {}'.format(total))


if __name__ == '__main__':
    accuracy()