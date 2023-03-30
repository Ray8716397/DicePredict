# coding=utf-8
# @Time : 2023/3/30 下午3:12
# @File : gpt4_predict.py
import collections
import random

def analyze_data(historical_data):
    outcomes_count = collections.Counter(historical_data)
    outcomes_prob = {k: v/len(historical_data) for k, v in outcomes_count.items()}
    return outcomes_prob

def sicbo_strategy(historical_data):
    outcomes_prob = analyze_data(historical_data)
    sorted_prob = sorted(outcomes_prob.items(), key=lambda x: x[1], reverse=True)

    print("历史数据中投注类型的胜率：")
    for outcome, prob in sorted_prob:
        print(f"{outcome}: {prob:.2%}")

    most_likely_outcome = sorted_prob[0][0]
    print(f"\n根据历史数据，下一步建议的投注策略是：{most_likely_outcome}")

# historical_data = [8,6,14,10,11,15,8,9,10,10,10,18,8,6,14,15,9,12,11,8,8,10,9,11,12,10,14,8,6,13,8,13,15,10,9,12,11,9,11,13,11,11,9,11,12,17,9,11,14,14,10,14,8,12]
historical_data = [11,7,8,11,9,8,12,12,6,13,14,10,11,5,14,12]  # 模拟生成100局游戏的历史数据
sicbo_strategy(historical_data)
