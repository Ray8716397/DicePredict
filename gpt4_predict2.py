# coding=utf-8
# @Time : 2023/3/30 下午3:12
# @File : gpt4_predict.py
import random
from collections import Counter

def roll_dice():
    return random.randint(1, 6)

def simulate_sicbo(rounds):
    results = []
    for _ in range(rounds):
        dice_sum = roll_dice() + roll_dice() + roll_dice()
        if dice_sum >= 4 and dice_sum <= 10:
            results.append('小')
        elif dice_sum >= 11 and dice_sum <= 17:
            results.append('大')
    return results

def get_next_bet_strategy(history, simulations=10000):
    counter = Counter(history)
    win_ratios = {bet_type: count / len(history) for bet_type, count in counter.items()}

    simulated_results = simulate_sicbo(simulations)
    sim_counter = Counter(simulated_results)
    sim_ratios = {bet_type: count / simulations for bet_type, count in sim_counter.items()}

    strategy = max(sim_ratios, key=lambda x: sim_ratios[x] * win_ratios.get(x, 0))
    return strategy

# 假设我们有一个包含历史数据的列表
history = ['大', '小', '大', '小', '小', '大', '大', '大', '小', '大']

next_bet = get_next_bet_strategy(history)
print("下一步投注策略:", next_bet)

