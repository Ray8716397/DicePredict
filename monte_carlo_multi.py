import random
import multiprocessing
import pickle
import os

history = [15, 11, 10, 13, 10, 12, 8, 9, 13, 8, 8, 13, 12, 9, 9, 11, 8, 3, 17, 11, 13, 11, 12, 11, 17, 7, 16, 10, 16,
           14, 10, 13, 9, 8, 12, 10, 4, 8, 11, 10, 7, 14, 8, 12, 11, 8, 13, 13, 9]
new_data = [12, 6, 9, 13, 12, 7]
data = history + new_data


# odd_history += [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, ]
# small_history += [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]


def roll_dice():
    return random.randint(1, 6)


def predict_roll(history, num_trials=2000):
    counts = [0] * 16
    for i in range(num_trials):
        roll1 = roll_dice()
        roll2 = roll_dice()
        roll3 = roll_dice()
        roll = roll1 + roll2 + roll3
        if roll in history:
            counts[roll - 4] += 1
    total_counts = sum(counts)
    probabilities = [count / total_counts for count in counts]
    predicted_roll = random.choices(range(4, 20), probabilities)[0]
    return predicted_roll


def predict_proc(pidx, history_data):
    random.seed(random.seed(random.randint(1, 100)))
    length = len(history_data)
    seed = 1
    max_epoch = 10
    count_poch = 0
    while count_poch < max_epoch:
        count_odd = 0
        count_small = 0
        for i, d in enumerate(history_data[:-1]):
            res = predict_roll([x for x in history_data[:i + 1]])
            r_odd = res % 2 == 1
            r_small = res < 11
            t_odd = history_data[i + 1] % 2 == 1
            t_small = history_data[i + 1] < 11
            if r_odd == t_odd:
                count_odd += 1
            if r_small == t_small:
                count_small += 1
        odd_acc = count_odd / length
        small_acc = count_small / length

        if small_acc > 0.7 and odd_acc > 0.7:
            break
        else:
            seed += 1
            count_poch += 1

    # print(f"---------------------------------------------------------------------------------------------------")
    res = predict_roll([x for x in history_data])
    # print(f"proc{pidx} res: \033[1;31m {res} \033[0m")
    # print(f"odd_acc: {odd_acc}, odd%: {sum([1 if i % 2 == 1 else 0 for i in history_data]) / len(history_data)}")
    # print(f"small_acc: {small_acc}, small%: {sum([1 if i < 11 else 0 for i in history_data]) / len(history_data)}")
    return [res, odd_acc, small_acc]


def predict_multi_proc(history_data):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as proc_pool:
        manager = multiprocessing.Manager()
        plock = manager.Lock()
        results = [proc_pool.apply_async(predict_proc, (x, data)) for x in range(4)]

        max_odd = 0.0
        max_small = 0.0
        max_oi = 0
        max_si = 0
        for idx, r in enumerate(results):
            s, oacc, sacc = r.get()
            if max_odd <= oacc:
                max_odd = oacc
                max_oi = idx
            if max_small <= sacc:
                max_small = sacc
                max_si = idx
        else:
            max_odd_r = results[max_oi].get()
            max_small_r = results[max_si].get()

            return max_odd_r[0] % 2 == 1, max_small_r[0] < 11


if __name__ == "__main__":

    # print(f"!!!!!!!!!!!max odd!!!!!!!!!!!!!!!!!!!!!!")
    # print(max_odd_r)
    #
    # print(f"!!!!!!!!!!!max small!!!!!!!!!!!!!!!!!!!!!!")
    # print(max_small_r)

    # print(f"!!!!!!!!!!!predict!!!!!!!!!!!!!!!!!!!!!")
    # print(f"true odd% {odd_history.count(1) / len(odd_history)}")
    # print(f"true small%{small_history.count(1) / len(small_history)}")

    # 判断有没数据库，没有的话根据最后一个生成
    if not os.path.exists('db'):
        is_odd, is_small = predict_multi_proc(data[:-2])
        # 是否预测正确数据库
        database = {"his": [data[-2]],
                    'odd_acc_his': [is_odd == (data[-1] % 2 == 1)],
                    'small_acc_his': [is_small == (data[-1] < 11)],
                    'real_odd_prev': [],
                    'real_small_prev': [],
                    'p_odd_prev': [],
                    'p_small_prev': [],
                    'p_odd_acc_his': [],
                    'p_small_acc_his': [],
                    }
        with open("db", 'wb') as f:
            pickle.dump(database, f)

    else:
        database = pickle.load(open('db', 'rb'))
        database['odd_acc_his'].append(database['real_odd_prev'][-1] == (data[-1] % 2 == 1))
        database['small_acc_his'].append(database['real_small_prev'][-1] == (data[-1] < 11))

        database['p_odd_acc_his'].append(database['p_odd_prev'][-1] == (data[-1] % 2 == 1))
        database['p_small_acc_his'].append(database['p_small_prev'][-1] == (data[-1] < 11))
        database['his'].append(data[-1])
        print("----------------------------------------------------------")
        print(f"read odd acc: {database['odd_acc_his'].count(True) / len(database['odd_acc_his'])}")
        print(f"read small acc: {database['small_acc_his'].count(True) / len(database['small_acc_his'])}")
        print(f"predict odd acc: {database['p_odd_acc_his'].count(True) / len(database['p_odd_acc_his'])}")
        print(f"predict small acc: {database['p_small_acc_his'].count(True) / len(database['p_small_acc_his'])}")

    next_odd, next_small = predict_multi_proc(data)
    print(f"!!!!!!!!!!!last:{new_data} predict start!!!!!!!!!!!!!!!!!!!!!")
    predict_odd = next_odd if database['odd_acc_his'][-1] else (not next_odd)
    predict_small = next_small if database['small_acc_his'][-1] else (not next_small)

    print(f"predict_odd:\033[1;31m {'ODD' if predict_odd else 'EVEN'} \033[0m\npredict_small:\033[1;31m {'SMALL' if predict_small else 'BIG'}\033[0m")
    database['real_odd_prev'].append(next_odd)
    database['real_small_prev'].append(next_small)
    database['p_odd_prev'].append(predict_odd)
    database['p_small_prev'].append(predict_small)

    with open("db", 'wb') as f:
        pickle.dump(database, f)
    print(database)
