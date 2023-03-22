import random
import multiprocessing


def roll_dice():
    return random.randint(1, 6)


def predict_roll(history, num_trials=1000):
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


def predict_proc(pidx):
    random.seed(random.seed(random.randint(1, 100)))
    data = [13, 11, 12, 8, 8, 9, 11, 8, 13, 7, 15, 6, 12, 10, 11, 12, 7, 8, 12, 16, 11, 12, 9, 16, 15, 13, 9, 16, 14,
            13, 7, 10, 8, 11, 5, 11, 10, 9, 11, 12, 11, 13, 9, 18, 6, 14, 10, 5, 9, 9, 13, 15, 12, 5, 6, 5, 9, 8, 8, 7, 11, 13, 12, 15, 5, 14, 11, 5, 10, 13, 11, 13]
    length = len(data)
    seed = 1
    max_epoch = 10
    count_poch = 0
    while count_poch < max_epoch:
        count_odd = 0
        count_small = 0
        for i, d in enumerate(data[:-1]):
            res = predict_roll([x for x in data[:i + 1]])
            r_odd = res % 2 == 1
            r_small = res < 11
            t_odd = data[i + 1] % 2 == 1
            t_small = data[i + 1] < 11
            if r_odd == t_odd:
                count_odd += 1
            if r_small == t_small:
                count_small += 1
        odd_acc = count_odd / length
        small_acc = count_small / length

        if small_acc > 0.55 and odd_acc > 0.55:
            break
        else:
            seed += 1
            count_poch += 1

    print(f"---------------------------------------------------------------------------------------------------")
    res = predict_roll([x for x in data])
    print(f"proc{pidx} res: \033[1;31m {res} \033[0m")
    print(f"odd_acc: {odd_acc}, odd%: {sum([1 if i % 2 == 1 else 0 for i in data]) / len(data)}")
    print(f"small_acc: {small_acc}, small%: {sum([1 if i < 11 else 0 for i in data]) / len(data)}")
    return [res, odd_acc, small_acc]


with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as proc_pool:
    manager = multiprocessing.Manager()
    plock = manager.Lock()
    results = [proc_pool.apply_async(predict_proc, (x,)) for x in range(4)]

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
        print(f"!!!!!!!!!!!max odd!!!!!!!!!!!!!!!!!!!!!!")
        print(results[max_oi].get())

        print(f"!!!!!!!!!!!max small!!!!!!!!!!!!!!!!!!!!!!")
        print(results[max_si].get())
