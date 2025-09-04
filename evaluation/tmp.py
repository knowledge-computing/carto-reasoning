import polars as pl
import re
import statistics
import copy
import numpy as np
from scipy.stats import mstats


regex_pattern = r"\d+\.?\d*"

pl_data = pl.read_json('/home/yaoyi/pyo00005/p2/carto-reasoning/questions/response_full_d10.json').filter(
    pl.col('answer_type') == 'distance'
)

def winsorized_bootstrap(data, proportion=0.10, n_boot=100, bool_out=False):
    means = []
    # data = data * 4
    for _ in range(n_boot):
        # sample = random.choice(data)
        sample = np.random.choice(data, size=len(data), replace=True)
        if bool_out:
            sample = mstats.winsorize(sample, limits=proportion)
        # means.append(statistics.mean(wins))
        means.append(np.mean(sample))
    return means

def std_calculation(expected_answer, annotator_response):
    tmp_answer = re.search(regex_pattern, expected_answer).group()
    other_remaining = expected_answer.replace(tmp_answer, "").strip()

    expected_answer = float(tmp_answer)

    list_response = []
    for r in annotator_response:
        if (r == "None") or (not r):
            pass
        else:
            list_response.append(float(re.search(regex_pattern, r).group()))

    list_response.append(expected_answer)
    list_annotator = copy.deepcopy(list_response)

    # list_response = 

    list_tf = []

    flag_bootstrap = False
    count = 0

    while True:
        std_value = statistics.stdev(list_response)

        count = 0
        for a in list_response:
            if (a < expected_answer-(3*std_value)) or (a > expected_answer+(3*std_value)):
                list_response.remove(a)
                count += 1
                flag_bootstrap = True
            else:
                pass
        
        
        if count == 0:
            break

    # std_value = np.std(winsorized_bootstrap(list_annotator, bool_out=flag_bootstrap), ddof=1)

    list_68 = []
    list_95 = []
    list_99 = []

    for a in list_annotator:
        if (a >= expected_answer-(1*std_value)) and (a <= expected_answer+(1*std_value)):
            list_68.append(True)
            list_95.append(True)
            list_99.append(True)
        elif (a >= expected_answer-(2*std_value)) and (a <= expected_answer+(2*std_value)):
            list_68.append(False)
            list_95.append(True)
            list_99.append(True)
        elif (a >= expected_answer-(3*std_value)) and (a <= expected_answer+(3*std_value)):
            list_68.append(False)
            list_95.append(False)
            list_99.append(True)
        else:
            list_68.append(False)
            list_95.append(False)
            list_99.append(False)

    # print(mean_value, median_value, std_value)
    
    # if flag_bootstrap:
    return {'annotator_response': list_annotator, 
            'std': round(std_value, 2),
            'expected_value': expected_answer,
            'expected_answer': f"{expected_answer} \pm {round(std_value, 2)} {other_remaining}",
            'list_68': list_68,
            'list_95': list_95,
            'list_99': list_99,}
    # else:
    #     return {'annotator_response': list_annotator, 
    #             'std': std_value,
    #             'win_std': std_value,
    #             'expected_value': expected_answer,}

pl_annotator = pl.read_json('/home/yaoyi/pyo00005/p2/carto-reasoning/questions/annotator_response/response_all.json')

pl_full = pl.concat(
    [pl_data, pl_annotator],
    how='align_left'
).select(
    pl.col('question_ref'),
    tmp = pl.struct(pl.all()).map_elements(lambda x: std_calculation(x['expected_answer'], x['annotator_response']))
).unnest('tmp').explode(['annotator_response', 'list_68', 'list_95', 'list_99'])

pl_full = pl_full.select(
    pl.col(['question_ref', 'expected_answer'])
).group_by('question_ref').agg([pl.all()]).with_columns(
    pl.col('expected_answer').list.first()
)

pl_full.write_csv('./tmp_w.csv')

print(pl_full)

####


# orientation = ['North',
#                'North West',
#                'West',
#                'South West',
#                'South',
#                'South East',
#                'East',
#                'North East']

# answer = 'North'

# answer_idx = orientation.index(answer)
# list_indexes = [answer_idx, answer_idx-1, answer_idx+1]
# acceptable = [orientation[i] for i in list_indexes]

# print(acceptable)

# import statistics

# data = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# # Calculate the mean
# mean_value = statistics.mean(data)
# print(f"Mean: {mean_value}")

# # Calculate the median
# median_value = statistics.median(data)
# print(f"Median: {median_value}")
