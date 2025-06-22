import csv
import numpy as np


csv_file_frag_flag = ''
csv_file_attention = ''

with open(csv_file_frag_flag, 'r') as file:
    frag_flag_list = []
    reader_frag_flag = csv.reader(file)
    for row in reader_frag_flag:
        for frag_flag in row:
            frag_flag_list.append(frag_flag)

with open(csv_file_attention, 'r') as file:
    row_attention_list = []
    reader_attention = csv.reader(file)
    for row in reader_attention:
        for attention in row:
            row_attention_list.append(attention)
    attention_list = [x for x in row_attention_list if x != '']
    attention_float_list = [float(num) for num in attention_list]


def merge_lists_to_dict(list1, list2):
    merged_dict = {}
    for key, value in zip(list1, list2):
        if key in merged_dict:
            if isinstance(merged_dict[key], list):
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [merged_dict[key], value]
        else:
            merged_dict[key] = value
    return merged_dict


result_dict = merge_lists_to_dict(frag_flag_list, attention_float_list)

mean_dict = {}
median_dict = {}
square_error_dict = {}
for key, value in result_dict.items():
    if isinstance(value, list):
        mean_value = sum(value) / len(value)
        median_value = np.median(value)
        square_error = sum((x - mean_value) ** 2 for x in value) / len(value)
    else:
        mean_value = value
        median_value = value
        square_error = 0
    mean_dict[key] = mean_value
    median_dict[key] = median_value
    square_error_dict[key] = square_error


def min_max_normalize(data_dict):
    values = list(data_dict.values())
    min_val, max_val = min(values), max(values)
    if max_val - min_val == 0:
        return data_dict
    normalized_dict = {key: (value - min_val) / (max_val - min_val) for key, value in data_dict.items()}
    return normalized_dict


normalized_mean_dict = min_max_normalize(mean_dict)
normalized_median_dict = min_max_normalize(median_dict)


def write_combined_results_to_csv(keys, mean_dict, median_dict, square_error_dict, normalized_mean_dict, normalized_median_dict, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fragment', 'Mean', 'Median', 'Variance', 'Normalized_Mean', 'Normalized_Median'])
        for key in keys:
            writer.writerow([
                key,
                mean_dict.get(key, ''),
                median_dict.get(key, ''),
                square_error_dict.get(key, ''),
                normalized_mean_dict.get(key, ''),
                normalized_median_dict.get(key, '')
            ])


all_keys = result_dict.keys()
write_combined_results_to_csv(
    all_keys,
    mean_dict,
    median_dict,
    square_error_dict,
    normalized_mean_dict,
    normalized_median_dict,
    'combined_results.csv'
)



