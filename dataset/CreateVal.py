def read_csv_to_set(filename):
    """读取CSV文件，将每一行转换为一个字符串，并存储到集合中返回。"""
    with open(filename, 'r') as file:
        # 使用集合来存储行，因为我们关心的是唯一性和存在性检查，集合在这方面更高效
        lines = set(line.strip() for line in file)
    return lines


def filter_evaluation_rows(evaluation_file, filter_set, output_file):
    """过滤掉evaluation文件中存在于filter_set中的行，并将结果输出到新的CSV文件。"""
    with open(evaluation_file, 'r') as eval_file, open(output_file, 'w') as out_file:
        for line in eval_file:
            if line.strip() not in filter_set:
                out_file.write(line)

# 步骤1: 读取car_1000.csv文件到一个集合中
car_1000_set = read_csv_to_set('car_1000.txt')

# 步骤2和步骤3: 过滤掉car_evaluation.csv中的行，并将结果输出到val.csv
filter_evaluation_rows('car_evaluation.csv', car_1000_set, 'val.csv')

print("处理完成，并将结果保存到val.csv中。")