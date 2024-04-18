import json

# JSON 文件的路径
json_file_path = 'S:\\Mv_py_ass\\counting\\test\\ground_truth.json'
# 新 TXT 文件的路径
txt_file_path = 'S:\\Mv_py_ass\\counting\\test\\test_ground_truth.txt'

# 读取 JSON 文件
with open(json_file_path, 'r') as f:
    annotations_json = json.load(f)

# 打开一个新的 TXT 文件来写入
with open(txt_file_path, 'w') as f:
    # 写入标题行
    f.write('Image,count\n')
    # 遍历 JSON 字典并写入每一行
    for image_name, count in annotations_json.items():
        f.write(f"{image_name},{count}\n")
