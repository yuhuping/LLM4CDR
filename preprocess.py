import csv
import json
import ast

# 文件名设置
merged_data_file = 'new_merged_data.csv'
movies_meta_file = 'Movies_TV_MetaData.csv'
cd_meta_file = 'CD_MetaData.csv'
output_json_file = 'llm4cdr_dataset.json'  # 输出为一个 JSON 数组

# 加载 Movies 元数据 (映射：iid -> title)
movies_meta = {}
with open(movies_meta_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader, None)  # 跳过表头，如无表头则注释此行
    for row in reader:
        if len(row) < 2:
            continue
        movie_iid, title = row[0].strip(), row[1].strip()
        movies_meta[movie_iid] = title

# 加载 CD 元数据 (映射：iid -> title)
cd_meta = {}
with open(cd_meta_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader, None)  # 跳过表头，如无表头则注释此行
    for row in reader:
        if len(row) < 2:
            continue
        cd_iid, title = row[0].strip(), row[1].strip()
        cd_meta[cd_iid] = title

# 用于存储所有 JSON 记录的列表
records = []

# 统计读取记录和写入记录数
total_records = 0
written_records = 0

# 读取 new_merged_data.csv 并生成 JSON 记录
with open(merged_data_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    # header = next(reader, None)  # 跳过表头，如无表头请注释此行
    for row in reader:
        total_records += 1
        # 这里假设 CSV 文件至少应包含4个字段：[uid, target_iid, rating, src_iids]
        if len(row) < 4:
            print(f"跳过第 {total_records} 行，字段数量不足：{row}")
            continue

        uid = row[0].strip()
        target_iid = row[1].strip()
        rating = row[2].strip()  # 评分字段保留字符串格式
        src_iids_str = row[3].strip()

        # 解析 src_iids 的字符串表示（例如 "['xxx', 'yyy', ...]"）
        try:
            src_iids = ast.literal_eval(src_iids_str)
            if not isinstance(src_iids, list):
                print(f"第 {total_records} 行解析后不是列表：{src_iids_str}")
                continue
        except Exception as e:
            print(f"第 {total_records} 行解析 src_iids 时出错：{src_iids_str}，错误：{e}")
            continue

        # 根据 src_iids_limited 查找对应电影标题，找不到则保留原 id
        src_titles = []
        for src_id in src_iids:
            title = movies_meta.get(src_id, None)
            if title:
                src_titles.append(title)
            # else:
            #     src_titles.append(src_id)
        src_titles = src_titles[:10]  # 限制最多 10 个标题

        # 根据 target_iid 查找对应的 CD 标题，找不到则保留原 id
        target_title = cd_meta.get(target_iid, target_iid)

        # 构造 JSON 对象，"input" 部分列出用户偏好的电影标题
        json_record = {
            "instruction": "Given the user's preference in Movies, identify whether the user will like the target music by answering on a scale of 1 to 5.",
            "input": 'User Preference: ' + '、'.join([f"\"{t}\"" for t in src_titles]) +
                     f'\nWhether the user will like the target music "{target_title}"?',
            "output": rating
        }

        records.append(json_record)
        written_records += 1

# 将所有记录存放在一个 JSON 数组中写入输出文件
with open(output_json_file, 'w', encoding='utf-8') as out_f:
    json.dump(records, out_f, ensure_ascii=False, indent=2)

print(f"共读取 {total_records} 条记录，写入 {written_records} 条 JSON 记录到 {output_json_file}")
