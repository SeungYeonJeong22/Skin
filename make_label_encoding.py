import json
import os
import pandas as pd
import argparse

def get_dataset(data_path):
    df = pd.read_csv(data_path)
    label = df['label'].unique()

    if label.dtype != 'O':
        label = label.astype('str')

    label_to_int = {label: idx for idx, label in enumerate(label)}
    int_to_label = {idx: label for idx, label in enumerate(label)}

    data_name = data_path.split("/")[2]

    os.makedirs('./data/', exist_ok=True)

    with open(f'./data/{data_name}/int2label.json', 'w', encoding='utf-8') as f:
        json.dump(
            {'label_to_int': label_to_int, 'int_to_label': int_to_label},
            f,
            indent=4,
            ensure_ascii=False  # 한글 깨짐 방지
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate label mapping JSON from CSV file")
    parser.add_argument('--data', type=str, required=True, help="Path to the input CSV file")

    args = parser.parse_args()
    data_path = os.path.join("./data", args.data, 'train.csv')

    get_dataset(data_path)


# import json
# import os
# import pandas as pd

# def get_dataset(data_path):
#     df = pd.read_csv(data_path)
#     label = df['label'].unique()

#     if label.dtype != 'O':
#         label = label.astype('str')

#     label_to_int = {label: idx for idx, label in enumerate(label)}
#     int_to_label = {idx: label for idx, label in enumerate(label)}

#     data_name = data_path.split("/")[2]

#     os.makedirs('./data/label', exist_ok=True)

#     # JSON 파일로 저장
#     with open(f'./data/label/{data_name}_int2label.json', 'w', encoding='utf-8') as f:
#         json.dump(
#             {'label_to_int': label_to_int, 'int_to_label': int_to_label},
#             f,
#             indent=4,
#             ensure_ascii=False  # <-- 한글 깨짐 방지
#         )

# if __name__=="__main__":
#     data_path_list = [
#                         # "./data/AICamp-2023-Skin-Conditions_Dataset/test.csv",
#                         # "./data/Augmented_Skin_Conditions_Kaggle/test.csv",
#                         # "./data/MIT_skin-lesions-classification-dataset/test.csv",
#                         # "./data/ISIC_2019/test.csv"
#                         # "./data/modified_AICamp/all.csv"
#                         # "./data/crawling_data/test.csv"
#                         # "./data/SCIN/train.csv"
#                         # "./data/SCIN2/train.csv"
#                         # "./data/SCIN_aug/train.csv"
#                         "./data/SCIN_7/train.csv"
#                       ]
    
#     for data_path in data_path_list:
#         get_dataset(data_path)