from collections import Counter
import pandas as pd
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import torchvision.transforms as transforms
import shutil
import time
from tqdm import tqdm

# 타이머 시작
print("-------------Start Augmentation-------------")
st = time.time()

train_df = pd.read_csv("./data/ISIC_2019/train.csv")
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])
test_df = pd.read_csv("./data/ISIC_2019/test.csv")
test_df['image_path'] = test_df['image_path'].apply(lambda x: x.replace("test/",""))

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

train_df['image_path'] = train_df['image_path'].apply(lambda x: x.replace('train/train', 'train'))
val_df['image_path'] = val_df['image_path'].apply(lambda x: x.replace('train/train', 'val'))

if not os.path.exists("./data/ISIC_2019_aug"):
    os.makedirs("./data/ISIC_2019_aug")
else:
    shutil.rmtree("./data/ISIC_2019_aug")
    os.makedirs("./data/ISIC_2019_aug")    

train_df.to_csv("./data/ISIC_2019_aug/train.csv", index=False)
val_df.to_csv("./data/ISIC_2019_aug/val.csv", index=False)
test_df.to_csv("./data/ISIC_2019_aug/test.csv", index=False)

train_df = pd.read_csv("./data/ISIC_2019_aug/train.csv")
val_df = pd.read_csv("./data/ISIC_2019_aug/val.csv")
train_class_count = Counter(train_df['label'])
val_class_count = Counter(val_df['label'])
print("Before train_df class_count: ", train_class_count)
print("Before val_df class_count: ", val_class_count)

# 데이터 증강을 위한 디렉토리 생성
if not os.path.exists('./data/ISIC_2019_aug/train'):
    os.makedirs('./data/ISIC_2019_aug/train')

if not os.path.exists('./data/ISIC_2019_aug/val'):
    os.makedirs('./data/ISIC_2019_aug/val')

# 기존 train 데이터 복사
for train in train_df.iterrows():
    shutil.copy(f"./data/ISIC_2019/train/{train[1]['image_path']}", f"./data/ISIC_2019_aug/train/{train[1]['image_path']}")

# 기존 val 데이터 복사
for val in val_df.iterrows():
    shutil.copy(f"./data/ISIC_2019/train/{val[1]['image_path'].replace('val', 'train')}", f"./data/ISIC_2019_aug/val/{val[1]['image_path']}")

# 기존 test 데이터 복사
for test in test_df.iterrows():
    shutil.copy(f"./data/ISIC_2019/test/{test[1]['image_path']}", f"./data/ISIC_2019_aug/test/{test[1]['image_path']}")


# train 데이터에 대해서만 증강
max_count = max(train_class_count.values())
oversample_plan = {
    cls: max_count - count
    for cls, count in train_class_count.items()
    if count < max_count
}


save_dir = './data/ISIC_2019_aug/train/'
os.makedirs(save_dir, exist_ok=True)

augmentations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomGrayscale(p=0.1)
])


new_filenames = []
new_labels = []

for cls in oversample_plan:
    cls_df = train_df[train_df['label'] == cls]
    samples_to_add = oversample_plan[cls]
    
    for i in tqdm(range(samples_to_add), desc=f"Augmenting class {cls}", total=samples_to_add):
        row = cls_df.sample(1).iloc[0]
        img = Image.open(os.path.join('./data/ISIC_2019_aug', 'train', row['image_path'])).convert('RGB')
        aug_img = augmentations(img)
                                                                                               
        new_filename = f"train_aug_{i}.jpg"
        aug_img.save(os.path.join(save_dir, new_filename))

        new_filenames.append(new_filename)
        new_labels.append(row['label'])

train_df = pd.concat([train_df, pd.DataFrame({'image_path': new_filenames, 'label': new_labels})], ignore_index=True)
train_df.to_csv("./data/ISIC_2019_aug/train.csv", index=False)

train_df = pd.read_csv("./data/ISIC_2019_aug/train.csv")
val_df = pd.read_csv("./data/ISIC_2019_aug/val.csv")
train_class_count = Counter(train_df['label'])
val_class_count = Counter(val_df['label'])
print("After train_df class_count: ", train_class_count)
print("After val_df class_count: ", val_class_count)

# 타이머 종료
et = time.time()
print("-------------End Augmentation-------------")
print("Augmentation Time: ", et - st)