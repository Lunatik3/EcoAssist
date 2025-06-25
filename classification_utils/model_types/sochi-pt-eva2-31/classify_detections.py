import sys
import os
import math
import logging
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
import albumentations as A

# -----------------------------------------------------------------------------
# 1. Подключаем локальные модули megadetector и visualise_detection
# -----------------------------------------------------------------------------
EcoAssist_files = str(sys.argv[1])
sys.path.insert(0, os.path.join(EcoAssist_files, 'cameratraps', 'megadetector'))
sys.path.insert(0, os.path.join(EcoAssist_files, 'visualise_detection'))

# -----------------------------------------------------------------------------
# 2. Парсим аргументы
# -----------------------------------------------------------------------------
cls_model_fpath = str(sys.argv[2])
cls_detec_thresh = float(sys.argv[3])
cls_class_thresh = float(sys.argv[4])
smooth_bool = sys.argv[5] == 'True'
json_path = str(sys.argv[6])
temp_frame_folder = None if sys.argv[7] == 'None' else str(sys.argv[7])

# -----------------------------------------------------------------------------
# 3. Настройка логирования
# -----------------------------------------------------------------------------
log_dir = os.path.join(EcoAssist_files, 'EcoAssist', 'logfiles')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'sochi_classification_model.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, 'w'), logging.StreamHandler()]
)
logging.info('Arguments:')
for name in ['EcoAssist_files','cls_model_fpath','cls_detec_thresh','cls_class_thresh','smooth_bool','json_path','temp_frame_folder']:
    logging.info(f'{name}={locals()[name]}')

# -----------------------------------------------------------------------------
# 4. Загрузка CSV с классами и модели
# -----------------------------------------------------------------------------
cls_dir = Path(cls_model_fpath).parent
class_csv = cls_dir / 'classes.csv'
checkpoint = cls_dir / 'eva_2_no_ema.pt'
logging.info(f'Loading classes from {class_csv}')
classes = pd.read_csv(class_csv, usecols=[0], header=None, names=['class'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Device: {device}')
logging.info(f'Loading model from {checkpoint}')
model = torch.jit.load(str(checkpoint), map_location=device)
model.to(device)
model.eval()

# -----------------------------------------------------------------------------
# 5. Вычисление input_size на основе pos_embed модели
# -----------------------------------------------------------------------------
patch_size = 14  # для eva-2
orig_pos = model.pos_embed  # [1, num_patches+1, C]
num_patches = orig_pos.shape[1] - 1
grid = int(math.sqrt(num_patches))
input_size = grid * patch_size
logging.info(f'Computed input_size={input_size} (grid={grid}, patch_size={patch_size})')

# -----------------------------------------------------------------------------
# 6. Предобработка изображений (Albumentations)
# -----------------------------------------------------------------------------
preprocess = A.Compose([
    A.LongestMaxSize(max_size=input_size, p=1.0),
    A.PadIfNeeded(min_height=input_size, min_width=input_size,
                  position='center', border_mode=0, fill=0, p=1.0),
    A.Resize(height=input_size, width=input_size, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
    A.transforms.ToTensorV2(p=1.0),
])

# -----------------------------------------------------------------------------
# 7. Функция вырезки (crop)
# -----------------------------------------------------------------------------
def get_crop(img, bbox):
    w, h = img.size
    x, y, w_box, h_box = bbox
    left = int(w * x)
    top = int(h * y)
    right = int(w * (x + w_box))
    bottom = int(h * (y + h_box))
    return img.crop((left, top, right, bottom))


# -----------------------------------------------------------------------------
# 8. Функция классификации одного кропа
# -----------------------------------------------------------------------------
def get_classification(pil_img):
    img_np = np.array(pil_img)
    tensor = preprocess(image=img_np)['image']  # C×H×W
    batch = tensor.unsqueeze(0).to(device).to(next(model.parameters()).dtype)
    out = model(batch)
    probs = F.softmax(out, dim=1).cpu().detach().numpy()[0]
    return [[classes['class'].iloc[i], float(p)] for i, p in enumerate(probs)]

# -----------------------------------------------------------------------------
# 9. Запуск классификации через inference_lib
# -----------------------------------------------------------------------------
import EcoAssist.classification_utils.inference_lib as ea
logging.info('Starting main classification...')
ea.classify_MD_json(
    json_path=json_path,
    GPU_availability=(device.type != 'cpu'),
    cls_detec_thresh=cls_detec_thresh,
    cls_class_thresh=cls_class_thresh,
    smooth_bool=smooth_bool,
    crop_function=get_crop,
    inference_function=get_classification,
    temp_frame_folder=temp_frame_folder,
    cls_model_fpath=cls_model_fpath
)
logging.info('Classification completed.')
