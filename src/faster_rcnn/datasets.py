# datasets.py


from typing import Dict, Any, List, Tuple
import torch, os,json
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

# xywh -> xyxy 변환 함수
def xywh_to_xyxy(box: List[float]) -> List[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]

# 타겟의 바운딩 박스, area를 원본크기에서 리사이즈 크기로 스케일링
def resize_target(target: Dict ,original_size, new_size):
    ow, oh = original_size
    nw, nh = new_size
    scale_x, scale_y = nw / ow, nh / oh

    if "boxes" in target and target["boxes"].numel() > 0:
        boxes = target["boxes"].clone().float()
        # x는 sx, y는 sy로 각각 스케일
        boxes[:, [0,2]] *= scale_x
        boxes[:, [1,3]] *= scale_y
        target["boxes"] = boxes

        # 리사이즈된 박스 좌표 맞춰 면적 다시 계산
        w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        h = (boxes[:, 3] - boxes[:,1]).clamp(min=0)
        target["area"] = (w * h).to(torch.float32)
    else:
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        target["area"] = torch.zeros((0,), dtype=torch.float32)

    return target

class CocoDetWrapped(CocoDetection):
    def __init__(self, img_folder: str, ann_file: str, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms


    def __getitem__(self, idx: int):
        img, anno = super().__getitem__(idx)

        orig_w, orig_h = img.size

        # bbox 없거나 crowd인 항목 제거
        anno = [o for o in anno if ('bbox' in o) and (o.get('iscrowd', 0) == 0)]

        boxes, labels, area, iscrowd = [], [], [], []
        for o in anno:
            boxes.append(xywh_to_xyxy(o["bbox"]))
            labels.append(int(o["category_id"]))
            area.append(float(o.get("area", o["bbox"][2] * o["bbox"][3])))
            iscrowd.append(int(o.get("iscrowd", 0)))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0,4), dtype=torch.float32)
            labels_t= torch.zeros((0,),  dtype=torch.int64)
            area_t = torch.zeros((0,),  dtype=torch.float32)
            iscrowd_t = torch.zeros((0,),  dtype=torch.uint8)
        else:
            boxes_t = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            area_t = torch.as_tensor(area,   dtype=torch.float32)
            iscrowd_t = torch.as_tensor(iscrowd,dtype=torch.uint8)

        image_id = torch.tensor([self.ids[idx]], dtype=torch.int64)
        target: Dict[str, Any] = {
            "boxes": boxes_t, "labels": labels_t, "image_id": image_id,
            "area": area_t, "iscrowd": iscrowd_t
        }
        if self._transforms is not None:
            img = self._transforms(img)
            # 모델 입력으로 들어가는 이미지 크기 기록
            _, nh, nw = img.shape
            target["img_size"] = (nw, nh)
            target = resize_target(target, (orig_w, orig_h), (nw, nh))
        return img, target





def build_datasets_from_paths(train_img: str, train_ann: str, val_img: str, val_ann: str):

    train_tfms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor()
    ])
    val_tfms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor()
    ])

    return (
        CocoDetWrapped(train_img, train_ann, transforms=train_tfms),
        CocoDetWrapped(val_img,  val_ann,  transforms=val_tfms),
    )
# Test Dataset
class TestDataset(Dataset):
    def __init__(self, img_dir="./test"):
        self.img_dir = Path(img_dir)
        self.images = sorted(list(self.img_dir.glob("*.png")))
        self.transform = T.Compose([T.Resize((640,640)), T.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")

        orig_w, orig_h = img.size

        img_tensor = self.transform(img)
        _, nh, nw = img_tensor.shape


        target = {"image_id": idx, "file_name": os.path.basename(img_path),
                  "orig_size": (orig_w, orig_h), "img_size": (nw, nh)}
        return img_tensor, target


def build_test_dataset(img_dir="./test"):
    return TestDataset(img_dir)

