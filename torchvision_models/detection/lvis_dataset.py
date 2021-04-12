from torchvision.datasets.vision import VisionDataset
from lvis import LVIS
import os
import os.path
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image

class LVISDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    """

    def __init__(
            self,
            root: str,
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(LVISDetection, self).__init__(root, transforms, transform, target_transform)
        self.lvis = LVIS(annFile)
        self.ids = list(sorted(self.lvis.imgs.keys()))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, id: int) -> Image.Image:
        path = self.lvis.load_imgs([id])[0]["coco_url"]
        path = path.split('/')[-2]+"/"+path.split('/')[-1]
        root = '/'.join(self.root.split('/')[:-1])
        return Image.open(os.path.join(root, path)).convert("RGB")

    def _load_target(self, id) -> List[Any]:
        return self.lvis.load_anns(self.lvis.get_ann_ids([id]))