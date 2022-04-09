from typing import Any, Tuple

import numpy as np
from torchvision.datasets import CocoDetection

class AvCocoDetection(CocoDetection):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            result = self.transform(image=np.array(image), target=target)
            image = result['image']
            target = result['target']
            # image, target = self.transform(image=np.array(image), target=target)

        return image, target
