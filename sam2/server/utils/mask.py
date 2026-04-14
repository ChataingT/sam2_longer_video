import numpy as np
from typing import List, Dict, Any
import torch

class FrameMask():
    target_ids:list
    mask:list

    def __init__(self, target_ids:list, mask_logits:list) -> None:
        """The order of ids and mask_logits are the same

        Args:
            ids (list): _description_
            mask_logits (list): _description_
        """
        self.target_ids = target_ids
        self.mask = mask_logits
        
    def items(self, binary=True):
        if binary:
            mask = [(self.mask[i] > 0.0).cpu().numpy() for i in range(len(self.mask))]
            return zip(self.target_ids, mask)
        return zip(self.target_ids, self.mask)

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.target_ids}-{self.mask}"
    
    def __getitem__(self, wanted_id):
        id = np.where(np.array(self.target_ids) == int(wanted_id))
        return self.mask[id]
    
    def __len__(self):
        """Return the number of masks"""
        return len(self.target_ids)
    
    def get_binary(self, id):
        """Mask are in float format with real value. They need to be binarized.

        Args:
            id (_type_): _description_

        Returns:
            _type_: _description_
        """
        m = self[id]
        m = (m > 0.0).cpu()
        return m
    
    def _convert_masks_to_rle(self):
        """
        Convert mask logits first to binary masks and then to RLE masks.
        """
        rle_masks = []
        for id in range(len(self.target_ids)):
            try:
                mask_index = self.target_ids[id]
                binary_mask_tensor = self.get_binary(mask_index)
                binary_mask_tensor = binary_mask_tensor.squeeze(0)
                if not isinstance(binary_mask_tensor, torch.Tensor):
                    raise ValueError(f"Expected a torch.Tensor, but got {type(binary_mask_tensor)}")
                if len(binary_mask_tensor.shape) != 3:
                    raise ValueError(f"Expected tensor shape (b, h, w), but got {binary_mask_tensor.shape}")
                rle_mask = self._mask_to_rle_pytorch(binary_mask_tensor)
                rle_masks.append(rle_mask)
            except Exception as e:
                print(f"Error converting mask to RLE for target id {self.target_ids[id]}: {e}")
        return rle_masks

    def to_dict(self):
        """Return a dictionary representation of the FrameMask instance"""
        rle_masks = self._convert_masks_to_rle()
        return {
            "target_ids": self.target_ids,
            "rle_masks": rle_masks
        }

    @classmethod
    def from_dict(cls, data: List[dict]):
        """Create a FrameMask instance from a dictionary

        Args:
            data (dict): Dictionary containing target_ids and mask_logits

        Returns:
            FrameMask: FrameMask instance
        """
        target_ids = [d.get('target_ids') for d in data]
        mask_logits = [d.get('mask_logits') for d in data]
        return cls(target_ids=target_ids, mask_logits=mask_logits)
    
    @staticmethod
    def _mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Encodes masks to an uncompressed RLE, in the format expected by
        pycoco tools.
        """
        # Put in fortran order and flatten h,w
        b, h, w = tensor.shape
        tensor = tensor.permute(0, 2, 1).flatten(1)

        # Compute change indices
        diff = tensor[:, 1:] ^ tensor[:, :-1]
        change_indices = diff.nonzero()

        # Encode run length
        out = []
        for i in range(b):
            cur_idxs = change_indices[change_indices[:, 0] == i, 1]
            cur_idxs = torch.cat(
                [
                    torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                    cur_idxs + 1,
                    torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
                ]
            )
            btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
            counts = [] if tensor[i, 0] == 0 else [0]
            counts.extend(btw_idxs.detach().cpu().tolist())
            out.append({"size": [h, w], "counts": counts})
        return out[0]
