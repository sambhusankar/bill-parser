from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from PIL import Image

@dataclass(frozen=True)
class Box:
  text: str
  score: float
  x0: float
  y0: float
  x1: float
  y1: float

  @property
  def cy(self) -> float:
    return (self.y0 + self.y1) / 2

  @property
  def height(self) -> float:
    return self.y1 - self.y0


@lru_cache(maxsize=1)
def _engine():
  from paddleocr import PaddleOCR
  return PaddleOCR(
    lang="en",
    use_doc_orientation_classify = False,
    use_doc_unwarping = False,
    use_textline_orientation = False,
    enable_mkldnn = False
  )

def read(image: Image.Image, min_score: float = 0.5) -> list[Box]:
  arr = np.array(image.convert("RGB"))
  boxes: list[Box] = []
  for res in _engine().predict(arr):
    r = res["res"] if "res" in res else res
    for text, score, poly in zip(r["rec_texts"], r["rec_scores"], r["rec_polys"]):
      if score < min_score or not text.strip():
        continue
      p = np.asarray(poly, dtype=float)
      boxes.append(Box(text.strip(), float(score),
                       p[:,0].min(), p[:,1].min(),
                       p[:, 0].max(), p[:,1].max()))

  return boxes