from __future__ import annotations
import statistics
from bill_parser.baseline.ocr import Box


def group_lines(boxes: list[Box], tol: float = 0.6) -> list[list[Box]]:
  if not boxes:
    return []

  h = statistics.median(b.height for b in boxes) or 1.0
  threshold = tol * h

  lines: list[list[Box]] = []

  for b in sorted(boxes, key = lambda b: b.cy):
    if lines and abs(b.cy - statistics.mean(x.cy for x in lines[-1])) <= threshold:
      lines[-1].append(b)
    else:
      lines.append([b])

  return [sorted(line, key = lambda b: b.x0) for line in lines]