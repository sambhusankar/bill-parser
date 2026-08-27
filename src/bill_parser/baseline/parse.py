from __future__ import annotations
import re
from bill_parser.baseline.ocr import Box
from bill_parser.data.cord import parse_amount
from bill_parser.schema import Item, Bill

AMOUNT = re.compile(r"^[\d][\d.,]*$")
QTY = re.compile(r"^(\d{1,3})\s*[xX]?$")

SUBTOTAL_KW = ("subtotal", "sub total", "sub-total")
TAX_KW = ("tax", "ppn", "pb1", "pajak")
TOTAL_KW = ("total", "jumlah", "grand total")
STOP_kW = ("cash", "change", "tunai", "kembali", "debit", "card")

def _looks_like_amount(b: Box) -> bool:
  return bool(AMOUNT.match(b.text.replace(" ", ""))) and parse_amount(b.text) is not None

def parse_lines(lines: list[list[Box]]) -> Bill:
  bill = Bill()
  in_items = True

  for line in lines:
    text = " ".join(b.text for b in line)
    low = text.lower()
    amounts = [b for b in line if _looks_like_amount(b)]
    last = parse_amount(amounts[-1].text) if amounts else None

    if any(k in low for k in SUBTOTAL_KW):
      bill.subtotal, in_items = last, False
    elif any(k in low for k in TAX_KW):
      bill.tax, in_items = last, False
    elif any(k in low for k in TOTAL_KW):
      bill.total, in_items = last, False
    elif any(k in low for k in STOP_kW):
      in_items = False
    elif in_items and last is not None:
      words = [b for b in line if not _looks_like_amount(b)]
      name = " ".join(b.text for b in words).strip(" .-")
      if not any(c.isalpha() for c in name):
        continue    
      qty = None
      if words and (m := QTY.match(words[0].text)):
        qty = int(m.group(1))
        name = " ".join(b.text for b in words[1:]).strip(" .-") or name
      bill.items.append(Item(name = name, qty = qty, price = last))

  return bill