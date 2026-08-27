from PIL import Image
from bill_parser.baseline import lines, ocr, parse
from bill_parser.schema import Bill

def image_to_bill(image: Image.Image) -> Bill:
  return parse.parse_lines(lines.group_lines(ocr.read(image)))