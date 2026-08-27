from PIL import Image
from bill_parser.baseline.pipeline import image_to_bill

img = Image.open("/home/user/projects/Broccly/RoomGrub-bill-scanner/scripts/test-bill.jpg")
bill = image_to_bill(img)
print(bill.model_dump_json(indent=2))