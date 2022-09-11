from PIL import Image
im = Image.open("/home/krishna/ML/ocd/test/test.jpeg")

crop_rectangle = (50, 50, 200, 200)
cropped_im = im.crop(crop_rectangle)

cropped_im.show()
