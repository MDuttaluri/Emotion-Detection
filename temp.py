import cv2

img = cv2.imread("image.png")
print(img.shape)

with open('imageText.txt', 'w') as out:
    for row in img:
        t = '['
        for col in row:
            t += str(col) + ','
        t = t[:-1] + ']'
        out.write(str(row))
