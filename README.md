## histogram
1.Title: and Description
Title: Visualizing Color Histogram of an Image
Description: This Python script reads an image, calculates its color histogram, and visualizes the histogram using matplotlib.

2.Installation:
pip install numpy opencv-python matplotlib and argparse

3.Libraries:
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import argparse

4.Usage:
A histogram allows you to see the frequency distribution of a data set. It offers an “at a glance” picture of a distribution pattern, charted in specific categories.

5.Example:
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("image_path", help = "Enter the path of the image")

parser.add_argument("out_dir", help = "name of the output directory where you want to save your output")

args = parser.parse_args()
image_dir = args.image_path
 
img = cv.imread('/home/swetha-konda/Downloads/parrot.jpeg')
cv.imwrite("/home/swetha-konda/Downloads/prt.jpeg",img)
assert img is not None, "file could not be read, check with os.path.exists()"
color = ('b','g','r')
for i,col in enumerate(color):
 histr = cv.calcHist([img],[i],None,[256],[0,256])
 plt.plot(histr,color = col)
 plt.xlim([0,256])
plt.show()

6.Explaination:
Parsing Command-line Arguments:
parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="Enter the path of the image")
parser.add_argument("out_dir", help="Name of the output directory where you want to save your output")
args = parser.parse_args()
image_dir = args.image_path
This section parses command-line arguments. It expects two arguments: the path of the image and the output directory where the output will be saved.
Reading and Displaying Image:
img = cv.imread('/home/swetha-konda/Downloads/parrot.jpeg')
cv.imwrite("/home/swetha-konda/Downloads/prt.jpeg",img)
Reads an image using OpenCV's `imread()` function.
Writes the image to a file named 'prt.jpeg' using imwrite() function.
Histogram Calculation and Plotting:
assert img is not None, "file could not be read, check with os.path.exists()"
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
Checks if the image was read successfully. 
Defines a tuple `color` containing the colors blue, green, and red.
Iterates over each color channel (blue, green, red).
Calculates the histogram for each color channel using `calcHist()` function from OpenCV.
Plots the histogram using `matplotlib` with each color channel represented by a different color.
Displays the histogram.
Overall, this code reads an image, calculates histograms for each color channel (blue, green, red), and plots them using matplotlib. It also allows for command-line arguments to specify the image path and the output directory for saving the output.

7.output:
input:
![parrot](https://github.com/kondasweth/manasa/assets/169050846/782d7016-64c3-4641-a6a5-c81aac383da7)
output:
![Figure_1](https://github.com/kondasweth/manasa/assets/169050846/da2aad9d-7524-4078-aefa-bea3cac3a93c) 

## Bounding Box
1.Title:Draw bounding boxes for images.

2.Installation
pip install pillow

3.Libraries
  `os`: for interacting with the operating system.
  `csv`: for reading CSV files.
  `PIL`: Python Imaging Library, for handling images.
  `argparse`: for parsing command-line arguments.

4.Usage:
A histogram allows you to see the frequency distribution of a data set. It offers an “at a glance” picture of a distribution pattern, charted in specific categories.

5.Example:
import os
import csv
from PIL import Image, ImageDraw
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image_path", help = "Enter the path of the image")
parser.add_argument("csv", help = "your csv file path, which has bounding box values")
parser.add_argument("out_dir", help = "name of the output directory where you want to save your output")
args = parser.parse_args()
image_dir = args.image_path
csv_file = args.csv
output_dir = args.out_dir


os.makedirs(output_dir, exist_ok=True)


def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        draw.rectangle([left, top, right, bottom], outline="red")
    return image


def crop_image(image, boxes):
    cropped_images = []
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        cropped_img = image.crop((left, top, right, bottom))
        cropped_images.append(cropped_img)
    return cropped_images


with open(csv_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    print(csv_reader)

    for row in csv_reader:
        image_name = row['filename']
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        image = Image.open(image_path)
        boxes = [{'left': row['xmin'], 'top': row['ymin'], 'right': row['xmax'], 'bottom': row['ymax']}]
        cropped_images = crop_image(image, boxes)
        for i, cropped_img in enumerate(cropped_images):
           cropped_img.save(os.path.join(output_dir, f"{i}_{image_name}"))  
        full_image_with_boxes = draw_boxes(image, boxes)
        full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))

6.Explaination:
Parsing Command-line Arguments:
parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="Enter the path of the image")
parser.add_argument("csv", help="Your CSV file path, which has bounding box values")
parser.add_argument("out_dir", help="Name of the output directory where you want to save your output")
args = parser.parse_args()
image_dir = args.image_path
csv_file = args.csv
output_dir = args.out_dir
This section uses argparse to parse command-line arguments provided when the script is run. It expects three arguments: image path, CSV file path, and output directory.
Creating Output Directory:
os.makedirs(output_dir, exist_ok=True)
This line ensures that the output directory exists. If it doesn't, it creates it.
Functions Definition:
`draw_boxes(image, boxes)`: This function takes an image and a list of boxes (bounding box coordinates) and draws rectangles around those boxes on the image.
`crop_image(image, boxes)`: This function crops the image based on the bounding box coordinates and returns a list of cropped images.
Reading and Processing CSV File:
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Extracting image information from the CSV
            image_name = row['filename']
            image_path = os.path.join(image_dir, image_name)
            output_path = os.path.join(output_dir, image_name)
            image = Image.open(image_path)
            boxes = [{'left': row['xmin'], 'top': row['ymin'], 'right': row['xmax'], 'bottom': row['ymax']}]
            cropped_images = crop_image(image, boxes)
            # Saving cropped images
            for i, cropped_img in enumerate(cropped_images):
                cropped_img.save(os.path.join(output_dir, f"{i}_{image_name}"))  
            # Drawing bounding boxes on the original image
            full_image_with_boxes = draw_boxes(image, boxes)
            # Saving the original image with bounding boxes
            full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))
It opens the CSV file and iterates over each row.  For each row, it extracts the image filename and constructs the full image path. Then it opens the image, extracts the bounding box coordinates, and calls the `crop_image()` function to crop the image based on the bounding box coordinates.
It saves each cropped image with a filename that includes an index and the original image name.After that, it calls the `draw_boxes()` function to draw bounding boxes on the original image and saves the image with the bounding boxes drawn.
This script essentially processes images and bounding box data from a CSV file, extracts objects from the images based on the provided bounding box coordinates, and saves both the cropped objects and the original images with bounding boxes drawn around the objects.

7.output:
input:
![image](https://github.com/kondasweth/Jiyanshi/assets/169050846/e02aa1b8-7227-4f8f-a538-3bc847177b7b)

output 1:
![image](https://github.com/kondasweth/Jiyanshi/assets/169050846/08564941-15b3-48f4-9032-c78631fe3f0c)

output 2:
![image](https://github.com/kondasweth/Jiyanshi/assets/169050846/821273b1-f771-45cb-b3a9-4b421ac09551)


## number
1.Example:
import argparse
parser = argparse.ArgumentParser(description='process some integers')

parser.add_argument('current_number', type=int)
parser.add_argument('previous_number', type=int)
args = parser.parse_args()
current_num = args.current_number
previous_num = args.previous_number

for i in range(10):
    sum = previous_num + i
    print(f'Current number {i} Previous Number {previous_num} is {sum}')
    previous_num = i

2.Output
Current number 0 Previous Number 1 is 1

Current number 1 Previous Number 0 is 1

Current number 2 Previous Number 1 is 3

Current number 3 Previous Number 2 is 5

Current number 4 Previous Number 3 is 7

Current number 5 Previous Number 4 is 9

Current number 6 Previous Number 5 is 11

Current number 7 Previous Number 6 is 13

Current number 8 Previous Number 7 is 15

Current number 9 Previous Number 8 is 17

    



























## web
```

import cv2 as cv
cam = cv.VideoCapture(0)
cc = cv.VideoWriter_fourcc(*'XVID')
file = cv.VideoWriter('output.avi', cc, 15.0, (640, 480))
if not cam.isOpened():
   print("error opening camera")
   exit()
while True:
   # Capture frame-by-frame
   ret, frame = cam.read()
   # if frame is read correctly ret is True
   if not ret:
      print("error in retrieving frame")
      break
   img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
   cv.imshow('frame', img)
   file.write(img)

   
   if cv.waitKey(1) == ord('q'):
      break

cam.release()
file.release()
cv.destroyAllWindows()
```



