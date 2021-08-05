import cv2 
import matplotlib.pyplot as plt 



# image_path = "/data/remote/github_code/face_detection/Pytorch_Retinaface/test_2.jpg"
image_path = "/data/remote/github_code/face_detection/Pytorch_Retinaface/detect_result1/53ef67d2caa93154054152556c4e34ee.jpeg"
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
img = cv2.rectangle(img, (438, 612), (549+438, 568+612), (255, 255, 0), 2)
cv2.imwrite("/data/remote/github_code/face_detection/Pytorch_Retinaface/test_3.jpg", img)