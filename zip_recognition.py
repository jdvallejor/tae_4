import cv2
from sklearn.externals import joblib
# from skimage.feature import hog
import numpy as np

print("Welcome to zip code analyzer")
print("----------------------------")
model = 0
while(not model):
    print("\nPlease choose the model:\n")
    print("1. Multinomial logistic regression")
    print("2. Decision tree")
    print("3. Random forest")
    print("4. Support vector machine")
    print("5. Neural network")

    model = input("\n-->\t")
    
    if(int(model) == 1):
        model_path = "models/Logistic_model.pkl"
    elif(int(model) == 2):
        model_path = "models/DecisionTree.pkl"
    elif(int(model) == 3):
        model_path = "models/RandomForest.pkl"
    elif(int(model) == 4):
        model_path = "models/SVM_model.pkl"
    elif(int(model) == 5):
        model_path = "models/NeuralNetwork.pkl"
    else:
        print("\nInvalid option\n")

# Load the classifier
clf = joblib.load(model_path)

# Read the input image 
im = cv2.imread("3.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate features
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    roi = roi / 255

    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
        
    # Prediction
    nbr = clf.predict([roi.flatten()])
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.imshow("Resulting Image", im)
cv2.waitKey(0) 


cv2.destroyAllWindows()
