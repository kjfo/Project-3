from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Training was done on Google Colab. The best weights was downloaded to my personal laptop
# together with other graphs.

# from google.colab import drive
# drive.mount('/content/drive', force_remount = True)

# # Load the YOLOv8 model.
# model = YOLO('yolov8n.pt')

# # Train the data.
# train_model = model.train(data = '/content/drive/MyDrive/AER850 Project 3/Project 3 Data/data/data.yaml',
#                           epochs = 150,
#                           batch = 4,
#                           imgsz = 1000,
#                           name = 'AER850_Project_3_Train_Model')

# Load the trained model.
model = YOLO('best.pt')

evaluation = [
      r"C:\Users\kjfo0\OneDrive\Desktop\AER850 - Python Codes\3\Project 3 Data\data\evaluation\ardmega.jpg",
      r"C:\Users\kjfo0\OneDrive\Desktop\AER850 - Python Codes\3\Project 3 Data\data\evaluation\arduno.jpg",
      r"C:\Users\kjfo0\OneDrive\Desktop\AER850 - Python Codes\3\Project 3 Data\data\evaluation\rasppi.jpg"
  ]
titles = ['Arduino Mega Board with Overlaid Predictions',
          'Arduino Uno Board with Overlaid Predictions',
          'Raspberry Pi Board with Overlaid Predictions']

for i in range(3):
      # Load the image.
      eval = cv2.imread(evaluation[i])

      # Predict objects within the image using the trained model.
      result = model.predict(evaluation[i])
      result_image = result[0].plot()

      # Show the final image with predictions.
      plt.figure(figsize = (6,6))
      plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
      plt.title(titles[i])
      plt.axis('off')
      plt.show()
