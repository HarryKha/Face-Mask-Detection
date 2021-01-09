import cv2
import preprocess
import training
import torch
import torchvision.transforms as transforms
from PIL import Image

cap = cv2.VideoCapture(0)
red = (0, 0, 255)
blue = (255, 0, 0)
green = (0, 255, 0)

filepath = preprocess.models_dir + str(1) + ".pth"
loaded_model = training.load_checkpoint(filepath)

train_transforms = transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                    ])
                                    
face_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_default.xml')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cropped_img = frame[y:y+h, x:x+w]
            pil_img = Image.fromarray(cropped_img, mode='RGB')
            pil_img = train_transforms(pil_img)
            # cv2.imshow('crop', pil_img)
            image = pil_img.unsqueeze(0)
            
            result = loaded_model(image.to("cuda:0"))
            predictions = torch.max(result.data, 1)[1]
            pred = predictions.item()
            print(pred)

            if pred == 0: # With mask
                cv2.putText(frame, "Mask On", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), green, 2)
            elif pred == 1: # No Mask
                cv2.putText(frame, "NO Mask", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), red, 2)
            elif pred == 2: # Mask worn incorrectly
                cv2.putText(frame, "Mask Worn Incorrectly", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, blue, 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), blue, 2)
        
        cv2.imshow('frame', frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

