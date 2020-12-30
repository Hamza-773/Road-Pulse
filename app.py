import pyrebase
from werkzeug.utils import secure_filename
from flask import render_template, request, redirect, session
import os
from datetime import datetime
from google.cloud import firestore
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
import glob
from flask import *
from flask import jsonify
from flask.helpers import url_for
import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os

# customize your API through the following parameters
classes_path = './data/labels/obj.names'
weights_path = './weights/yolov3.tf'
tiny = False                    # set to True if using a Yolov3 Tiny model
size = 416                      # size images are resized to for model
output_path = './detections/'   # path to output folder where images with detections are saved
num_classes = 2                # number of classes in model
displayData = {}
SIZE_vid=0


# load in weights and classes
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tiny:
    yolo = YoloV3Tiny(classes=num_classes)
else:
    yolo = YoloV3(classes=num_classes)

yolo.load_weights(weights_path).expect_partial()
print('weights loaded')

class_names = [c.strip() for c in open(classes_path).readlines()]
print('classes loaded')

# # API that returns image with detections on it
config = {
  "apiKey": "AIzaSyAZ1NfqphkXILRHSOQklQLJvB_hyFOthIg",
  "authDomain": "flaskapp-990a1.firebaseapp.com",
  "databaseURL": "https://flaskapp-990a1.firebaseio.com",
  "projectId": "flaskapp-990a1",
  "storageBucket": "flaskapp-990a1.appspot.com",
  "messagingSenderId": "829954679478",
  "appId": "1:829954679478:web:9b4bc407419ff359977c17",
  "measurementId": "G-LMKTB9S62K"
}
cred = credentials.Certificate('flaskapp-990a1-firebase-adminsdk-qede9-6d433e29ff.json')
firebase_admin.initialize_app(cred)
firebase = pyrebase.initialize_app(config)

storage = firebase.storage()

db = firestore.client()

auth = firebase.auth()

# cam = cv2.VideoCapture(0)
app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    if (request.method == 'POST'):
            email = request.form['name']
            password = request.form['password']
            try:
                auth.sign_in_with_email_and_password(email, password)
                #user_id = auth.get_account_info(user['idToken'])
                #session['usr'] = user_id
                return render_template('Dashboard.html')
            except:
                unsuccessful = 'Please check your credentials'
                return render_template('index.html', umessage=unsuccessful)
    return render_template('index.html')

@app.route('/create_account', methods=['GET', 'POST'])
def create_account():
    if (request.method == 'POST'):
            email = request.form['name']
            password = request.form['password']
            auth.create_user_with_email_and_password(email, password)
            return render_template('index.html')
    return render_template('create_account.html')

# def stream():
#     while 1 :
#         __,frame = cam.read()
#         imgencode = cv2.imencode('.jpg',frame)[1]
#         strinData = imgencode.tostring()
#         yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+strinData+b'\r\n')

# @app.route('/video')
# def video():
#     return Response(stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/webcam')
# def main():
#     return render_template('cam.html')


@app.route('/Contactus')
def Contactus():
    return render_template('Contactus.html')

@app.route('/Dash')
def Dash():
    return render_template('Dashboard.html')

@app.route('/Homepage')
def Homepage():
    return render_template('home.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if (request.method == 'POST'):
            email = request.form['name']
            auth.send_password_reset_email(email)
            return render_template('index.html')
    return render_template('forgot_password.html')


@app.route('/getDisplayData')
def getDisplayData():
    print(displayData)
    return  jsonify(displayData)


@app.route('/home', methods=['GET', 'POST'])
def basic():

    # Remove already existing files in the output_frames directory :)
            counter1=0
            files_del = glob.glob('data/output_frames/*')
            for counter1 in files_del:
                os.remove(counter1)
    # Remove already existing files in the Clipped directory :)
            counter2=0
            files_clipped = glob.glob('data/Clipped/*')
            for counter2 in files_clipped:
                os.remove(counter2)
    # Request Video File to act like live stream
            f = request.files['file']
            print('FILENAME: ', f.filename)
            print('SECURE FILE NAME: ', f.save(secure_filename(f.filename)))
            times = []
            i=0
            h=0
            print(f.filename)
            try:
                vid = cv2.VideoCapture(f.filename)
            except:
                vid = cv2.VideoCapture(f.filename)
            out = None
            fps = 0.0
            count = 0
            while True:
                _, img = vid.read()
                if img is None:
                    logging.warning("Empty Frame")
                    time.sleep(0.1)
                    count+=1
                    if count < 3:
                        continue
                    else: 
                        break
                img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                img_in = tf.expand_dims(img_in, 0)
                img_in = transform_images(img_in,416)

                t1 = time.time()
                boxes, scores, classes, nums = yolo.predict(img_in)
                fps  = ( fps + (1./(time.time()-t1)) ) / 2

                img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                # Checking threshold i first row of 2D "scores array" because score array has only one row
                if(scores[0][1].any() >0.50):
                    cv2.imwrite('data/Clipped/clipp'+str(i)+'.jpg',img)
                    storage.child('ClippedCrash/' +str(i)+ '/crash.jpg').put('data/Clipped/clipp'+str(i)+'.jpg')
                    link_image = storage.child('ClippedCrash/' +str(i)+ '/crash.jpg').get_url(None)
                    doc_ref = db.collection(u'Crash')
                    doc_ref.add({
                        u'Name': u'Vehicle Crash',
                        u'Type': u'Anomaly',
                        u'Timestamp': str(datetime.now()),
                        u'Image_Url': link_image
                    })
                elif(scores[0][0].any() >0.50):
                    cv2.imwrite('data/Clipped/clipp'+str(i)+'.jpg',img)
                    # storage.child('LaneClipped/' +str(i)+ '/Lane.jpg').put('data/Clipped/clipp'+str(i)+'.jpg')
                    # link_image = storage.child('ClippedLane/' +str(i)+ '/Lane.jpg').get_url(None)
                    # doc_ref = db.collection(u'LaneVoilation')
                    # doc_ref.add({
                    #     u'Name': u'Lane Voilation',
                    #     u'Type': u'Anomaly',
                    #     u'Timestamp': datetime.now(),
                    #     u'Image Url': link_image
                    # })
                
                print(boxes, scores, classes, nums, class_names)
                global displayData
                displayData = {

                    "scores":str(scores),
                    "classes":str(classes),
                    "classes_names":str(class_names)
                }
                # print(displayData)
                # data['boxes'] = i
                img = cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                cv2.imwrite('data/output_frames/anomaly'+str(i)+'.jpg',img)
                i=i+1
                   
                cv2.destroyAllWindows()

            # os.remove('data\output-vid\short.mp4')
            vid_array = []
            for img_video in glob.glob('data/output_frames/*.jpg'):
                vid_img = cv2.imread(img_video)
                height, width, layers = vid_img.shape
                SIZE_vid = (width,height)
                vid_array.append(vid_img)
 
            out = cv2.VideoWriter('data/output-vid/short.mp4',cv2.VideoWriter_fourcc(*'mp4a'), 15,SIZE_vid)

            for n in range(len(vid_array)):
                out.write(vid_array[n])
            out.release()
            
            cv2.destroyAllWindows()

            users_ref = db.collection(u'Crash')
            Crashdata = users_ref.stream()
            sasta = []
            print(Crashdata)
            for doc in Crashdata:
                print(f'{doc.id} => {doc.to_dict()}')                   
                my_dict = doc.to_dict() 
                sasta.append(my_dict)
                # print(my_dict)
            print(sasta)
            storage.child("videos/new.mp4").put("data\output-vid\short.mp4")
            links = storage.child('videos/new.mp4').get_url(None)
            return render_template('upload.html', l=(links,sasta))
# #very slowavi
# @app.route('/load_data', methods=['GET'])
# def load_data():
#         return jsonify(data)

@app.route('/uploads', methods=['GET', 'POST'])
def uploads():
        if request.method == 'POST':
            return redirect(url_for('basic'))
        if True:
            links = storage.child('videos/new.mp4').get_url(None)
            return render_template('upload.html', l=links)
        return render_template('upload.html')

# @app.route('/webcam', methods=['GET', 'POST'])
# def webcam():
#     start_time = time.time()
#     # displays the frame rate every 2 second
#     display_time = 2
#     # Set primarry FPS to 0
#     fps = 0

#     # we create the video capture object cap
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise IOError("We cannot open webcam")

#     while True:
#         ret, frame = cap.read()
#         # resize our captured frame if we need
#         frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)

#         # # detect object on our frame
#         # r_image, ObjectsList = yolo.detect_img(frame)

#         # show us frame with detection
#         cv2.imshow("Web cam input", r_image)
#         if cv2.waitKey(25) & 0xFF == ord("q"):
#             cv2.destroyAllWindows()
#             break

#         # calculate FPS
#         fps += 1
#         TIME = time.time() - start_time
#         if TIME > display_time:
#             print("FPS:", fps / TIME)
#             fps = 0 
#             start_time = time.time()


#     cap.release()
#     cv2.destroyAllWindows()

@app.route('/detection_vid', methods= ['GET'])
def get_image():
    image = request.files["images"]
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))
    img_raw = tf.image.decode_image(
        open(image_name, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    print('detections:')
    for i in range(nums[0]):
        print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(output_path + 'detection.jpg', img)
    print('output saved to: {}'.format(output_path + 'detection.jpg'))
    
    # prepare image for response
    _, img_encoded = cv2.imencode('.png', img)
    response = img_encoded.tostring()
    
    #remove temporary image
    os.remove(image_name)

    try:
        return Response(response=response, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(404)

if __name__ == '__main__':
    app.run()

