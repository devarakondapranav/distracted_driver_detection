from flask import Flask, render_template, request
from  tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
import tensorflow as tf

from werkzeug.utils import secure_filename
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

new_model = load_model('vgg_lrely_22kimgs.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})

UPLOAD_FOLDER = 'static'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def get_parsed_output(output):
    l = list(output[0])
    top3 = []
    for i in list(reversed(sorted(l)))[:3]:
        top3.append(l.index(i))

    m = {0: "safe driving",
    1: "texting - right",
    2: "talking on the phone - right",
    3: "texting - left",
    4: "talking on the phone - left",
    5: "operating the radio",
    6: "drinking",
    7: "reaching behind",
    8: "hair and makeup",
    9: "talking to passenger"}

    arr = []
    for i in top3:
        arr.append(str(m[i]))
    if(arr[0] == 'safe driving'):
    	arr.insert(0, "Not distracted")
    else:
    	arr.insert(0, "Distracted")
    return arr


@app.route('/', methods=['POST', 'GET'])
def hello_world():

	if(request.method == "POST"):

		form_data = request.form

		
		file =request.files['driverImage']
		filename = secure_filename(file.filename) # save file 
		filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename);
		file.save(filepath)


		# img = request.files["driverImage"].read()
		# img = np.fromstring(img, np.uint8)
		# img = cv2.imdecode(img,cv2.IMREAD_COLOR)

		#img = cv2.imread(filepath)
		img = image.load_img("C:\\Users\\PranavDevarakonda\\Documents\\project_papers\\testing\\static\\"+filename,target_size=(224,224))
		extra = []
		#extra.append(img.shape)
		#img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
		#extra.append(img.shape)
		

		img = np.asarray(img)
		extra.append(img.shape)
		img = np.expand_dims(img, axis=0)
		extra.append(img.shape)
		img = img.astype('float32')
		output = new_model.predict(img)
		extra.append(output)

		predictions = get_parsed_output(output)



		return render_template("index.html", predictions=predictions, extra=extra, img_path=filepath)
	else:
		return render_template("index.html", predictions = [])

 

if __name__ == '__main__':
   app.run("0.0.0.0", 5000, debug=True)