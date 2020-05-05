from flask import Flask, render_template, request
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from model_nst import load_image, model, stylize, im_convert

app = Flask(__name__)
UPLOAD_FOLDER = './static/image/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
style = "" # initialize style name (cf. line 26)


@app.route("/") # root of the app (home)

def home(): # this is a view function run when the route specified above is requested by user
	return render_template("index_nst.html") # flask function to display page based on html template


@app.route("/success", methods=['POST']) # sets the URL name and the methods used on this page

def upload_file():		
	content = request.files['file']
	style = request.form.get('style') # get style name from the pre-loaded files in the form, not from user files
		
	content.save(os.path.join(app.config['UPLOAD_FOLDER'], 'content.jpg'))
		
	content = load_image('./static/images/upload/content.jpg') # load_image is from model_nst.py
	style = load_image('./static/images/s'+ style+'.jpg', shape=content.shape[-2:]) # resize style to match content

	vgg = model() # model is from model_nst.py
	target = stylize(content, style, vgg) # stylize is from model_nst.py
	x = im_convert(target) # imconvert is from model_nst.py
	plt.imsave(app.config['UPLOAD_FOLDER']+'/target.png', x) # imsave is a plt function

	return render_template('success_nst.html')


if __name__ =="__main__": # the app will run only if it's called as main, i.e. not if you import app_nst in another code
	app.run(debug=True) # run starts the development (local) server

# Need to accelerate computation:
# could we serialize a pre-trained model with pickle and charge it on top of this module?
# could we have it run on a GPU? Colab, databricks, other?