import numpy as np
import tkinter as tk

from tkinter import filedialog
from tkinter import *

from PIL import ImageTk, Image

from tensorflow.keras.models import load_model


def classify(file_path):
    global label_packed
    img = Image.open(file_path)
    img = img.resize((128, 128))
    img = np.expand_dims(img, axis=0)
    img = np.array(img)
    img = img / 255.0

    pred = model.predict_classes([img])[0]
    sign = classes[pred]

    print(sign)
    label.configure(foreground='#011638', text=sign)


def show_classify_button(file_path):
    button = Button(top, text="Classify Image",
                    command = lambda: classify(file_path), 
                    padx=10, pady=5)
    button.configure(background='#364156', foreground='black',
                    font=('arial', 10, 'bold'))
    button.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        img = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=img)
        sign_image.image = img
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

# 95.65% accuracy
model = load_model('model.h5')

classes = {
    0: 'It\'s a cat',
    1: 'It\'s a dog',
}

top = tk.Tk()
top.geometry('800x600')
top.title('Dog or Cat Classifier')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='black', font=('arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(top, text='Dog or Cat Classifier', pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()



