from PyQt5 import QtCore, QtGui, QtWidgets
import random
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import keras
import os

def load_image(img_path):
    img = image.load_img(img_path, target_size=(100, 100))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.                                      
    return img_tensor

def activation(number):
    return "Uninfected" if (number >= 0.5) else "Parasitized"

model = load_model("model.h5")

def predict(path):
    return activation(model.predict(load_image(path))[0][0])


class Ui_MainWindow(object):
   

    def getImage(self):
        self.graphicsScene.clear()
        classes = ["Parasitized","Uninfected"]
        choice = random.randint(0,1)
        path = os.path.join(os.getcwd(),"cell_images/test/" + classes[choice])
        images = os.listdir(path)
        image_path = os.path.join(path,images[random.randint(0,len(images)-1)])
        self.pixMap = QtGui.QPixmap(image_path)
        self.pixMapItem = QtWidgets.QGraphicsPixmapItem(self.pixMap)
        self.graphicsScene.addItem(self.pixMapItem)
        self.graphicsView.show()
        
        classes = ["Parasitized","Uninfected"]
        choice = random.randint(0,1)
        path = os.path.join(os.getcwd(),"cell_images/test/" + classes[choice])
        images = os.listdir(path)
        image_path = os.path.join(path,images[random.randint(0,len(images)-1)])

        prediction = predict(image_path)

        self.label.setText("Predicted: " + prediction)
        isCorrect = "Yes" if (prediction == classes[choice]) else "No"
        self.label_2.setText("Correct: " + isCorrect)

    def buttonEvent(self):
        self.getImage()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(426, 400)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.graphicsScene = QtWidgets.QGraphicsScene()

        self.graphicsView = QtWidgets.QGraphicsView(self.graphicsScene,self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(30, 50, 361, 231))
        self.graphicsView.setObjectName("graphicsView")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(113, 10, 200, 30))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.buttonEvent)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(150, 300, 200, 20))
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(163, 340, 100, 20))
        self.label_2.setObjectName("label_2")

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 426, 18))
        self.menubar.setObjectName("menubar")

        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")

        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.getImage()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Malaria Identification"))
        self.pushButton.setText(_translate("MainWindow", "Get A Sample Image!"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    MainWindow.show()

    sys.exit(app.exec_())
