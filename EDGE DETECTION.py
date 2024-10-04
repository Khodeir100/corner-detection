import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QPushButton, QGraphicsView, QGraphicsScene, QSlider, QLabel
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np

class ImageViewer(QDialog):
    def __init__(self) -> None:
        """Initialize the Image Viewer"""
        super(ImageViewer, self).__init__()
        uic.loadUi('D:/Computer vision/3/P/GUI.ui', self)  # Load the UI file

        # Find the buttons, graphics views, and sliders from the UI
        self.load_button = self.findChild(QPushButton, 'pushButton')  # Button for loading images
        self.detect_button = self.findChild(QPushButton, 'pushButton_2')  # Button for Harris detection
        self.sift_button = self.findChild(QPushButton, 'pushButton_4')  # Button for SIFT edge detection
        self.image_viewer = self.findChild(QGraphicsView, 'graphicsView')  # QGraphicsView for displaying images
        self.harris_viewer = self.findChild(QGraphicsView, 'graphicsView_2')  # QGraphicsView for Harris results
        self.sift_viewer = self.findChild(QGraphicsView, 'graphicsView_3')  # QGraphicsView for SIFT results
        self.slider = self.findChild(QSlider, 'horizontalSlider')  # QSlider for Harris parameters
        self.sift_slider = self.findChild(QSlider, 'horizontalSlider_2')  # QSlider for SIFT parameters
        self.slider_value_label = self.findChild(QLabel, 'sliderValueLabel1')  # Label to show Harris slider value
        self.sift_slider_value_label = self.findChild(QLabel, 'sliderValueLabel1_2')  # Label to show SIFT slider value

        # Create QGraphicsScenes
        self.scene = QGraphicsScene(self)
        self.result_scene = QGraphicsScene(self)
        self.sift_scene = QGraphicsScene(self)
        self.image_viewer.setScene(self.scene)
        self.harris_viewer.setScene(self.result_scene)
        self.sift_viewer.setScene(self.sift_scene)

        # Connect the buttons and sliders to their respective functions
        if self.load_button:
            self.load_button.clicked.connect(self.openFileDialog)
        if self.detect_button:
            self.detect_button.clicked.connect(self.apply_harris_detection)
        if self.sift_button:
            self.sift_button.clicked.connect(self.apply_sift_detection)
        if self.slider:
            self.slider.valueChanged.connect(self.update_slider_value)
        if self.sift_slider:
            self.sift_slider.valueChanged.connect(self.update_sift_slider_value)

        self.threshold = 50  # Default threshold for corner detection
        self.img = None  # To store the loaded image
        self.original_img = None  # To store the original image

    def openFileDialog(self):
        """Open file dialog to select an image"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                    "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        if file_name:
            self.img = cv2.imread(file_name)
            self.original_img = self.img.copy()  # Keep a copy of the original image
            if self.img is not None:
                self.displayImage(self.img)

    def displayImage(self, img):
        """Display the image in the QGraphicsView"""
        if len(img.shape) == 2:  # If the image is grayscale
            height, width = img.shape
            bytes_per_line = width
            qImg = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  # If the image is color
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            qImg = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Create a QPixmap and add it to the QGraphicsScene
        pixmap = QPixmap.fromImage(qImg)
        self.scene.clear()  # Clear the scene before adding a new image
        self.scene.addPixmap(pixmap)  # Add the QPixmap to the scene
        self.image_viewer.setScene(self.scene)  # Update the QGraphicsView

    def apply_harris_detection(self):
        """Apply Harris corner detection and display the result only in graphicsView_2"""
        if hasattr(self, 'original_img') and self.original_img is not None:
            gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray, 2, 3, 0.04)

            # Result is dilated for marking the corners
            dst = cv2.dilate(dst, None)

            # Use the threshold from the slider
            self.threshold = self.slider.value() / 1000.0  # Convert to a fraction
            img_with_corners = self.original_img.copy()  # Create a copy of the original image
            img_with_corners[dst > self.threshold * dst.max()] = [0, 0, 255]  # Mark corners in red

            # Convert and display in the Harris view only
            result_pixmap = QPixmap.fromImage(QImage(img_with_corners.data, img_with_corners.shape[1], img_with_corners.shape[0], 3 * img_with_corners.shape[1], QImage.Format_RGB888).rgbSwapped())
            self.result_scene.clear()
            self.result_scene.addPixmap(result_pixmap)
            self.harris_viewer.setScene(self.result_scene)  # Update the Harris QGraphicsView

    def apply_sift_detection(self):
        """Apply SIFT edge detection and display the result without modifying original image"""
        if hasattr(self, 'original_img') and self.original_img is not None:
            gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()

            # Adjust the number of keypoints with the slider value
            nfeatures = self.sift_slider.value() * 10
            sift.setNFeatures(nfeatures)

            keypoints, descriptors = sift.detectAndCompute(gray, None)
            img_with_keypoints = cv2.drawKeypoints(self.original_img.copy(), keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # Create a copy

            # Display the SIFT result
            self.displaySIFTImage(img_with_keypoints)

    def displaySIFTImage(self, img):
        """Display the SIFT image in the QGraphicsView"""
        qImg = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888).rgbSwapped()
        sift_pixmap = QPixmap.fromImage(qImg)
        self.sift_scene.clear()  # Clear the scene before adding a new image
        self.sift_scene.addPixmap(sift_pixmap)  # Add the QPixmap to the scene
        self.sift_viewer.setScene(self.sift_scene)  # Update the QGraphicsView

    def update_slider_value(self, value):
        """Update the slider value label and reapply detection"""
        self.slider_value_label.setText(f"Threshold: {value / 10.0:.1f}")  # Display as a percentage
        self.apply_harris_detection()  # Reapply detection on slider change

    def update_sift_slider_value(self, value):
        """Update the SIFT slider value label"""
        self.sift_slider_value_label.setText(f"SIFT Features: {value}")  # Display the SIFT features count
        self.apply_sift_detection()  # Reapply SIFT detection on slider change

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())