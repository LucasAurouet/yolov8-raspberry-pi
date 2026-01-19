import cv2
from picamera2 import Picamera2
import time

picam2 = Picamera2()

# Configuration simple
picam2.configure(picam2.create_still_configuration())

print("Démarrage de la caméra...")
picam2.start()

time.sleep(2)  # laisse le temps au capteur de démarrer

# Capture une image
picam2.capture_file("test_camera.jpg")

picam2.stop()
print("✅ Image capturée : test_camera.jpg")

