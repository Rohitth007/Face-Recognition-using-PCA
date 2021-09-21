import os
import shutil

source = "/home/rohitth007/Documents/MOOCs/Computer Vision/Prof.Rajagopalan Labs/Face Recognition using PCA/Dataset"
destination = "/home/rohitth007/Documents/MOOCs/Computer Vision/Prof.Rajagopalan Labs/Face Recognition using PCA/Testing"
os.system(f"mkdir '{destination}'")

folders = os.listdir(source)
for folder in folders:
    os.system(f"mkdir '{destination}/{folder}'")
    for file in os.listdir(source + "/" + folder)[:10]:
        shutil.move(source + "/" + folder + "/" + file, destination + "/" + folder)
