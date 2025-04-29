from PIL import Image
import numpy as np

X = np.empty((400, 92*112))

for subject in range(40) :

    base_path = f"D:\\data_sets\\AT&T Database of Faces\\s{subject + 1}\\"

    for image in range(10) : 
        
        img_path = base_path + f"{image + 1}.pgm"

        img = Image.open(img_path)

        img_pixel = np.array(img).flatten()

        # print(subject * 10 + image)

        X[subject * 10 + image] = img_pixel

import pandas as pd

df = pd.DataFrame(X, columns = [f"pixel{i}" for i in range(1, 10305)])

df.to_csv("D:\\Users\\Downloads\\AT&T_Database_of_Faces.csv", index = False)