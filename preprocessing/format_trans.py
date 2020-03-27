import numpy as np
from PIL import Image
import os
import os.path



a = os.listdir('Messidor DB__')
print(a[0])

for infile in os.listdir("./Messidor DB__"):
    print("file: \t\t\t" +infile)
    if infile[-3:] == "tif":
        outfile = infile[:-3] + "png"
        img = Image.open("./Messidor DB__/"+infile)
        print("new filename: \t" +outfile)
        out = img.convert("RGB")
        out.save(outfile, "png")