import Network as NWRK
from PIL import Image
import numpy as np

md = NWRK.Network()
md.load("./model.json")

im = np.asarray(Image.open("/Users/bracho/Downloads/pixil-frame-0 (5).png")).reshape((784,4))
im = np.array([t[3]/255 for t in im]).reshape((784, 1))

print(np.argmax(md.predict(im)))
