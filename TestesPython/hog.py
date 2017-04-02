# script para passar todas as imagens no filtro
# HOG e escreve-las em uma pasta na saida

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.feature import hog
from skimage import color
from skimage.filters import roberts

# neste caso utiliza-se o detector de bordas Robert
# neste site possuem outros algoritmos de deteccao de borda:
# http://scikit-image.org/docs/dev/auto_examples/edges/plot_edge_filter.html

A = color.rgb2gray(imread("train_5a_00007.png"))
a1 = roberts(A)
v, B = hog(a1,orientations=8, pixels_per_cell=(16, 16),
	cells_per_block=(1, 1), visualise=True)

imshow(A)
plt.show()
imshow(a1)
plt.show()
imshow(B)
plt.show()

# TODO: o que falta: pegar a matriz v e escrever
# em um txt e ver o que fazer com a imagem B

print("Imagem final em array:")
print(v)
