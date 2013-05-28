import image
import cifar
import auto_encoder

data = cifar.load()
patches = image.prepare_patches(data, 8, 10000)
patches = image.normalize(patches)

ae = auto_encoder.AutoEncoder(3*8**2,200)
ae.cost(patches)
