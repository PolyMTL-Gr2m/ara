from keras2c import k2c
from tensorflow import keras
from huggingface_hub import from_pretrained_keras

model = from_pretrained_keras("SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net")


function_name = 'zanga'
k2c(model, function_name, malloc=False, num_tests=10, verbose=True)
