[Back to Overview](../../../README.md)



# leanai.model.models.imagenet

> An implemntation of various imagenet models.


---
---
## *class* **ImagenetModel**(Module)

Create one of the iconic image net models in one line.
Allows for only using the encoder part.

This model assumes the input image to be 0-255 (8 bit integer) with 3 channels.

* **encoder_type**: The encoder type that should be used. Must be in ("vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "resnet50", "resnet101", "resnet152", "densenet121", "densenet169", "densenet201", "inception_v3", "mobilenet_v2")
* **only_encoder**: Leaves out the classification head for VGG16 leaving you with a feature encoder.
* **pretrained**: If you want imagenet weights for this network.
* **last_layer**: Index of the last layer in the encoder. Allows to cutoff encoder after a few layers.
* **standardize**: (bool) If this is enabled the module will normalize a 0-255 uint image into a float32 image which is normalized and standardized, as with the magic values in pytorch. (Makes them compatible with the imagenet models.)
* **to_channel_first**: (bool) It this is enabled, the image will be converted from h,w,c into the c,h,w format which pytorch uses.


---
### *def* **build**(*self*, image)

*(no documentation found)*

---
### *def* **forward**(*self*, image)

*(no documentation found)*

