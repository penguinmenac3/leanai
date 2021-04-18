[Back to Overview](../../../README.md)



# leanai.model.layers.image_conversion

> A simple fully connected layer (aka Linear Layer or Dense).


---
---
## *class* **ImageConversion**(Module)

This layer takes care of image conversion for models.

* **standardize**: (bool) If this is enabled the module will normalize a 0-255 uint image into a float32 image which is normalized and standardized, as with the magic values in pytorch. (Makes them compatible with the imagenet models.)
* **to_channel_first**: (bool) It this is enabled, the image will be converted from h,w,c into the c,h,w format which pytorch uses.


---
### *def* **build**(*self*, image)

*(no documentation found)*

---
### *def* **forward**(*self*, image)

*(no documentation found)*

