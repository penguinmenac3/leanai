[Back to Overview](../../../README.md)



# leanai.model.models.detector

> An generic implementation of a detector.


---
---
## *class* **DetectionModel**(nn.Module)

Create a detector with a common structure.

inputs->backbone(->neck)->dense_head(->roi_head)->outputs

Returns the outputs from dense and roi head. If both are present a tuple is returned (dense, roi).


---
### *def* **forward**(*self*, inputs)

*(no documentation found)*

