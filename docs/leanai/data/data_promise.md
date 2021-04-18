[Back to Overview](../../README.md)



# leanai.data.data_promise

> The promise object for data can be used to abstract how the data is loaded and only load it lazy.


---
---
## *class* **DataPromise:*

*(no documentation found)*

---
### *def* **data**(*self*) -> bytes

Get the data as raw bytes.

Must be implemented by all subclasses.


---
---
## *class* **DataPromiseFromFile**(DataPromise)

A promise on the data in a file.
Only loads the data on access and buffers it then.


---
### *def* **data**(*self*) -> bytes

*(no documentation found)*

---
---
## *class* **DataPromiseFromBytes**(DataPromise)

A promise on bytes that are already loaded.

Can be usefull when promising data from a tar stream.


---
### *def* **data**(*self*) -> bytes

*(no documentation found)*

