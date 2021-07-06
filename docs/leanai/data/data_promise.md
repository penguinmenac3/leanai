[Back to Overview](../../README.md)



# leanai.data.data_promise

> The promise object for data can be used to abstract how the data is loaded and only load it lazy.



---
---
## *class* **DataPromise**(object)

The interface for a datapromise can be used when data is expected, but the user
does not care if the data is loaded lazy or ahead of time.


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

Get the data as raw bytes.

Lazy loads the data and buffers it for future access.


---
---
## *class* **DataPromiseFromBytes**(DataPromise)

A promise on bytes that are already loaded.

Can be usefull when promising data from a tar stream.


---
### *def* **data**(*self*) -> bytes

Get the data as raw bytes.

Uses the bytes provided in the constructor.


