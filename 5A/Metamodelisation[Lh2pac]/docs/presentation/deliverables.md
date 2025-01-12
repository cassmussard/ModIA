# Deliverables

## Code

A series of Python scripts containing:

- Python code,
- documentation.

These scripts must be placed in the directory **docs/scripts/project**.

Moreover, they must be properly formatted
in order to be correctly compiled when using `mkdocs build` and `mkdocs serve`
(or `mkdocs.exe build` and `mkdocs.exe serve` under windows):

* the header docstring will be converted into an HTML block,
* the Python comments starting with `#%%` will be converted into HTML blocks,
* the Python code will be executed,
* the text resulting from the method `print`
  and the figure resulting from the Matplotlib `show` method
  will benefit from a special rendering.

!!! note

    You do not need to compile the project every time you change the Python script.
    You can use this Python script like any other Python script
    and compile the project only after the script writing is finished.
    When the project is compiled,
    only the scripts that have evolved are recompiled.

Here is an example of a well-formatted and documented Python code:

```

   r"""
   # Sum function

   In this example,
   we implement a function summing the elements of a vector $x\in\mathbb{R}^d$:

   $$f(x)=\sum_{i=1}^d x_i$$
   """
   from numpy import array

   #%%
   # Firstly,
   # we define the function:
   def f(x):
       return x.sum()

   #%%
   # Then,
   # we evaluate this function from the input vector $x=[1,2]$:
   y = f(array([1.,2.]))

   #%%
   # Lastly,
   # we print the output value:
   print(y)

   # Contrarily to the previous ones,
   # this comment will not be converted into an HTML block
   # because it does not start with #%%.
   # It will appear as a Python comment.
```

## Report

A markdown-based HTML report containing:

* an introduction,
* a section for each problem, ending with a synthesis.
* a general conclusion.

The syntheses and the general conclusion summarize the main facts
for someone who does not want to read the details.

Furthermore, all the results provided must be interpreted.

The markdown files must be placed in the directory **lh2pac/docs/report**
and referenced in the **nav/Report** section of the *mkdocs.yml* file.