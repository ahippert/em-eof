
# The emeof package

Interpolation of geophysical/displacement fields as described in:

[_EM-EOF: gap-flling in incomplete SAR displacement time series._](https://ieeexplore.ieee.org/abstract/document/9170898)
<br/>
Hippert-Ferrer A., Yan Y., Bolon P. 
<br/>
`IEEE Transactions on Geoscience and Remote Sensing`, **2020**.
<br/>
\[<a href="ahippert.github.io/pdfs/tgrs_2020.pdf" target="_blank">PDF</a>\] \[<a href="https://github.com/ahippert/em-eof" target="_blank">Code</a>\]

## Download
You can `git clone` this or directly download the ZIP file under the **Code** button.

`$ git clone https://github.com/ahippert/em-eof`

## Test the code
You will need to install the `pytest` package facility first.
To run the tests, go under the parent directory and run:

`$ pytest --pyargs emeof`

This should plot simulated displacement fields and their reconstruction. Some of these plots were used in (Hippert-Ferrer et al., 2020).

## Play with code
Open emeof/main.py and run it from the command line or your favorite IDE. The use of the [_Python Debugger_](https://docs.python.org/3/library/pdb.html) is recommanded, as for example:

`$ python -m pdb main.py`
