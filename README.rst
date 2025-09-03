diffhtwo
============

Data
----
Data for this project can be found at the following URL:

https://portal.nersc.gov/project/hacc/aphearin/diffhtwo_data/

Installation
------------
To install diffhtwo into your environment from the source code::

    $ cd /path/to/root/diffhtwo
    $ pip install -e .

Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/diffhtwo
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html

