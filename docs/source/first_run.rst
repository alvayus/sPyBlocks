=========
First run
=========

This brief first tutorial explains how to configure the library by installing its dependencies and then configuring it.

Python & Dependencies
---------------------

Firstly, you must ensure Python is installed in your machine. Although in principle the version of Python is not relevant, it is recommended to use the most current versions to avoid errors. Python 3.9 has been used to build sPyBlocks. 

On the other hand, when Python is already installed, other dependencies must be installed before starting to work with sPyBlocks. These dependencies are PyNN, sPyNNaker, Numpy, Matplotlib and XlsxWriter. PyNN 0.9.6, sPyNNaker 6.0.0, Numpy 1.22.1, Matplotlib 3.5.1 and XlsxWriter 3.0.2 have been used for library code and test design. Later versions should be also supported.

Configuring sPyNNaker
---------------------

Once all the dependencies are installed, it is necessary to configure sPyNNaker to make use of the hardware platform correctly. To do this, we recommend going to the configuration section of the SpiNNaker installation guide, which can be found at the following link: `PyNN on SpiNNaker Installation Guide <http://spinnakermanchester.github.io/spynnaker/5.0.0/PyNNOnSpinnakerInstall.html>`_.

After configuring sPyNNaker, you should be able to launch any of the tests contained in the sPyBlocks GitHub repository.