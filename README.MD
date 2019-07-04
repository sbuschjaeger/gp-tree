#Gaussian Model Trees for Regression

This repository contains the experiments for the paper "Gaussian Model Trees for Regression" by Sebastian Buschjäger, Thomas Liebig and Katharina Morik. If you have any questions please contact me directly

Sebastian Buschjäger (http://www-ai.cs.tu-dortmund.de/PERSONAL/buschjaeger.html) using sebastian.buschjaeger@tu-dortmund.de

In order to run the experiments you will need the following tools:
- git
- cmake>=3.8
- A compiler for C++17 code (The code is tested with g++ 7 and 8)
- make
- For GMT-NN you will need to have PyTorch with C++ bindings (https://pytorch.org/cppdocs/)
- For faster matrix operations you can use MKL (https://software.intel.com/en-us/mkl)

In order to plot the results you will need the following tools:
- python3
- python numpy
- python pandas
- python plotly incl. plotly-dash, plotly-io (with orca, see https://plot.ly/python/static-image-export/)

This repository is structured as the following:
- submodules/ensembles: This contains the actual source code for training GP, IVM, GMT and GMT-NN. I try to maintain this code independent from specific experiments / paper in its own repository (https://bitbucket.org/sbuschjaeger/ensembles/). Please make sure, that you actually clone this repository as well, e.g. by doing `git submodule update --init --recursive`
Important: These experiments have been run with code from a different branch than master. Thus please make sure that you checkout the branch `ensembles-classification` of the ensembles project, e.g. 
`cd submodules/ensembles && git checkout ensembles-classification` (Yes I know the name is somewhat misleading. Stuff happend and mistakes have been made.)
- Luxembourg/Housing/Crime: These folder contain the actual experiments. You will find three files. 
	- CMakeLists.txt: This file contains the compiler configurations for the experiments. There are four options provided for CMake:
		- USE_NATIVE=ON/OFF to compile with -march and -mtune flag
		- USE_OPENMP=ON/OFF to compile with openmp support. Note that this will also enable parallel STL algorithms if available.
		- USE_MKL=ON/OFF to compile and link with Intel MKL. Please note that the search path is currently hard-coded to "/opt/intel/compilers_and_libraries/linux/mkl". If you have installed MKL under a different path you should update "set(MY_MKL_ROOT "/opt/intel/compilers_and_libraries/linux/mkl")" in line 65.
		- USE_TORCH=ON/OFF to compile with PyTorch support. Please note, that the search path is currently hard-coded to "/opt/libtorch/intel-nightly". If you have installed MKL under a different path you should update "set(CMAKE_PREFIX_PATH "/opt/libtorch/intel-nightly")" in line 80. Also note, that depending on your MKL and PyTorch versions these two libs may not go well together (I re-compiled PyTorch with my specific MKL version).
	- main.cpp: This file contains the actual experiments including the grid search. 
	- main.py: This file prepares the plots. To do so, it starts a dash app which invokes a flask app, so you can browse the plots interactivley in your browser. Also, it produced the PDF files used in the paper. Note that it assumes that the results are placed in "build/xval.csv" (line 204).
	
Given you have everything installed you can do

Switch in the correct folder
`cd luxembourg`

Prepare build folder
`mkdir build && cd build`

Prepare makefiles folder
`cmake -DUSE_OPENMP=ON -DUSE_MKL=ON -DUSE_TORCH=ON -DUSE_NATIVE=ON ..`

Compile
`make`

Run 
`./main`

Plot the results
`cd .. && ./main.py`