# What is Hystorian?

Keeping track of the post-treatement done on our data can sometimes be cumbersome.

Hystorian is a project born from our AFM group to solve this issue. It is a python package with two main functionalities:

1. It handles multiple file format and converted all of them into uniformized hdf5 files. 
2. It keeps track of the processing done on the data. 

Find the link to the paper here: http://dx.doi.org/10.1016/j.ultramic.2021.113345

We would be gratefull if you cited it if it was usefull for your research

You can install the package using pip, you can find it here https://pypi.org/project/hystorian/.

Simply run: ```pip install hystorian```

## Structure of an hdf5 file
An hdf5 file has a tree-like structure consisting of groups (folders) and datasets. Each groups and datasets can have attributes attached to them. An attribute is a small metadata containing informations about the corresponding dataset or group. Here we use them to store general information about the data or parameters of the post-processing.

The tree-structure we use in Hystorian is the following :
1. A dataset group, containing all the raw data you want to regroup inside a single hdf5.
2. A metadata group, containing the metadatas of the initial files.
3. A process group, containing the data processed from the original data, or from other processed data.

## m_apply

Our goal was to streamline the process and make the usage of Hystorian as seamless as possible for people already using Python. The key to this is the [`m_apply`](/Functions/Core/m_apply) function. It allows the application of any usual python function working on array on a datasets inside an hdf5 and to write the results into the same hdf5, while keeping all the key arguments of the function stored with the datasets inside the hdf5 file. Tutorials below show you how to use it in real case examples.

## Tutorials

A few tutorial were written by R.Bulanadi and can be found in the examples folder of the project, or in the links below:

* [Basic](https://gitlab.unige.ch/paruch-group/hystorian/-/wikis/tutorial/basic)
* [Intermediate](https://gitlab.unige.ch/paruch-group/hystorian/-/wikis/tutorial/intermediate)
* [Programming](https://gitlab.unige.ch/paruch-group/hystorian/-/wikis/tutorial/programming)
