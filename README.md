# Multiscale

Multiscale is a large scale project. Which has the motivation to gatherer all the snippets of code currently in the group under an organised library.

The project is based on the fact that, at the core our datas are only 1D, 2D, ND array, such that all the functions developped independently, could in theory be used for a lot of our datas.

The first part of the project is to create a file convert, which will convert most of the file type we use, into a well defined hdf5 format. Such that any data processing would be done using this format, for inter-compatibility.