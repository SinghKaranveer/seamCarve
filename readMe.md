# Seam Carve


## Getting Started

In order to run this code, there are a few prerequisites.  Firstly, this code uses the OpenCV C++ library, so that needs to be installed.  Also, POSIX pthreads are used, therefore this will only run on MacOS, Linux, or any other POSIX compliant operating systems.  A makefile as been included, so to build the executable simply run ```make``` and it should compile.  If you do not wish to compile, a pre-compiled executable ```seamCarve.out``` is also provided.

### Running

In order to run the seam carve, once the executable has been built, run it with the format:
```./seamCarve.out [path to image] [HORIZONTAL or VERTICAL] [Number of seams to remove]```
The repo includes a few test files, so for example, ```./seamCarve.out test.jpg VERTICAL 200``` will remove 200 vertical seams from test.jpg.  The resulting image will be placed into test_output.jpg

#### Example Images

Before:
![alt text](https://i.imgur.com/N3ELxXH.jpg)

After:
![alt text](https://i.imgur.com/GOnQirR.jpg)


Before:
![alt text](https://i.imgur.com/mKVZELZ.jpg)

After:
![alt text](https://i.imgur.com/j61h29I.jpg)
