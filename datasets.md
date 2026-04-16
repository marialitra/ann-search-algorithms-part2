## Datasets

### MNIST Dataset

The `input.dat` file for the MNIST dataset is a **Big-Endian binary file** with the following structure:

| Offset  | Type            | Value              | Description         |
|---------|------------------|--------------------|---------------------|
| 0x0000  | 32-bit integer   | 0x00000803 (2051)  | Magic number        |
| 0x0004  | 32-bit integer   | 60000              | Number of images    |
| 0x0008  | 32-bit integer   | 28                 | Number of rows      |
| 0x000C  | 32-bit integer   | 28                 | Number of columns   |
| 0x0010  | Unsigned byte    | ??                 | Pixel no. 1         |
| 0x0011  | Unsigned byte    | ??                 | Pixel no. 2         |
| ...     | ...              | ...                | ...                 |
| 0x031F  | Unsigned byte    | ??                 | Pixel no. 784       |

This file corresponds to the **MNIST** dataset, which contains images of handwritten digits.  
Each image has dimensions **28 × 28 pixels**, and each pixel takes an integer value from **0 to 255**.

The vector corresponding to each image is formed by concatenating its rows, resulting in a vector of dimension **784**.

Dataset source:
- [MNIST original dataset files](https://github.com/mrgloom/MNIST-dataset-in-different-formats/tree/master/data/Original%20dataset)

The MNIST dataset has already been uploaded in the [`Data/MNIST`](Data/MNIST) folder, so no additional download is required.

---

### SIFT Dataset

The `input.dat` file for the SIFT dataset is a **Little-Endian binary file** with the following structure:

| Offset  | Type          | Value | Description                    |
|---------|---------------|-------|--------------------------------|
| 0x0000  | 32-bit integer| 128   | Dimension of vector            |
| 0x0004  | 32-bit float  | ??    | Coordinate 001 of 1st vector   |
| 0x0008  | 32-bit float  | ??    | Coordinate 002 of 1st vector   |
| ...     | ...           | ...   | ...                            |
| 0x0200  | 32-bit float  | ??    | Coordinate 128 of 1st vector   |
| 0x0204  | 32-bit integer| 128   | Dimension of 2nd vector        |
| 0x0208  | 32-bit float  | ??    | Coordinate 001 of 2nd vector   |
| ...     | ...           | ...   | ...                            |

This file corresponds to the **SIFT1M** dataset, which contains **1 million SIFT (Scale-Invariant Feature Transform) feature vectors**.

Each vector:
- has dimension **128**
- is stored in **float32**
- describes features extracted from real images

The file `sift_base.fvecs` contains a continuous sequence of vectors.  
Each vector follows the same repeated pattern of **516 bytes**:
- **4 bytes** for the dimension (`128`)
- **128 × 4 bytes** for the float coordinates

Dataset source:  
- [SIFT1M dataset](http://corpus-texmex.irisa.fr/)
- File used: `sift_base.fvecs` from **ANN_SIFT1M**

The SIFT1M dataset is too large to be uploaded to GitHub. To run the code with SIFT data, please download the required files manually, then create a folder named `SIFT` inside the [`Data`](Data) directory, similar to the existing `MNIST` folder, and place the downloaded files there.
