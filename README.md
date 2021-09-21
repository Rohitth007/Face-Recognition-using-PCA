# Face-Recognition-using-PCA

This project uses the [faces94](https://cmp.felk.cvut.cz/~spacelib/faces/faces94.html) database to perform
Face Recognition for 45 people using **Principle Component Analysis**.

Prinicple Component Analysis, also known as, **Karhunen Loeve Transform (KLT)** or **Hotelling Transform** is
a data dependent transform which represents a stochastic process as a linear combination of orthogonal vectors
such that it minimizes error and maximizes variance. Hence, this is often used for decorrelating data
as the Covariance Matrix can be diagonalized to get a **Decorrelation Efficiency** of 100%,
courtesy **Spectram Theorem for Normal Matrices**.

Turns out the rows of this unitary transform happens to be the eigenvectors of the Covariance matrix, in this case
eigenfaces which look quite scary. This can be found using the **SVD** of **X/√n**, where **X.X^T/n** is the Covariance Matrix.

<img src='https://user-images.githubusercontent.com/64144419/134182819-73ad942f-0182-4b6a-a930-cc8018535f36.png' height=80>  <img src='https://user-images.githubusercontent.com/64144419/134182842-8c1e224f-420d-4519-a821-6ce30c3ee7b8.png' height=80>  <img src='https://user-images.githubusercontent.com/64144419/134182853-d7f48e04-ff39-419d-9b3d-71b8556b3ae7.png' height=80>  <img src='https://user-images.githubusercontent.com/64144419/134183185-393ee2e9-70b7-44dd-8165-a2db14ae36c5.png' height=80>

PCA really helps in **dimentionality reduction** as we can keep only those eigenfaces that contribute more to
the data.

| Original | 100 PCs | 50 PCs | 25PCs | 10PCs |
| :------: | :-----: | :----: | :---: | :---: |
| <img src='https://user-images.githubusercontent.com/64144419/134184414-96dd047b-b55d-467b-a80b-634ba879bf72.png' width=80> | <img src='https://user-images.githubusercontent.com/64144419/134188538-e2fceff7-0529-4552-87f4-f0e81cdd8498.png' width=80> | <img src='https://user-images.githubusercontent.com/64144419/134187339-571fefe5-fbcd-47a8-822f-e585c2364836.png' width=80> | <img src='https://user-images.githubusercontent.com/64144419/134186535-209968b0-f896-4e03-9cbe-95f3fe450325.png' width=80> | <img src='https://user-images.githubusercontent.com/64144419/134183459-a7cbbbef-d381-49e0-8f31-ae3632153d9c.png' width=80> |

**Energy Packing Efficiency** ( EPE, Cumulative Variance Ratio ) or **Explained Variance Ratio** ( Energy Fraction per Principle Component )
can be useful in deciding which dimension to use.

<img src='https://user-images.githubusercontent.com/64144419/134182227-83fc749f-63c7-45d6-9561-405d5dd795e5.png' width=290>  <img src='https://user-images.githubusercontent.com/64144419/134182272-811450b3-1ae6-43e4-ad3f-e0aaada0e121.png' width=300> <img src='https://user-images.githubusercontent.com/64144419/134182305-d4143b92-991c-4b0a-9482-ac1f8d1327c9.png' width=290>

The model was trained using 10 images for each person by creating an "Average Face" of each person in the
reduced eigen space. These along with the "Average Human Face" and the top x eigenfaces are stored in memory to recognize
any new test image given from the same class.


| Input Face | Recognized Face |
| :--------: | :-------------: |
| <img src='https://user-images.githubusercontent.com/64144419/134189344-674d55bb-dc8d-4ceb-91ec-aeeb0c69063b.png' width=100> | <img src='https://user-images.githubusercontent.com/64144419/134189573-93f501af-af4e-48b8-9b12-32cca9a62e8d.png' width=105> |


> Use `python3 pca.py` to run the program

### Database Description
* Resolution: 180x200 (downscaled to 90x100)
* Backgrounds: the background is plain green
* Head Scale:  none
* Head turn,tilt and slant:  very minor variation in these attributes
* Position of face in image:  minor changes
* Image lighting variation:  none
* Expression variation:  considerable expression changes
* Additional comment:  there is no individual hairstlyle variation as the images were taken in a single session.



## Potential Issues of Basic PCA
* Not robust to Shading, Illumination, Expressions, Hairstyles, Spectacles, Non Frontal Images
* PCA works only for Linear Basis and Large Variance having importance

## Potential Explorations
* Try out different distance measures.
* Try selecting only those eigenfaces which encode useful features.
* Try ordering eigenfaces based on Like-Face characteristics.
* Try preprocessing images to align faces, histogram equilize lighting/shading, etc.
* Try tracking faces using DELAUNAY TRIANGULATION.
* Try Kernel PCA
