# Photogrammetry Digest

## [Large Pose 3D Face Reconstruction from a Single Image via Direct Volumetric CNN Regression](https://arxiv.org/abs/1703.07834)

**Source Code:** [Github](https://github.com/AaronJackson/vrn)

**Datasets:** 
1. [Face Alignment Across Large Poses: A 3D Solution](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)

**Time to read:** 150 mins

**Easy to read?:** Yes

**Author:** Aaron S. Jackson et al

**Year of Submission:** 2017

### What problem does it solve?

Face reconstruction from a single image.

1. Works with a single 2D facial image.
2. Does not require accurate alignment.
3. Does not require dense correspondence between images.
4. Works for arbitrary poses and expressions.

Simplifying the technical challenges involved in the related works is the main motivation of this paper.

### How does it solve it?

1. 2D image to facial feature heatmaps.
2. Original image + facial feature heatmaps to 3D voxels.

Model: Volumetric Regression Network [Hourglass network](https://arxiv.org/abs/1603.06937)

Loss function: Cross entrophy loss between ground truth voxels and reconstructed voxels.

Optimizer: RMSProp

Error metric: Per vertex Euclidean distance

### How is this paper novel?

Directly maps pixels to 3D coordinates using a CNN instead of 2D height maps or 3D Morphable Models.

Regresses to a much larger volumetric resolution.

### Key takeaways

1. 3D Morphable Models uses an iterative flow procedure for dense image correspondence which is prone to failure. Slow, highly non convex optimization.
2. Direct regression of 3D points using L2 loss is difficult.
3. Volumetric Regression Networks largely outperform 3DDFA and EOS on all datasets, verifying that directly regressing the 3D facial structure is a much easier problem for CNN learning.
4. The best performing VRN is the one guided by detected landmarks.
5. The shape for all faces appears to remain almost identical.
6. Performance of this model decreases with increase in pose. (?)

### What I still do not understand?

1. What are 3D Morphable Models?
2. Why is direct regression of 3D points using L2 loss is difficult?
3. What is Mahalanobis distance?
4. What is Iterative Closest Point?

### Ideas to pursue

1. Make a CNN predict the height of each vertex of a large number of randomly generated, shaded, height map images. Then use transfer learning for face topology.
2. Predict the height deltas betwen the generated and target images instead of the height in one go.
3. Semantically label parts of the face and then use texture super resolution. The super resolution generating model would get both the semantic label and the low resoluion texture. This enables the usage of a much smaller model and kernel apertures. 
4. Make the edges in the generated topology align with the global coordinates.

## [Hierarchical Surface Prediction for 3D Object Reconstruction](https://arxiv.org/abs/1704.00710)

**Source Code:** [???]()

**Datasets:** [Shapenet](https://shapenet.org/) (Aeroplanes, Chairs and Cars)

**Time to read:** 200 mins

**Easy to read?:** Easy (Moderate notation usage)

**Author:** Christian Hane et al

**Year of Submission:** 2017

### What problem does it solve?

It generates voxels from a single 2D image. Surface area increases quadratically while volume increases cubically so a very large number of voxels are "uninteresting".

### How does it solve it?

#### Overview 

1. Label a voxel grid into three (free space, boundary and occupied)

2. Divide the boundary voxels using octrees and repeat

3. The division is also done by a factor of 2 to increase resolution

Block size (Conv apperture) = 16

Tree depth = 5

#### Model

1. Conv encoder encodes the color OR depth image OR low res 3D volume to 128 dimensions
2. Conv decoder decodes it to a voxel grid with labels, outside, inside and boundary
3. A threshold parameter is used to differentiate between free and filled voxels
4. The generated low res voxel map is sent through another upsampler this upsampler only upsamples voxels on boundary regions. It is also supplied with padding around the boundary regions such that the voxel regions overlap. This allows smoother results.
5. A mesh is generated using marching cubes

There is supervision for each level of the tree (upsampling)

#### Training

In the beginning of training the number of false positives are very high. So, the deep layers are not evaluated. As the validation loss decreases, the next layer is allowed to be trained. This ensures that all levels of the tree eventually get trained.

#### Voxelization (Dataset generation)

1. A low res voxel map is generated.
2. Outermost layer is eroded.
3. A next higher layer voxel map is generated

### How is this paper novel?

Other approaches predict only a coarse resolution voxel grid. They directly minimize reprojection errors.

### Key takeaways

### What I still do not understand?

1. What is implicit volumetric reconstruction?

2. Delaunay tedrahedrization.

### Ideas to pursue

1. Use three.js to feed live data to tensorflow.js

2. Allow the model to move the camera around

3. Orthogonal autoencoders for facial morphs

## [Deep Learning of Convolutional Auto-encoder for Image Matching and 3D Object Reconstruction in the Infrared Range](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w31/Knyaz_Deep_Learning_of_ICCV_2017_paper.pdf)

**Source Code:** [???]()

**Datasets:** [multi-view stereo infrared imagery dataset](http://www.zefirus.org/mvsir17)

**Time to read:** 106 mins

**Easy to read?:** Meh

**Author:** Vladimir A. Knyaz et al

**Year of Submission:** 2017

### What problem does it solve?

In dust, fog and low light conditions infrared images can be used to generate 3D geometry.

### How does it solve it?

Takes a stereo infrared scan, encodes parts of image using CAE, matches corresponding parts of both images and uses that information to generate a point cloud.

1. **Local patch extraction:** Unstable on infrared imagery. Select small path 28x28 which is invariant to perspective transformations

2. **CAE:** The patches are autoecoded

#### Model

| Layer       | Size out    | Kernel | Stride |
| ----------- | ----------- | ------ | ------ |
|Input        | 1 × 28 × 28 |        |        |
|Convolution  | 8 × 13 × 13 | 4 × 4  | 2      |
|Convolution  | 16 × 5 × 5  | 4 × 4  | 2      |
|Inner Product| 16 x 1 x 1  |        |        |
|Deconvolution| 8 × 5 × 5   | 5 × 5  | 2      |
|Deconvolution| 1 × 28 × 28 | 8 × 8  | 5      |

3. **Patch matching:** Prepare a codebook that establishes correspondences between the patch code and patch ID of a unique 3D point on a test object. 

All patches are processed using CAE to get a code.

Query the code book to find n nearest neighbours of the code.

### How is this paper novel?

1. DL based image patch matching method

2. Usage of images with only untextured diffuse shading

### Key takeaways

1. Standard approaches rely on texture for matching keypoints. So if the object does not provide sufficient texture then the quality drops significantly.

2. Image matching methods that use finite object planes such as plane sweep matching or PatchMatch seem to be robust on low textured areas.

### What I still do not understand?

1. Semi global block matching

2. Patch matching, finite object planes.

3. SFM

4. Semi Global Matching

5. How are patches converted to point cloud?

### Ideas to pursue

EMPTY
