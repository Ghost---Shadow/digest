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