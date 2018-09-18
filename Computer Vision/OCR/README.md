# OCR

## [Detecting Text in Natural Scenes with Stroke Width Transform](https://www.researchgate.net/publication/224164328_Detecting_Text_in_Natural_Scenes_with_Stroke_Width_Transform)

**Source Code:** [java](https://github.com/openimaj/openimaj/blob/master/image/image-feature-extraction/src/main/java/org/openimaj/image/text/extraction/swt/SWTTextDetector.java) [c++](https://github.com/aperrau/DetectText)

**Datasets:** Cant find it

> The database, which will be made freely downloadable from our website, consists of 307 color images of sizes ranging from 1024x1360 to 1024x768.

**Author:** (Microsoft) Boris Epshtein,  Eyal Ofek, Yonatan Wexler

**Year of Submission:** 2010

**Time to read (mins):** 280

### What problem does it solve?

Detect text in the wild, without training.

### How does it solve it?

1. Run canny detector to find gradients and gradient directions (d).
2. If a pixel p is an edge boundary then the gradient direction at p should be roughly perpendicular to the stroke direction.
3. Ray trace `r = p + n * d, n = [0,1,2,...]` until the opposite edge is found. Opposite edge is determined if `dq = -dp +- pi/6`
4. Each element on the line `pq` is assigned the value `||p - q||` unless it already has a smaller value. The output matrix was initialized with `+INFINITY`.
5. In the second pass, for each ray we compute the median value for all pixels in the ray and set that value for all pixels which have a value greater than the median.
6. We run a connected component algorithm. We group two components if their ratio does not exceed `3.0`. 
7. In order to accomodate for both bright text on dark background and vice versa. The algorithm is run twice. Once for `dp` and again for `-dp`.
8. Parameters were learnt from a training set.
9. Compute variance of stroke width within each connected component. Reject if variance is too big.
10. Aspect ratio should be between `0.1` and `10`
11. Ratio between diameter and median stroke should be a value less than `10`.
12. Sometimes the text frames are selected. These are filtered by making sure than it should not include more than two other components.
13. Limit font height between `10` and `300`. Only height is limited to enable detection of connected languages like arabic.
14. Since text appears in a line, a line of text is expected to have similar characteristics. e.g. stroke width, letter width, height, spaces, color.
15. Letters are separated into words and lines using a histogram of horizontal distances. (Optional)

### Key takeaways

1. All human text have fonts of similar stroke widths.
2. Road detection and artery detection use similar algorithms.

### What I still do not understand?

### Ideas to pursue

1. Curved text detection by considering the stroke directions.
