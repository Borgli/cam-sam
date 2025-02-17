## Code repository for:
# Automatic Prompt Generation Using Class Activation Maps for Foundational Models: A Polyp Segmentation Case Study


### Abstract:
We introduce a weakly supervised segmentation approach that leverages class activation maps and the Segment Anything Model to generate high-quality masks using only classification data. A pre-trained classifier produces class activation maps that, once thresholded, yield bounding boxes encapsulating the regions of interest. These boxes prompt SAM to generate detailed segmentation masks, which are then refined by selecting the best overlap with automatically generated masks from the foundational model using the intersection-over-union metric. In a polyp segmentation case study, our approach outperforms existing zero-shot and weakly supervised methods, achieving a mean intersection over union of 0.63. This method offers an efficient and general solution for image segmentation tasks where segmentation data are scarce.

---

##  Code Documentation

The code is split into each experiment file which can be ran independently.


Please cite the paper if using any of this work.

Bib citation:
```

```
