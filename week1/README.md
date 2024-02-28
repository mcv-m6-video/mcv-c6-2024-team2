## C6 Video Analysis Week 1 

### Task 1
### Task 2

### Task 3

In task 3 we had to compare different SOTA methods and compute AP50. Besides the one which were in the instruction, we have implemented GMC, CNT, KNN, GSOC. Some of them, such as MOG, LSBP, GMG, GSOC and CNT did not have a direct implementation in OpenCV so we had to create manually a class for each method. In terms of the evaluation, we have tested 2 methods, in the first one we consider all annotations and skip the frames that don't have an annotation. The second one only consideres the annotations that are not bikes, so they are generally bigger. For both methods, we have obtained the same results, because some methods have by default some parameteres set or logic. 

| Method | AP                   |
|--------|----------------------|
| MOG    | 0.4202364799379725  |
| MOG2   | 0.2180253164556962  |
| LSBP   | 0.822399420079739   |
| GMG    | 0.30575692963752665 |
| CNT    | 0.822399420079739   |
| KNN    | 0.18617961451056309 |
| GSOC   | 0.26045221843003413 |



### Task 4
In Task 4, we optimized foreground extraction by:

Color Space Analysis: Evaluating various color spaces (e.g., RGB, HSV, Lab) for foreground/background discrimination.

Alpha-Rho Combination: Experimenting with different Alpha (α) and Rho (ρ) values to refine background subtraction.

Enhancement Steps:

Fixing Frame Brightness: Set illumination to 150 for consistent results.
Noise Removal: Implemented noise reduction for clearer foreground masks.
Bounding Box Refinement: Ignored bounding boxes with an area < 150 for better accuracy.
Techniques Used:

Non-Adaptive Method: Static approach to foreground extraction.
Adaptive Method: Dynamic adjustment of parameters based on input data, yielding superior results.
These enhancements significantly improved the stability and accuracy of foreground extraction, with the adaptive method proving more effective.

