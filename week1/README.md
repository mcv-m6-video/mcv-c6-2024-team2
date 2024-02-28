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




### Task 4: Color Space Analysis and Background Extraction
In Task 4, we employed various color spaces and combinations of Alpha (α) and Rho (ρ) values to improve foreground extraction. This involved two additional steps aimed at enhancing the quality of results:

Fixing Frame Brightness/Illumination: Ensuring consistent brightness across frames by setting it to a standard value, such as 150, which aids in achieving more stable results.

Noise Removal from Foreground Mask: Employing techniques to eliminate noise from the extracted foreground mask, enhancing the clarity and accuracy of the final result.

Bounding Box Refinement: Implementing a refinement step to avoid detecting bounding boxes for small objects or noise. Bounding boxes with an area less than 150 were disregarded, leading to more reliable outcomes.

Methodology:
Color Space Analysis:
Evaluation of different color spaces such as RGB, HSV, Lab, etc., to determine their effectiveness in foreground/background discrimination.
Alpha-Rho Combination:
Experimentation with various combinations of Alpha (α) and Rho (ρ) values in the background subtraction algorithm to optimize foreground extraction.
Techniques Employed:
Non-Adaptive Method:
Implementing a static approach to foreground extraction, where parameters remain constant throughout the process.
Adaptive Method:
Utilizing an adaptive approach for foreground extraction, where parameters dynamically adjust based on the input data. This method demonstrated superior performance compared to the non-adaptive approach.
Additional Steps for Enhancement:
1. Fixing Frame Brightness/Illumination:
Setting frame brightness/illumination to a predefined value (e.g., 150) to ensure uniformity and stability across frames.
2. Noise Removal:
Employing noise reduction techniques to enhance the clarity and accuracy of the extracted foreground mask.
3. Bounding Box Refinement:
Implementing a refinement step to filter out bounding boxes with an area less than 150, thereby eliminating small objects or noise from the final result.
Results:
The combination of these techniques led to significant improvements in foreground extraction, with the adaptive method outperforming the non-adaptive approach. The stability and accuracy of the results were notably enhanced by the additional steps implemented in the process.
