S
# Shape Detection Script

This code processes an image to detect three types of shapes: dots, circles, and diamond shapes. It uses OpenCV for image processing.
## Prerequisites

- Python 3.x
- OpenCV
- NumPy
- scikit-learn

Install the required libraries using pip:

```sh
pip install opencv-python numpy scikit-learn
```

## How to Use

1. **Load Image**: Place the image you want to process in the same directory as the script and rename it to `demo_1.png`.
2. **Run the Script**: Execute the script in your Python environment. The script will process the image, detect the shapes, and display the highlighted shapes.

## Script Breakdown

1. **Load and Preprocess the Image**:
   - The image is loaded and converted to grayscale.
   - Gaussian blur is applied to reduce noise.

2. **Thresholding**:
   - Adaptive thresholding is applied to create a binary image.

3. **Contour Detection**:
   - Contours of the shapes are detected.

4. **Filter Circular Contours**:
   - Circular shapes are filtered based on their circularity and area.
   - K-means clustering is used to classify the circular shapes into two clusters.

5. **Shape Classification and Highlighting**:
   - Shapes are classified as red circles, yellow circles, or white diamond shapes based on their area and circularity.
   - Additional shape features like aspect ratio are used for better detection of diamond shapes.

6. **Display Results**:
   - The counts of red circles, yellow circles, and white diamond shapes are printed.
   - The processed image with highlighted shapes is displayed.

## Example Output

- Count of red circles: `<number>`
- Count of yellow circles: `<number>`
- Count of white diamonds: `<number>`
