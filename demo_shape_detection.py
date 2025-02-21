import cv2
import numpy as np

def preprocess_image(image, blur_kernel_size=13, adaptive_block_size=11, adaptive_c=2, morph_kernel_size=5, morph_iterations=0):
    """Preprocesses the image for shape detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Reduce noise and smooth out the image.
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    #Adaptive thresholding (cv2.adaptiveThreshold). This helps separate the shapes (dots, circles, etc.) from the background.
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_c)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    return closed

def detect_shapes(image, contours, circularity_threshold=0.7, aspect_ratio_lower=0.68, aspect_ratio_upper=1.55, dot_area_threshold=35, unknown_area_multiplier=2.0):
    """Detects circles, dots, and Diamond shapes in the contours.

    Calculate Area and Perimeter:
              The area (cv2.contourArea) and perimeter (cv2.arcLength) of the contour are calculated.
    
    Check for Minimum Size:  
             If the area is too small, the contour is ignored.
    
    Calculate Circularity:
             Circularity is a measure of how close the shape is to a perfect circle. It's calculated using the area and perimeter.
    
    Calculate Aspect Ratio:
             Aspect ratio is the ratio of the contour's width to its height.
    
    Classify Shape:
       
             Shapes are classified as circles or dots based on circularity and aspect ratio thresholds. Dots are smaller circles.
    
    Circle areas are stored to calculate the average circle area later to calculate diamonds 

    """
    circle_count = 0
    dot_count = 0
    unknown_count = 0
    circle_areas = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2:  # Minimum size for dots
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = (6 * np.pi * area) / (perimeter ** 2.1)
        (x, y, w, h) = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        is_circle = circularity > circularity_threshold and aspect_ratio_lower <= aspect_ratio <= aspect_ratio_upper
        is_dot = area < dot_area_threshold and is_circle

        if is_circle and not is_dot:  # Only add real circles
            circle_areas.append(area)

    avg_circle_area = np.mean(circle_areas) if circle_areas else 0
    avg_circle_area = avg_circle_area * unknown_area_multiplier

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2:  # Minimum size for dots
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # --- Shape Detection Logic ---
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)  # Adjust epsilon as needed
        num_vertices = len(approx)

        # Circle/Dot Detection
        circularity = (6 * np.pi * area) / (perimeter ** 2.1)
        (x, y, w, h) = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        is_circle = circularity > circularity_threshold and aspect_ratio_lower <= aspect_ratio <= aspect_ratio_upper
        is_dot = area < dot_area_threshold and is_circle

        if is_circle:
            if is_dot:
                dot_count += 1
                color = (231, 201, 239)  # Green for dots
            else:
                circle_count += 1
                color = (0, 255, 254)  # Red for circles

            cv2.drawContours(image, [cnt], -1, color, 1)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(image, (cX, cY), 3, color, 1)

        # Unknown Shape Detection
        else:
            if avg_circle_area > 0 and area > avg_circle_area and len(approx) > 2:  # Check for zero and > 2 vertices
                unknown_count += 1
                color = (255, 245, 255)  # Gray for unknown
                cv2.drawContours(image, [cnt], -1, color, 1)
                cv2.putText(image, "D", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)  # "U" label

    return image, dot_count, circle_count, unknown_count

def display_results(image, original, dot_count, circle_count, unknown_count):
    """Displays the results on the image."""

    cv2.putText(image, f"Dots: {dot_count}, Circles: {circle_count}, Diamond: {unknown_count}", (5, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    combined = np.hstack([original, image])
    # Save the processed image
    output_path = "detected_shapes.png"
    cv2.imwrite(output_path, combined)

    cv2.imshow("Detection Results", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# --- Main Program ---
image = cv2.imread("demo_1.png")  # Replace with your image path
original = image.copy()

# Parameterized values (you can adjust these)
blur_kernel_size = 13
adaptive_block_size = 11
adaptive_c = 2
morph_kernel_size = 5
morph_iterations = 0
circularity_threshold = 0.7
aspect_ratio_lower = 0.68
aspect_ratio_upper = 1.55
dot_area_threshold = 35
unknown_area_multiplier = 2.0  # Adjust this to change how "larger" is defined

closed = preprocess_image(image, blur_kernel_size, adaptive_block_size, adaptive_c, morph_kernel_size, morph_iterations)
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image, dot_count, circle_count, unknown_count = detect_shapes(image, contours, circularity_threshold, aspect_ratio_lower, aspect_ratio_upper, dot_area_threshold, unknown_area_multiplier)
display_results(image, original, dot_count, circle_count, unknown_count)
