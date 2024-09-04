import cv2
import pytesseract

# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological transformations to clean the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    
    return image, gray, cleaned

# Function to detect and extract number plates
def detect_number_plate(image, gray, cleaned):
    # Find contours in the cleaned image
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_plates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        # Filter contours based on aspect ratio and size to find number plates
        if 3 < aspect_ratio < 6 and h > 30:  # Adjusted for Kenyan plates
            plate = gray[y:y+h, x:x+w]
            
            # Use Tesseract to extract text from the detected plate
            config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            plate_text = pytesseract.image_to_string(plate, config=config)
            plate_text = plate_text.strip().replace('\n', '')
            
            if plate_text and len(plate_text) > 5:  # Check for typical Kenyan plate length
                detected_plates.append(plate_text)
                print("Detected Number Plate:", plate_text)
                
                # Draw a rectangle around the detected plate
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return image, detected_plates

# Main function
def main(image_path):
    try:
        image, gray, cleaned = preprocess_image(image_path)
        result_image, detected_plates = detect_number_plate(image, gray, cleaned)
        
        if not detected_plates:
            print("No number plates detected.")
        
        # Display the result
        cv2.imshow('Number Plate Recognition', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Error: {e}")

# Path to the image file
image_path = 'image.jpeg'
main(image_path)
