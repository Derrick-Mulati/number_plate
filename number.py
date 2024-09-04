import cv2
import pytesseract

# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Apply morphological transformations to clean the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return image, gray, cleaned

# Function to detect and extract number plates
def detect_number_plate(image, gray, cleaned):
    # Find contours in the cleaned image
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        # Filter contours based on aspect ratio to find number plates
        if 2 < aspect_ratio < 5:
            plate = gray[y:y+h, x:x+w]
            
            # Use Tesseract to extract text from the detected plate
            plate_text = pytesseract.image_to_string(plate, config='--psm 8')
            plate_text = plate_text.strip().replace('\n', '')
            
            print("Detected Number Plate:", plate_text)
            
            # Draw a rectangle around the detected plate
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return image

# Main function
def main(image_path):
    image, gray, cleaned = preprocess_image(image_path)
    result_image = detect_number_plate(image, gray, cleaned)
    
    # Display the result
    cv2.imshow('Number Plate Recognition', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the image file
image_path = 'path_to_image.jpg'
main(image_path)
