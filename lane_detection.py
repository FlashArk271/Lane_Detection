
import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur

def detect_edges(image):
    return cv2.Canny(image, 100, 120)

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(image, mask)
    return masked

def hough_lines(image):
    return cv2.HoughLinesP(image, 2, np.pi/60, 50, np.array([]), minLineLength=40, maxLineGap=80)

def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
    return cv2.addWeighted(image, 0.8, line_image, 1, 0)

def process_image(image):
    preprocessed = preprocess_image(image)
    edges = detect_edges(preprocessed)
    height, width = edges.shape
    vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)
    lines = hough_lines(roi)
    result = draw_lines(image, lines)
    return result

def resize_frame(frame, width=640):
    height = int(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to standard size
        resized_frame = resize_frame(frame)
        
        processed_frame = process_image(resized_frame)
        
        # Resize processed frame back to original size
        output_frame = cv2.resize(processed_frame, (frame_width, frame_height))
        
        out.write(output_frame)
        cv2.imshow('Lane Detection', output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video('test_video.mp4')