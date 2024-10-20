import cv2
import google.generativeai as genai
import numpy as np
from PIL import Image
import time
from datetime import datetime
import gc
from img_to_text import text_extractor
from flask import Flask, render_template, Response, jsonify
import threading

app = Flask(__name__)

# Configure Gemini API - Replace with your actual API key
GOOGLE_API_KEY = 'AIzaSyDu-u4TKO92aM8yUSjCoiXM-WJV6v0ODYY'
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the model
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 1024,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Global variables
camera = None
analysis_result = "Press 'Analyze' to detect items"
frame = None

def analyze_damage(frame):
    """
    Analyze the image using Gemini API to identify texts in the image.
    """
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)

        # Create prompt for damage analysis
        prompt = """
        In this image, output the number of items available and display the brand name with product name and count of the quantity in a tabular form and do not throw an error line just detect whatever it is
        """

        # Generate response from Gemini
        response = model.generate_content([prompt, pil_image])

        # Check if response was blocked
        if response.parts:
            return response.text
        else:
            return "Analysis was blocked by safety settings. Please try again."

    except Exception as e:
        print(f"Detailed error: {str(e)}")
        return f"Error in analysis. Please check your API key and internet connection."

def release_memory():
    """
    Function to release unused memory to prevent overload.
    """
    gc.collect()

def gen_frames():
    global frame
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze')
def analyze():
    global frame, analysis_result
    if frame is not None:
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(frame, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None or img.size == 0:
                print("Error: Decoded image is empty")
                return jsonify({"result": "Error: Unable to process the image"})

            # Save the frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analyzed_frame_{timestamp}.jpg"
            success = cv2.imwrite(filename, img)

            if not success:
                print(f"Error: Failed to save image {filename}")
                return jsonify({"result": "Error: Failed to save the image"})

            print(f"Frame saved as {filename}")

            # Analyze the frame
            analysis_result = analyze_damage(img)
            print(f"Analysis result: {analysis_result}")

            # Log results safely
            try:
                with open('log.txt', 'a') as file:
                    file.write(f"Timestamp: {timestamp}\n")
                    file.write(analysis_result + '\n')
            except Exception as log_error:
                print(f"Logging error: {str(log_error)}")

            release_memory()
            return jsonify({"result": analysis_result})
        except Exception as e:
            print(f"Error in analyze function: {str(e)}")
            return jsonify({"result": f"Error: {str(e)}"})
    else:
        return jsonify({"result": "No frame available for analysis"})

@app.route('/extract_text')
def extract_text():
    global frame
    if frame is not None:
        # Convert bytes to numpy array
        nparr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save the frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analyzed_frame_{timestamp}.jpg"
        cv2.imwrite(filename, img)
        print(f"Frame saved as {filename}")

        # Extract text from the frame
        extracted_text = text_extractor(filename)
        print(f"Extracted text: {extracted_text}")

        release_memory()
        return jsonify({"result": extracted_text})
    else:
        return jsonify({"result": "No frame available for text extraction"})

def start_camera():
    global camera
    camera = cv2.VideoCapture(0)

if __name__ == '__main__':
    start_camera()
    app.run(debug=True)