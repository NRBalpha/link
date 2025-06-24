from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import re
import secrets
import google.generativeai as genai
from dotenv import load_dotenv
import json
import base64

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__, 
    static_url_path='',
    static_folder='static',
    template_folder='templates'
)
CORS(app)
app.secret_key = secrets.token_hex(32)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Gemini API config
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise EnvironmentError("❌ GEMINI_API_KEY not found. Please check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)
model = None

def initialize_model():
    """Initialize the Gemini model with optimized settings"""
    global model
    try:
        model = genai.GenerativeModel('gemini-1.5-flash',
            generation_config={
                'temperature': 0.85,
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 2048,
                'candidate_count': 1,
            }
        )
        # Test the model
        response = model.generate_content("Hello")
        return bool(response and response.text)
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return False

def format_response(text):
    """Clean and format AI response for better readability"""
    if not text:
        return "I'm not sure how to answer that."
    
    text = text.strip()
    
    # Remove markdown formatting symbols
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'[_~`]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Improve structure
    text = re.sub(r'\n([A-Z][A-Z\s:]+:)\n', r'\n\n\1\n', text)
    text = re.sub(r'([A-Z][A-Z\s]+:)([^\n])', r'\1\n\2', text)
    text = re.sub(r'\n(\d+\.)', r'\n\n\1', text)
    text = re.sub(r'\n(-\s)', r'\n\n\1', text)
    text = re.sub(r'([.!?])\s*\n([A-Z])', r'\1\n\n\2', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

def build_conversation_context(conversation_history):
    """Build context from conversation history"""
    if not conversation_history:
        return ""
    
    context = "\n\nCONVERSATION CONTEXT:\n"
    for msg in conversation_history[-10:]:  # Last 10 messages
        role = "User" if msg.get('role') == 'user' else "Assistant"
        context += f"{role}: {msg.get('content', '')}\n"
    return context + "\n"

# System instruction for the AI
SYSTEM_INSTRUCTION = (
    "You are a helpful, knowledgeable, and professional AI assistant. "
    "Provide clear, accurate, and well-structured responses. "
    "Use proper formatting with headings, lists, and examples when helpful. "
    "Never use asterisks (*) for formatting. Use dashes (-) for bullet points. "
    "Be comprehensive but concise, and always aim for clarity and helpfulness."
)

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Handle file upload requests
        if 'file' in request.files:
            return handle_file_upload()
        
        # Handle regular text chat
        data = request.get_json()
        if not data or not data.get('message'):
            return jsonify({"error": "No message provided"}), 400
            
        message = data.get('message', '')
        conversation_history = data.get('conversation_history', [])
        
        # Build prompt with context
        context = build_conversation_context(conversation_history)
        prompt_text = SYSTEM_INSTRUCTION + context + message + "\n\nAssistant:"
        
        # Generate response
        if model:
            response = model.generate_content(prompt_text)
            answer = format_response(response.text) if response and response.text else "I'm not sure how to respond to that."
        else:
            answer = "AI model is not available."

        return jsonify({"response": answer})
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

def handle_file_upload():
    """Handle file upload and analysis"""
    file = request.files['file']
    message = request.form.get('message', '')
    
    # Get conversation history
    history_json = request.form.get('conversation_history', '[]')
    try:
        conversation_history = json.loads(history_json)
    except:
        conversation_history = []
    
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Create secure filename and save file
    filename = secrets.token_hex(8) + "_" + file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": "Failed to save file"}), 500

    # Build context
    context = build_conversation_context(conversation_history)
    
    # Handle different file types
    if file.mimetype.startswith('image/'):
        answer = process_image_file(file_path, file.mimetype, message, context)
    elif file.mimetype.startswith('text/') or file.filename.endswith(('.txt', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv')):
        answer = process_text_file(file_path, file.filename, message, context)
    else:
        answer = f"Sorry, I don't support files of type '{file.mimetype}'. I can analyze images and text files."

    return jsonify({
        "response": answer,
        "file_url": f"/static/uploads/{filename}",
        "file_name": file.filename,
        "file_type": file.mimetype
    })

def process_image_file(file_path, mime_type, message, context):
    """Process and analyze image files"""
    try:
        with open(file_path, 'rb') as img_file:
            image_data = img_file.read()
        
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        prompt_text = SYSTEM_INSTRUCTION + context + "\n\n" + (
            message if message else 
            "Please analyze this image and provide:\n\n"
            "1. A brief overview of what you see\n"
            "2. Key details about the main subjects or objects\n"
            "3. Information about colors, composition, and setting\n"
            "4. Any text or important details you notice\n"
            "5. Notable features, patterns, or technical aspects\n\n"
            "Please organize your response clearly."
        )
        
        content = [prompt_text, {"mime_type": mime_type, "data": image_b64}]
        
        if model:
            response = model.generate_content(content)
            return format_response(response.text) if response and response.text else "I couldn't analyze this image."
        else:
            return "AI model is not available."
            
    except Exception as e:
        return f"Sorry, I couldn't process this image. Error: {str(e)}"

def process_text_file(file_path, filename, message, context):
    """Process and analyze text files"""
    try:
        with open(file_path, 'r', encoding='utf-8') as text_file:
            file_content = text_file.read()
        
        prompt_text = SYSTEM_INSTRUCTION + context + "\n\n" + (
            f"I've uploaded a file called '{filename}'. Could you analyze it?\n\n"
            f"Here's the content:\n\n```\n{file_content}\n```\n\n" +
            (message if message else 
             "Please help me understand this file by providing:\n\n"
             "1. What type of file this is and its main purpose\n"
             "2. Key components, functions, or sections\n"
             "3. How it's structured and organized\n"
             "4. Any notable features or important aspects\n"
             "5. If it's code, any observations about quality or improvements\n\n"
             "Please explain everything clearly.")
        )
        
        if model:
            response = model.generate_content(prompt_text)
            return format_response(response.text) if response and response.text else "I couldn't analyze this file."
        else:
            return "AI model is not available."
            
    except UnicodeDecodeError:
        return "Sorry, I couldn't read this file. It might be in a format I don't support."
    except Exception as e:
        return f"Sorry, I couldn't process this file. Error: {str(e)}"

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Entry point
if __name__ == '__main__':
    # Initialize model
    if initialize_model():
        print("✅ AI model initialized successfully.")
    else:
        print("❌ Failed to initialize AI model. Chat functionality will be limited.")
    
    # Get local IP for network access
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "127.0.0.1"
    
    # Show startup info only once
    if not os.environ.get('WERKZEUG_RUN_MAIN'):
        print(f"🌐 Server starting...")
        print(f"📱 Local: http://127.0.0.1:5000")
        print(f"📱 Network: http://{local_ip}:5000")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")
        # Fallback without debug mode
        app.run(debug=False, host='0.0.0.0', port=5000)

