# Link AI Chat 🤖

A modern, responsive AI-powered chat application built with Flask and Google's Gemini AI. Features real-time conversations, file analysis, and a beautiful glassmorphism UI design.

![Link AI Chat](https://img.shields.io/badge/AI-Powered-blue) ![Flask](https://img.shields.io/badge/Flask-2.0.1-green) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Status](https://img.shields.io/badge/Status-Active-success)

## ✨ Features

### 🎯 Core Features
- **AI-Powered Chat**: Intelligent conversations using Google's Gemini 1.5 Flash model
- **File Analysis**: Upload and analyze images, text files, code files, and documents
- **Multi-format Support**: Supports images (JPG, PNG, GIF), text files, code files (Python, JavaScript, HTML, CSS, JSON, XML, CSV)
- **Conversation History**: Maintains context across conversations for better responses
- **Real-time Responses**: Fast, streaming-like responses with proper formatting

### 🎨 User Interface
- **Modern Design**: Beautiful glassmorphism UI with smooth animations
- **Dark/Light Mode**: Toggle between dark and light themes
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile devices
- **Particle Effects**: Dynamic background animations for enhanced visual appeal
- **Centered Navigation**: Clean, centered navigation bar design

### 🔐 Authentication
- **User Registration**: Sign up with email and password
- **Secure Login**: Firebase-based authentication system
- **Session Management**: Secure user sessions and data protection

### 🛠️ Technical Features
- **Auto-reload**: Development server automatically restarts on code changes
- **File Upload**: Secure file handling with unique filename generation
- **Error Handling**: Comprehensive error handling and user feedback
- **CORS Support**: Cross-origin resource sharing for API access
- **Rate Limiting**: Built-in request rate limiting (configurable)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/link-ai-chat.git
   cd link-ai-chat
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env and add your API key
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. **Run the application**
   ```bash
   # Using the main script
   python chat.py
   
   # Or using the development runner
   python run_dev.py
   ```

6. **Access the application**
   - Local: http://127.0.0.1:5000
   - Network: http://your-local-ip:5000

## 📖 Usage

### Starting a Chat
1. Navigate to the welcome page
2. Click on "Start Chatting" or go to `/chat`
3. Type your message and press Enter or click Send
4. The AI will respond with helpful, formatted answers

### File Analysis
1. In the chat interface, click the file upload button (📎)
2. Select an image or text file
3. Optionally add a message describing what you want to know
4. The AI will analyze the file and provide detailed insights

### Supported File Types
- **Images**: JPG, JPEG, PNG, GIF, BMP, WEBP
- **Text Files**: TXT, MD, CSV
- **Code Files**: PY, JS, HTML, CSS, JSON, XML
- **Documents**: Various text-based formats

## 🏗️ Project Structure

```
link-ai-chat/
├── chat.py                 # Main Flask application
├── run_dev.py             # Development server runner
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── README.md             # This file
├── templates/            # HTML templates
│   ├── index.html        # Landing page
│   ├── welcome.html      # Welcome/home page
│   ├── chat.html         # Main chat interface
│   ├── signup.html       # User registration
│   └── signin.html       # User login
├── static/               # Static assets
│   ├── styles.css        # Main stylesheet
│   ├── script.js         # JavaScript functionality
│   ├── auth.js           # Authentication logic
│   ├── firebase-config.js # Firebase configuration
│   ├── particles.json    # Particle effects config
│   ├── logos/            # Logo assets
│   └── uploads/          # User uploaded files
└── __pycache__/          # Python cache files
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
FLASK_ENV=development
FLASK_DEBUG=True
```

### Getting a Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

### Rate Limiting Configuration
Modify these variables in `chat.py`:
```python
RATE_LIMIT_WINDOW = 60  # Time window in seconds
MAX_REQUESTS_PER_WINDOW = 30  # Max requests per window
```

## 🎨 Customization

### Changing the AI Model
In `chat.py`, modify the model initialization:
```python
model = genai.GenerativeModel('gemini-1.5-pro')  # Use Pro model
```

### UI Themes
The application supports dark and light modes. You can customize colors in `static/styles.css`:
```css
:root {
    --primary-color: #your-color;
    --background-color: #your-background;
}
```

### System Instructions
Modify the AI's behavior by editing `SYSTEM_INSTRUCTION` in `chat.py`:
```python
SYSTEM_INSTRUCTION = "Your custom instructions here..."
```

## 🚀 Deployment

### Local Network Access
The server runs on `0.0.0.0:5000` by default, making it accessible on your local network.

### Production Deployment
For production deployment, consider:

1. **Using a production WSGI server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 chat:app
   ```

2. **Setting up reverse proxy** (Nginx):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **Environment variables for production**:
   ```env
   FLASK_ENV=production
   FLASK_DEBUG=False
   ```

## 🐛 Troubleshooting

### Common Issues

**Server won't start - Port already in use**
```bash
# Kill processes on port 5000
lsof -ti:5000 | xargs kill -9
```

**API Key not working**
- Ensure your `.env` file is in the root directory
- Check that your API key is valid and has proper permissions
- Verify the key is not wrapped in quotes in the `.env` file

**File upload not working**
- Check that the `static/uploads/` directory exists
- Ensure proper file permissions
- Verify file size limits

**Module not found errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Debug Mode
Enable debug mode for detailed error messages:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comments for complex logic
- Test your changes thoroughly
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini AI** - For providing the powerful AI capabilities
- **Flask** - For the excellent web framework
- **Particles.js** - For beautiful background animations
- **Firebase** - For authentication services

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/link-ai-chat/issues) page
2. Create a new issue with detailed information
3. Include error messages and steps to reproduce

## 🔄 Updates

### Version 1.0.0
- Initial release with core chat functionality
- File upload and analysis features
- Modern glassmorphism UI design
- Dark/light mode support
- Firebase authentication integration

---

**Made with ❤️ and AI** | [Report Bug](https://github.com/yourusername/link-ai-chat/issues) | [Request Feature](https://github.com/yourusername/link-ai-chat/issues) 
