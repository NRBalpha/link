import os
import requests

# Create directories if they don't exist
os.makedirs('static/logos', exist_ok=True)

# Logo URLs and their corresponding filenames
logos = {
    'html5.svg': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/html5/html5-original.svg',
    'css3.svg': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/css3/css3-original.svg',
    'javascript.svg': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/javascript/javascript-original.svg',
    'firebase.svg': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/firebase/firebase-plain.svg',
    'python.svg': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg',
    'flask.svg': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/flask/flask-original.svg',
    'ollama.svg': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg',
    'mistral.svg': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/ai/ai-original.svg'
}

# Fallback SVG content in case the download fails
fallback_svgs = {
    'ollama.svg': '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <path d="M8 12h8M12 8v8"/>
    </svg>''',
    'mistral.svg': '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
    </svg>'''
}

def download_logo(url, filename):
    try:
        print(f"Attempting to download {filename} from {url}")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(f'static/logos/{filename}', 'wb') as f:
                f.write(response.content)
            print(f'Successfully downloaded {filename}')
        else:
            print(f'Failed to download {filename}: Status code {response.status_code}')
            # Use fallback SVG if available
            if filename in fallback_svgs:
                with open(f'static/logos/{filename}', 'w') as f:
                    f.write(fallback_svgs[filename])
                print(f'Created fallback SVG for {filename}')
    except Exception as e:
        print(f'Error downloading {filename}: {str(e)}')
        # Use fallback SVG if available
        if filename in fallback_svgs:
            with open(f'static/logos/{filename}', 'w') as f:
                f.write(fallback_svgs[filename])
            print(f'Created fallback SVG for {filename}')

# Download each logo
for filename, url in logos.items():
    download_logo(url, filename)

print("\nLogo download complete. Please check the static/logos directory.") 