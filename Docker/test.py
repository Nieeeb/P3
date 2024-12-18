import io
import requests
from PIL import Image

def main():    
    url = 'https://clarity.schniebster.dk/enhance' # Url for service
    file = {'file': open('test_00011.png', 'rb')} # Reading testing image
    resp = requests.post(url=url, files=file) # Send request to service
    print(f"Response time: {resp.elapsed.total_seconds()}s") # Extract response time
    img = Image.open(io.BytesIO(resp.content)) # Decode bytearray
    img.show() # Show output image

if __name__ == '__main__':
    main()