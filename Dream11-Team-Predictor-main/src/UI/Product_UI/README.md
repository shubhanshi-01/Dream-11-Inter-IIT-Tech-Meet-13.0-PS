# Dream 11 Next Gen Team Builder Product UI

### Installation
NodeJS and Python3 are required to run this project. Install the dependencies using the following commands:
```bash
npm install
npm run build
cd backend
pip install -r requirements.txt
```

### .env
Create a .env file in the root directory (of product_UI) with the following content:
```bash
NEXT_PUBLIC_GOOGLE_TRANSLATE_API_KEY=<Your Google Cloud API KEY supporting Translate>
NEXT_PUBLIC_GOOGLE_TTS_API_KEY=<Your Google Cloud API KEY supporting TTS>
```
### Usage

Run the frontend using
```bash
npm run start
``` 
Then run the backend using
```bash
cd backend
python3 app.py
```
Navigate to http://localhost:3000/ to view the frontend.
