import pickle
import uvicorn
import io
import uuid
import os
import zipfile
import re
import nltk
from docx import Document
from shutil import make_archive, rmtree
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
model_file = open("pipeline2.pkl", 'rb')
model = pickle.load(model_file)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


WPT = nltk.WordPunctTokenizer()
stop_word_list = nltk.corpus.stopwords.words('turkish')


def preprocess_doc(single_doc):
    # remove numbers
    number_pattern = re.compile('\d+')
    single_doc = number_pattern.sub('', single_doc)

    # remove punctuation and underscore
    punc_pattern = re.compile('[\W_]+')
    single_doc = punc_pattern.sub(' ', single_doc)

    # make lowercase
    single_doc = single_doc.lower()

    # remove trailing spaces
    single_doc = single_doc.strip()

    # remove multiple spaces
    single_doc = single_doc.replace('\s+', ' ')

    # remove stop words
    tokens = WPT.tokenize(single_doc)
    filtered_tokens = [token for token in tokens if token not in stop_word_list]
    single_doc = ' '.join(filtered_tokens)
    return single_doc


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    results_file_name = str(uuid.uuid4())

    os.mkdir(results_file_name)

    zf = zipfile.ZipFile(io.BytesIO(contents), "r")
    for file_name in zf.namelist():
        doc = Document(io.BytesIO(zf.read(file_name)))
        text = ''.join([paragraph.text for paragraph in doc.paragraphs])
        text = preprocess_doc(text)
        prediction = model.predict([text])[0].strip()

        if not os.path.exists(f'{results_file_name}/{prediction}'):
            os.makedirs(f'{results_file_name}/{prediction}')

        doc.save(f'{results_file_name}/{prediction}/{file_name}')

    make_archive(results_file_name, "zip", results_file_name)
    rmtree(results_file_name, ignore_errors=True)
    return FileResponse(path=f'{results_file_name}.zip', filename=f'{results_file_name}.zip')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
