import pickle
import uvicorn
import io
import uuid
import os
import zipfile
import re
import nltk
import time
from docx import Document
from shutil import make_archive, rmtree
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models_folder_name = 'models'
results_folder_name = 'results'

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


@app.post("/train_documents")
async def train_documents(file: UploadFile = File(...)):
    start_time = time.time()
    contents = await file.read()

    text_list = []
    category_list = []
    zf = zipfile.ZipFile(io.BytesIO(contents), "r")

    for file_name in zf.namelist():
        if not file_name.endswith("/"):
            doc = Document(io.BytesIO(zf.read(file_name)))
            text = ''.join([paragraph.text for paragraph in doc.paragraphs])
            text = preprocess_doc(text)
            text_list.append(text)

            category = file_name.split('/')[0]
            category_list.append(category)

    x_train, x_test, y_train, y_test = train_test_split(text_list, category_list, test_size=0.2)
    svc = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', LinearSVC())])
    svc.fit(x_train, y_train)
    y_prediction = svc.predict(x_test)
    prediction_report = classification_report(y_test, y_prediction)
    trained_model_name = str(uuid.uuid4())

    if not os.path.exists(models_folder_name):
        os.makedirs(models_folder_name)
    pickle.dump(svc, open(f"{models_folder_name}/{trained_model_name}.pkl", 'wb'))

    return {
        "prediction_report": prediction_report,
        "elapsed_time": time.time() - start_time,
        "model_name": trained_model_name
    }


@app.post("/organize_folders/{model_name}")
async def organize_folders(model_name: str, file: UploadFile = File(...)):
    model_file = open(f"{models_folder_name}/{model_name}.pkl", 'rb')
    model = pickle.load(model_file)

    contents = await file.read()
    results_file_name = str(uuid.uuid4())

    if not os.path.exists(results_folder_name):
        os.makedirs(results_folder_name)

    os.mkdir(f'{results_folder_name}/{results_file_name}')

    zf = zipfile.ZipFile(io.BytesIO(contents), "r")
    for file_name in zf.namelist():
        doc = Document(io.BytesIO(zf.read(file_name)))
        text = ''.join([paragraph.text for paragraph in doc.paragraphs])
        text = preprocess_doc(text)
        prediction = model.predict([text])[0].strip()

        if not os.path.exists(f'{results_folder_name}/{results_file_name}/{prediction}'):
            os.makedirs(f'{results_folder_name}/{results_file_name}/{prediction}')

        doc.save(f'{results_folder_name}/{results_file_name}/{prediction}/{file_name}')

    make_archive(f'{results_folder_name}/{results_file_name}', "zip", f'{results_folder_name}/{results_file_name}')
    rmtree(f'{results_folder_name}/{results_file_name}', ignore_errors=True)
    return FileResponse(path=f'{results_folder_name}/{results_file_name}.zip', filename=f'{results_file_name}.zip')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
