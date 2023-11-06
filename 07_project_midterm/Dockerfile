FROM python:3.10

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"] 

RUN pipenv install --system --deploy

COPY ["predict.py", "model.pkl", "requirements.txt", "./"]

RUN pip install -r requirements.txt

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]