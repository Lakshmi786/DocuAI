FROM python:3.10

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

# Install required Python packages
RUN pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./* /app

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "80"]


