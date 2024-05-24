# Use the official Python image from the Docker Hub
FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user

# Set the working directory
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

RUN chown -R user:user /code


# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY --chown=user:user . /code





# Copy the rest of the application code
COPY . .


# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]