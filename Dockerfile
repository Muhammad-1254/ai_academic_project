# Use the official Python image from the Docker Hub
FROM python:3.10-slim

RUN useradd -m -u 1000 app


# Set the working directory
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt



# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


# Create the attendance.csv with ownership by app user
RUN touch /code/attendance.csv && chown app:app /code/attendance.csv


USER app


# Copy the rest of the application code
COPY . .


# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]