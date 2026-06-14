
FROM python:3.12-slim

#set work directory inside contianer
WORKDIR /app

#copy the requiremtnt
COPY requirements.txt .


# install dependences 
RUN pip install --no-cache-dir -r requirements.txt


# Copy the whole project
COPY . .

# Expose the port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]