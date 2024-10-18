## Face Recognition Attendance System

## Project Overview
The Face Recognition Attendance System is an innovative project designed to streamline the process of attendance management using advanced AI technologies. The system leverages face recognition to identify and verify individuals, ensuring an efficient and secure attendance tracking mechanism. This project is particularly trained on many popular celebrities, as well as myself.

## Project Structure
Although the project is still under development, the core functionalities have been implemented. The system is capable of being trained on user faces and can be integrated into various applications. The API is ready for recognizing user faces.

### Key Features
- **Face Detection**: Utilizes MTCnn for detecting faces in images.
- **Face Embeddings**: Employs the InceptionResnetV1 model pretrained on VGGFace2 to generate embeddings for detected faces.
- **Face Indexing**: Uses FAISS (Facebook AI Similarity Search) to store and index face embeddings.
- **Face Recognition**: Matches new face embeddings against the stored embeddings to recognize users.

## Libraries and Frameworks
- **MTCnn**: For face detection.
- **opencv-python**: For image processing.
- **facenet_pytorch**: For generating face embeddings.
- **faiss-gpu**: For efficient similarity search.
- **numpy**: For numerical operations.
- **InceptionResnetV1 (pretrained on VGGFace2)**: For obtaining robust face embeddings.

## Dataset
The dataset consists of a collection of user images. Each user's face is detected using MTCnn, and embeddings are generated using the VGGFace model. These embeddings are then stored using FAISS, indexed by username or ID. When a face is to be recognized, the system detects the face, generates an embedding, and checks if the user exists in the FAISS index.



### Dataset CSV File
The dataset is organized in a CSV file containing the following columns:
- `Username`: The username of the individual.
- `Image Path`: The file path to the user's image.
- `Embedding`: The face embedding vector.


## Project Deployment
The project is deployed using Docker and FastAPI on huggingface.co sapces. The API is accessible for recognizing faces. You can access the API documentation [here](https://muhammad-1254-ai-project01.hf.space/docs#/default/recognize_recognize__post).


You can also find the project repository on huggingface.co [here](https://huggingface.co/spaces/Muhammad-1254/ai_project01/tree/main).



## Future Work
The project is ongoing, with several enhancements planned:
- **Improved Accuracy**: Fine-tuning the face recognition model for better accuracy.
- **Scalability**: Enhancing the system to handle larger datasets efficiently.
- **Real-time Processing**: Optimizing the system for real-time face recognition.

## Conclusion
The Face Recognition Attendance System is a promising application of AI in automating attendance management. By leveraging state-of-the-art face recognition technologies, the system aims to provide a seamless and secure solution for user identification.

## Contact
For more information about the project or collaboration opportunities, feel free to contact me.
