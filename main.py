from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import faiss
import numpy as np
import io
import glob
from contextlib import asynccontextmanager
from tqdm.notebook import tqdm
import os
# from utils import read_csv, add_attendance, clear_csv


# file_path = "attendance.csv"
# os.chmod(file_path, 0o666)

cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
os.makedirs(cache_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device detected: {device}")
mtcnn = MTCNN(image_size=160, margin=10, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()
print("loading resnet model completed...")

embed_file_name = 'completeEmb.npy'
id2name_file_name = 'id2name.npy'

# Placeholder for embeddings and index
embeddings = None
index = None
id2name = dict()




async def on_start():
    global index, embeddings, id2name
    if os.path.exists(embed_file_name):
        embeddings = np.load(embed_file_name)
        print(f"embedding file loaded...")
    else:
        print(f"{embed_file_name} not found")
        # for file in tqdm(files):
        #     img = Image.open(file)
        #     img = mtcnn(img).to(device)
        #     embedding = resnet(img.unsqueeze(0)).cpu().detach().numpy()
        #     if embeddings is None:
        #         embeddings = embedding
        #     else:
        #         embeddings = np.concatenate((embeddings, embedding), axis=0)  
        print(f"new embeddings completed... ")
        # np.save(embed_file_name, embeddings)
        # print("embeding saving completed")
    print(f"embedding shape: {embeddings.shape}")
      # loading the user with id's
    labels = np.load("nameLabels.npy")
        
    if os.path.exists(id2name_file_name):
        id2name = np.load(id2name_file_name, allow_pickle=True).item()
        print(f"id2name file loaded...")
    else:
        for i,e in enumerate(set(labels)):
            id2name[i] = e
    
    name2id = {id2name[i]:i for i in id2name}
    print(f"id2name: {id2name}")
    
    label_with_ids = []
    for i in range(0, len(labels)):
        label_with_ids.append(name2id[labels[i]])
    print(f"label_with_ids len: {len(label_with_ids)}")

    print(f"preparing faiss")
    nlist = 10
    m = 32
    dim = m * 16 # = 512
    nbits = 5
    print(f"dim: {dim}")
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)    
    print(f"embedding shape: {embeddings.shape}")
    index.train(embeddings)
    index.add_with_ids(embeddings, np.array(label_with_ids))
    print(f"loading index completed with id's")
    
async def on_shutdown():
    # np.save("newEmbeddings.npy", embeddings)
    print("saving embeddings completed...")

@asynccontextmanager
async def lifespan(app:FastAPI):
    await on_start()
    
    yield
    await on_shutdown()
    yield
app = FastAPI(lifespan=lifespan)



@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
    
        img_cropped = mtcnn(image).to(device)
    
        print("recog image face detec complted")
        if img_cropped is None:
            return HTTPException(status_code=400, detail="No face detected")
        embedding = resnet(img_cropped.unsqueeze(0)).cpu().detach().numpy()
    except Exception as e:
        raise {"error" : f"error occur {e}"}
    print('embed for recognize completed')
    D, I = index.search(embedding, 15)  # Search for the closest match
    new_D = []
    new_I = []
    
    for i in range(0, len(D[0])):
        if D[0][i] >= 0.6:
            new_D.append(D[0][i])
            new_I.append(I[0][i])
    
    if len(new_D) <1:
        return HTTPException(status_code=400, detail="Person not Found in DB")

    found_name = id2name[new_I[0]]
    print(f"found name: {found_name}")
    print(f"D: {D}")
    print(f"I: {I}")
    print(f"find in id2name file: {id2name[new_I[0]]}")
    if len(list(set(new_I)))>1:
        print(f"found multiple: {set(new_I)}")
        t = {}       
        for i in new_I:
            if i in t:
                t[i] =t[i] +1
            else:
                t[i] = 1
        print(f"t: {t}") 
        total = 0
        for _ in  t.values():
            total += _
        most_like_person =  id2name[max(t, key=t.get)]
        
        if total != 0:
            found_percentage = t[max(t, key=t.get)] /total
            return {"multipleFound": f"multiple name found, but most like person is {most_like_person} with % of {found_percentage*100}","personFound": [id2name[_] for _ in t]}
    
    # add_attendance(seat_no=seat_no,student_name=student_name)
    
    return {"found name in db": found_name, }




# @app.get("/attendance")
# async def get_attendance():
#     data = read_csv()
#     return JSONResponse(content=data)

# @app.get("/clear_attendence")
# async def clear_attendence():
#     clear_csv()
#     return {"status": "attendance cleared"}


@app.get("/api/v1/health")
async def health():
    return {"status": "health is good"}








