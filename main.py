from fastapi import FastAPI, UploadFile, File, HTTPException
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
    files = glob.glob("dataset/Original Images/Original Images/*/*")
    print(f"files: {files[0]}")
    print(f"files len: {len(files)}")
    if os.path.exists(embed_file_name):
        embeddings = np.load(embed_file_name)
        print(f"embedding file loaded...")
    else:
        for file in tqdm(files):
            img = Image.open(file)
            img = mtcnn(img).to(device)
            embedding = resnet(img.unsqueeze(0)).cpu().detach().numpy()
            if embeddings is None:
                embeddings = embedding
            else:
                embeddings = np.concatenate((embeddings, embedding), axis=0)  
        print(f"new embeddings completed... ")
        np.save(embed_file_name, embeddings)
        print("embeding saving completed")
    print(f"embedding shape: {embeddings.shape}")
      # loading the user with id's
    labels = []
    for f in files:
        s  = f.split("/")[-1].split("_")[0]
        labels.append(s)
    print(f"labels len: {len(labels)}")
        
    if os.path.exists(id2name_file_name):
        id2name = np.load(id2name_file_name, allow_pickle=True).item()
        print(f"id2name file loaded...")
    else:
        for i,e in enumerate(set(labels)):
            id2name[i] = e
    
    name2id = {id2name[i]:i for i in id2name}
    print(f"id2name: {id2name}")
    
    label_with_ids = []
    for i in range(0, len(files)):
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
    D, I = index.search(embedding, 50)  # Search for the closest match
    
    found_name = id2name[I[0][0]]
    print(f"found name: {found_name}")
    print(f"D: {D}")
    print(f"I: {I}")
    print(f"find in id2name file: {id2name[I[0][0]]}")
    if len(list(set(I[0])))>1:
        print(f"found multiple: {set(I[0])}")
        t = {}       
        for i in I[0]:
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
    return {"name": found_name, }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Face Recognition API"}





