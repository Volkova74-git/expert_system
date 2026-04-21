import pickle

with open("standards_texts.pkl", "rb") as f:
    chunks = pickle.load(f)

with open("chunks_dump.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"\n{'='*60}\n")
        f.write(f"INDEX: {i}\n")
        f.write(f"TEXT: {chunk}\n")