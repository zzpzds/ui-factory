from transformers import AutoTokenizer, AutoModel
import torch
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = "/Users/didi/.cache/modelscope/microsoft/codebert-base"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

html = """
<html><body>
  <nav style="background:#333; height:60px">
    <a href="/" style="color:white; font-size:18px">Logo</a>
  </nav>
  <main style="padding:20px">
    <img src="hero.jpg" style="width:600px; height:300px"/>
    <h1 style="font-size:32px; color:#333">欢迎</h1>
    <button style="background:#1890ff; color:white; width:120px; height:40px">立即使用</button>
  </main>
</body></html>
"""
soup = BeautifulSoup(html, "html.parser")
nodes = soup.find_all()
# 排除html、body标签
nodes = [str(node) for node in nodes if node.name not in ["html", "body"]]

batch_inputs = tokenizer(
    nodes, return_tensors="pt", padding=True, truncation=True, max_length=128
)
with torch.no_grad():
    outputs = model(**batch_inputs)

node_features = outputs.last_hidden_state
node_features_np = node_features.numpy()

sim_matrix = []
for i in range(len(nodes)):
    line_matrix = []
    for j in range(len(nodes)):
        sim = cosine_similarity(
            node_features_np[i], node_features_np[j]
        )[0][0]
        line_matrix.append(sim)
    sim_matrix.append(line_matrix)
print(sim_matrix)
