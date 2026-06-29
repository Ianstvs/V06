"""
GIN — Graph Isomorphism Network
================================
Tarea: Clasificación de grafos (graph classification)
Dataset: MUTAG (moléculas + etiqueta de mutagenicidad)
Librería: PyTorch Geometric (PyG)
 
Nota: GIN es especialmente adecuado para clasificar GRAFOS COMPLETOS
      (a diferencia de GAT que suele usarse en clasificación de nodos).
 
Instalación:
    pip install torch torchvision
    pip install torch-geometric
"""
 
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
 
 
# ─── 1. Dataset ───────────────────────────────────────────────────────────────
# MUTAG: 188 moléculas, 2 clases (mutagénicas o no)
dataset = TUDataset(root="data/MUTAG", name="MUTAG")
dataset = dataset.shuffle()
 
print(f"Grafos     : {len(dataset)}")
print(f"Clases     : {dataset.num_classes}")
print(f"Features   : {dataset.num_node_features}")
 
# Split 80/20 train/test
n_train = int(len(dataset) * 0.8)
train_dataset = dataset[:n_train]
test_dataset  = dataset[n_train:]
 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
 
 
# ─── 2. Bloque MLP interno de GIN ─────────────────────────────────────────────
def make_mlp(in_channels, out_channels):
    """
    GINConv requiere un nn.Module que mapee hᵢ + Σhⱼ → nuevo embedding.
    Usamos un MLP de 2 capas con BatchNorm (recomendado en el paper original).
    """
    return Sequential(
        Linear(in_channels, out_channels),
        BatchNorm1d(out_channels),
        ReLU(),
        Linear(out_channels, out_channels),
        ReLU(),
    )






# ==============================================================================
# ==============================================================================
 
# ─── 3. Modelo GIN ────────────────────────────────────────────────────────────
class GIN(torch.nn.Module):
    def __init__(self, in_channels, 
                 hidden_channels, 
                 out_channels,
                 num_layers=5, 
                 dropout=0.5):
        super().__init__()
 
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
 
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            mlp   = make_mlp(in_ch, hidden_channels)
            # eps=0, train_eps=True → ε es un parámetro aprendible
            self.convs.append(GINConv(mlp, eps=0, train_eps=True))
            self.bns.append(BatchNorm1d(hidden_channels))
 
        # Clasificador final
        self.lin1    = Linear(hidden_channels, hidden_channels)
        self.lin2    = Linear(hidden_channels, out_channels)
        self.dropout = dropout
 
    def forward(self, x, edge_index, batch):
        # ── Capas GIN ──────────────────────────────────────────────────────
        for conv, bn in zip(self.convs, self.bns):
            # GINConv: hᵢ' = MLP( (1+ε)·hᵢ + Σⱼ∈N(i) hⱼ )
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
 
        # ── Readout: suma global de todos los nodos del grafo ──────────────
        # global_add_pool agrupa por grafo usando el tensor `batch`
        x = global_add_pool(x, batch)   # [num_grafos, hidden_channels]
 
        # ── Clasificador ───────────────────────────────────────────────────
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
 
 
 
 
# ==============================================================================
# ==============================================================================
 
# ─── 4. Entrenamiento ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
model = GIN(
    in_channels=dataset.num_node_features,
    hidden_channels=64,
    out_channels=dataset.num_classes,
    num_layers=5,
    dropout=0.5,
).to(device)
 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
 
 
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.batch)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_loader.dataset)
 
 
@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct = 0
    for batch in loader:
        batch = batch.to(device)
        out   = model(batch.x, batch.edge_index, batch.batch)
        pred  = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
    return correct / len(loader.dataset)
 
 
# ─── 5. Loop principal ────────────────────────────────────────────────────────
print("\nEntrenando GIN...")
for epoch in range(1, 101):
    loss     = train()
    scheduler.step()
    if epoch % 10 == 0:
        train_acc = evaluate(train_loader)
        test_acc  = evaluate(test_loader)
        print(f"Epoch {epoch:>3} | Loss {loss:.4f} | "
              f"Train {train_acc:.4f} | Test {test_acc:.4f}")
 
print(f"\nTest accuracy final: {evaluate(test_loader):.4f}")
 
 
# ─── 6. Ver ε aprendidos por capa ─────────────────────────────────────────────
print("\nValores de ε por capa (parámetro aprendido):")
for i, conv in enumerate(model.convs):
    print(f"  Capa {i+1}: ε = {conv.eps.item():.4f}")
