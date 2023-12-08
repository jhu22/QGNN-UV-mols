import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt


def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol
def generate_2d_coordinates(mol):
    AllChem.Compute2DCoords(mol)
    return mol
def extract_atom_features(mol):
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())
    return atom_features
def extract_bond_features(mol):
    bond_features = []
    for bond in mol.GetBonds():
        bond_features.append(bond.GetBondTypeAsDouble())
    return bond_features
def create_graph(mol):
    G = nx.Graph()
    
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondTypeAsDouble())

    return G
def smiles_to_graph(smiles):
    mol = smiles_to_mol(smiles)
    mol = generate_2d_coordinates(mol)
    atom_features = extract_atom_features(mol)
    bond_features = extract_bond_features(mol)
    G = create_graph(mol)
    return G, atom_features, bond_features
def visualize_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=8)
    edge_labels = nx.get_edge_attributes(G, 'bond_type')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.show()

from torch_geometric.data import Data
def graph_to_pyg_data(G):
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    x = torch.tensor(list(G.nodes)).view(-1, 1).float()

    data = Data(x=x, edge_index=edge_index)

    return data
