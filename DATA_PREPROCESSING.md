# Data Preprocessing Documentation

This document explains the data pipeline implemented in [data_preprocessing.py](file:///home/vaibhav/Documents/xai/mutag_project/data_preprocessing.py), detailing how raw semantic web data is converted into graph representations suitable for deep learning.

---

## 1. Input Data Structure (Semantic Web / RDF)

The pipeline starts with raw relational molecular data represented as a Resource Description Framework (RDF) Knowledge Graph:
* **File**: `mutag-hetero/mutag_stripped.nt` (N-Triples format).
* **Format**: A set of semantic triples in the form of `(subject, predicate, object)`.
* **Semantics**:
  * Molecules are subjects that link to atoms via the predicate `http://dl-learner.org/carcinogenesis#hasAtom`.
  * Molecules link to bonds via the predicate `http://dl-learner.org/carcinogenesis#hasBond`.
  * Bonds link to their participating atoms via `http://dl-learner.org/carcinogenesis#inBond`.
  * Atoms and bonds are assigned specific chemical types (classes) via `http://www.w3.org/1999/02/22-rdf-syntax-ns#type`.
* **Labels**: Two Tab-Separated Value (TSV) files (`trainingSet.tsv` and `testSet.tsv`) map molecule/bond URIs to a binary classification label (`label_mutagenic` $\in \{0, 1\}$), indicating whether the molecule is mutagenic (carcinogenic) or not.

---

## 2. Processing Pipeline

The script [data_preprocessing.py](file:///home/vaibhav/Documents/xai/mutag_project/data_preprocessing.py) executes the following sequential steps to parse and convert the knowledge graph:

### Step A: RDF Graph Parsing
The script utilizes the `rdflib` library to load and parse the raw `.nt` file (`mutag_stripped.nt`) into an in-memory graph structure. 
* **Parsing mechanism**: The `.nt` file contains flat line-by-line statements of triples represented as:
  ```text
  <subject_uri> <predicate_uri> <object_uri> .
  ```
* `rdflib.Graph().parse()` reads these lines and compiles them into a searchable network database where we can query relationships by providing any combination of subject, predicate, or object.

### Step B: Molecule Subgraph Extraction
Because the raw RDF file is a flat set of triples representing all molecules mixed together, the script extracts isolated graph representation for each molecule defined in the TSV splits. Given a molecule's URI (for example, `http://dl-learner.org/carcinogenesis#d50`), it queries the RDF graph to extract its constituent components as follows:

1. **Querying and Extracting Atoms**:
   * **Triples matched**: The script looks up all triples matching the pattern:
     ```text
     (molecule_uri, CARC.hasAtom, ?atom_uri)
     ```
     *Example:* `(<...#d50>, <...#hasAtom>, <...#d50_1>)` $\rightarrow$ Extracts `<...#d50_1>` as an atom URI.
   * **Extracting Atom Types**: For each extracted atom URI, it queries its chemical class using the RDF type predicate:
     ```text
     (?atom_uri, RDF.type, ?atom_type)
     ```
     *Example:* `(<...#d50_1>, <...type>, <...#Carbon-10>)` $\rightarrow$ Extracts `Carbon-10` as the atom type.
   * **Storage**: Atoms are stored in a sorted list, mapping each atom URI to a unique local index (e.g. `0, 1, 2, ...`).

2. **Querying and Extracting Bonds**:
   * **Triples matched**: The script looks up all triples matching the pattern:
     ```text
     (molecule_uri, CARC.hasBond, ?bond_uri)
     ```
     *Example:* `(<...#d50>, <...#hasBond>, <...#d50_b1>)` $\rightarrow$ Extracts `<...#d50_b1>` as a bond URI.
   * **Extracting Bond Types**: For each bond URI, it queries its category/relation class:
     ```text
     (?bond_uri, RDF.type, ?bond_type)
     ```
     *Example:* `(<...#d50_b1>, <...type>, <...#Bond-1>)` $\rightarrow$ Extracts `Bond-1` as the bond relation type.
   * **Extracting Endpoints**: It resolves which two atoms the bond connects by querying:
     ```text
     (?bond_uri, CARC.inBond, ?endpoint_atom)
     ```
     This query yields exactly 2 triples matching the bond. The objects of these triples are the two endpoint atom URIs (e.g., `<...#d50_1>` and `<...#d50_2>`).
   * **Mapping**: The script maps these endpoint atom URIs to their corresponding 0-based local indices created during the atom extraction step. The bond is then represented as a tuple: `(atom1_index, atom2_index, bond_type_name)`.

### Step C: Categorical Encoding
To prepare the extracted subgraphs for neural network execution, categorical labels are mapped to integer features:
* **Node Encoder**: Builds an index map of all 66 unique atom type categories.
* **Edge/Relation Encoder**: Builds an index map of all 4 unique relation (bond) categories.
* **One-Hot Featurization**: Encodes each atom's class into a one-hot feature vector of dimension `[66]`.

### Step D: PyTorch Geometric Format Mapping
Each molecule's structured features are compiled into a PyTorch Geometric (`PyG`) `Data` object:
* **`x`**: Node feature tensor of shape `[num_atoms, 66]`, representing one-hot encoded atom types.
* **`edge_index`**: Bidirectional coordinate format (COO) tensor of shape `[2, 2 * num_bonds]`, mapping the connectivity of the molecule graph.
* **`edge_type`**: Relational edge type index tensor of shape `[2 * num_bonds]`, categorizing the bond types.
* **`y`**: Target label tensor containing the mutagenic classification label `[0]` or `[1]`.
* **Metadata**: Appends additional metadata (`molecule_id`, `atom_info`, `bond_info`) for downstream explanation and visualization purposes.

---

## 3. Output Dataset (Serialized Tensors)

* **File**: `mutag-hetero/pyg_dataset.pt`
* **Format**: A serialized list of PyTorch Geometric `Data` objects.
* **Usage**: This file is loaded directly by [RGCN_Train.py](file:///home/vaibhav/Documents/xai/mutag_project/RGCN_Train.py), [RGCN_Kfold.py](file:///home/vaibhav/Documents/xai/mutag_project/RGCN_Kfold.py), and [GNNExplainer.py](file:///home/vaibhav/Documents/xai/mutag_project/GNNExplainer.py) to train and explain the GNN model without repeating the costly RDF-parsing operations.
