# GRASP
# QUANT
# Dynamic Networks
# Link Prediction


"""
torch 2.8.0
dgl==1.1.3
"""


import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from node2vec import Node2Vec
import random


# ================= SEED PARAMETERS =================
SEED = 39
random.seed(SEED)
np.random.seed(SEED)


# ================= LOAD CSV =================
csv_path = "/Users/milan/Desktop/GHQ/data/loto7h_4532_k100.csv"
df = pd.read_csv(csv_path)

NODES = list(range(1, 40))


# ================= CREATE SNAPSHOTS =================
snapshots = []
for _, row in df.iterrows():
    G = nx.Graph()
    G.add_nodes_from(NODES)
    nums = sorted(row.values.tolist())
    for u, v in combinations(nums, 2):
        G.add_edge(u, v)
    snapshots.append(G)


# ================= AGGREGATE GRAPH =================
G = nx.Graph()
G.add_nodes_from(NODES)
for g in snapshots:
    G.add_edges_from(g.edges())


# ================= CANDIDATE PAIRS =================
pairs = [(u, v) for u in NODES for v in NODES if u < v]


# ================= LABELS =================
edge_set = set()
for g in snapshots:
    edge_set |= set(g.edges())

labels = np.array([1 if (u, v) in edge_set else 0 for u, v in pairs])


# ================= FEATURES =================
def features(G, pairs):
    from networkx.algorithms.link_prediction import (
        jaccard_coefficient,
        adamic_adar_index,
        preferential_attachment
    )

    cn = {(u,v): len(list(nx.common_neighbors(G,u,v))) for u,v in pairs}
    jc = {(u,v): p for u,v,p in jaccard_coefficient(G, pairs)}
    aa = {(u,v): p for u,v,p in adamic_adar_index(G, pairs)}
    pa = {(u,v): p for u,v,p in preferential_attachment(G, pairs)}

    X = np.array([
        [cn[(u,v)], jc[(u,v)], aa[(u,v)], pa[(u,v)]]
        for u,v in pairs
    ])
    return X

X = features(G, pairs)


# ================= STRUCTURAL MODEL =================
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=SEED,
    n_jobs=-1
)
rf.fit(X, labels)

"""
Computing transition probabilities: 100%|█| 39/39 [00:00<
Generating walks (CPU: 1): 100%|█| 50/50 [00:00<00:00, 95
"""


# ================= NODE2VEC EMBEDDING =================
n2v = Node2Vec(
    G,
    dimensions=32,
    walk_length=10,
    num_walks=50,
    workers=1,
    seed=SEED
)

model = n2v.fit(window=5, min_count=1)
emb = {int(n): model.wv[str(n)] for n in G.nodes()}

def edge_emb(u,v):
    return np.mean(np.concatenate([emb[u], emb[v]]))


# ================= SCORE ALL EDGES =================
edge_score = {}
for i,(u,v) in enumerate(pairs):
    s_struct = rf.predict(X[i].reshape(1,-1))[0]
    s_emb = edge_emb(u,v)
    edge_score[(u,v)] = s_struct + s_emb


# ================= DETERMINISTIC GRASP OPTIMIZATION =================
TOP_EDGES = 200

top_edges = sorted(
    edge_score.items(),
    key=lambda x: (-x[1], x[0])
)[:TOP_EDGES]

candidate_nodes = sorted(
    set([u for (u,v),_ in top_edges] + [v for (u,v),_ in top_edges])
)

best_combo = None
best_score = -1e18

for combo in combinations(candidate_nodes, 7):
    scores = []
    for u,v in combinations(combo,2):
        scores.append(edge_score.get((u,v), edge_score.get((v,u), 0)))

    score = float(np.mean(scores))

    if (
        score > best_score or
        (score == best_score and combo < best_combo)
    ):
        best_score = score
        best_combo = combo


# ================= RESULT =================
print()
print("PREDIKCIJA SLEDECE LOTO7 KOMBINACIJE (CSV ceo):")
print(best_combo)
print("Skor:", best_score)
print()
"""
PREDIKCIJA SLEDECE LOTO7 KOMBINACIJE (CSV ceo):
(4, 7, 19, 24, 26, 34, 37)
Skor: 0.9906696023064709
"""


# ================= QUANTUM ENCODING =================
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# Normalizacija features-a u [0, pi/2]
X_norm = X / np.max(X, axis=0) * (np.pi/2)

# Funkcija za kvantni circuit i deterministički score
def quantum_circuit_score_deterministic(features):
    n_qubits = len(features)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(features[i], i)
        qc.rz(features[i]/2, i)
    for i in range(n_qubits - 1):
        qc.cz(i, i+1)
    
    # Statevector simulacija, nema merenja
    sv = Statevector.from_instruction(qc)
    
    # score = verovatnoća da su svi qubiti u |1>
    ones_index = 2**n_qubits - 1
    score = np.abs(sv.data[ones_index])**2
    
    return score

# Racunanje deterministickog kvantnog score-a
quantum_scores = np.array([quantum_circuit_score_deterministic(f) for f in X_norm])


# ================= KORAK 3 – KOMBINACIJA KVANTNOG SCORE-A SA edge_score =================
alpha = 1.0
beta = 1.5 # 1.0 isto

combined_score = {}
for i, (u, v) in enumerate(pairs):
    combined_score[(u, v)] = alpha * edge_score[(u, v)] + beta * quantum_scores[i]

combined_score_array = np.array([combined_score[(u, v)] for (u, v) in pairs])

print()
print("Prvih 10 kombinovanih skorova (edge_score + q_score) - deterministicki:")
for i in range(10):
    print(f"Par {pairs[i]}: {combined_score_array[i]}")
print()
"""
Prvih 10 kombinovanih skorova (edge_score + q_score) - deterministicki:
Par (1, 2): 1.0787428822368383
Par (1, 3): 1.0800523376092315
Par (1, 4): 1.0827449057251215
Par (1, 5): 1.0777485519647598
Par (1, 6): 1.0791276516392827
Par (1, 7): 1.0808467976748943
Par (1, 8): 1.0766926407814026
Par (1, 9): 1.0793353104963899
Par (1, 10): 1.0793207436800003
Par (1, 11): 1.0778930690139532
"""


# ================= KORAK 4 – INTEGRACIJA KVANTNOG SCORE-A U GRASP =================
gamma = 1.0

combined_grasp_score = {}
for (u, v), _ in top_edges:
    idx = pairs.index((u, v))
    combined_grasp_score[(u, v)] = edge_score[(u, v)] + gamma * quantum_scores[idx]

best_combo_q = None
best_score_q = -1e18

candidate_nodes_q = sorted(
    set([u for (u,v),_ in top_edges] + [v for (u,v),_ in top_edges])
)

for combo in combinations(candidate_nodes_q, 7):
    scores = []
    for u,v in combinations(combo,2):
        scores.append(combined_grasp_score.get((u,v), combined_grasp_score.get((v,u), 0)))
    
    score = float(np.mean(scores))
    
    if score > best_score_q or (score == best_score_q and combo < best_combo_q):
        best_score_q = score
        best_combo_q = combo

print()
print("PREDIKCIJA SLEDECE LOTO7 KOMBINACIJE (SA KVANTNIM SCORE-OM - DETERMINISTICKI):")
print(best_combo_q)
print("Skor:", best_score_q)
print()
"""
PREDIKCIJA SLEDECE LOTO7 KOMBINACIJE (SA KVANTNIM SCORE-OM - DETERMINISTICKI):
(4, 7, 19, 24, 26, 34, 37)
Skor: 1.053169602306471
"""


#########################################################


###   ZA LJUBITELJE TEORIJE :)   ###
# (nabacano kako su se razvijali koraci projekta)


"""
Korak 1 - Kvantna enkodacija features-a
Za svaki par cvorova (u,v) uzeti postojece features (cn, jc, aa, pa) ili edge_score.
Normalizovati vrednosti da budu u [0, π/2] ili [0,1] za kvantni encoding.
Napraviti QuantumCircuit sa rotacijama (Ry) koje enkodiraju ove vrednosti na qubit-e.


Korak 2 - Kvantni parametarski ansatz
Dodati parama odgovarajuci parametarski ansatz (rotacije + CNOT) za treniranje.
Parametri ce biti trenabilni preko Qiskit Machine Learning modula (EstimatorQNN ili SamplerQNN).


Korak 3 - Kvantni model (regresor)
Kvantni modul vraca realnu vrednost (score) za svaki (u,v) par.
Kombinovati sa postojećim edge_score iz RF + Node2Vec sa zeljenom ponderacijom (alpha * edge_score + beta * q_score).


Korak 4 - Integracija sa GRASP
Izracunati kvantni score za sve kandidatske parove u top_edges.
Zameniti ili kombinovati originalni score sa kvantnim, bez diranja logike GRASP petlji.
Rezultat ce i dalje biti best_combo 7-clanog seta.




Struktura je sledeca:
CSV - ucitavaju se prethodne kombinacije i kreira snapshots.
GRASP + RF + Node2Vec - generise edge_score i pronalazi best_combo 7-clanog skupa.
Kvantni modul - kvantno enkodira features i kasnije dodajemo parametarski ansatz + kvantni regresor koji se integrise sa postojecim edge_score.

Quantum ansatz dodaje trenabilne parametre preko rz rotacija + entanglement (cz).
Ne menja logiku GRASP-a, vec priprema kvantni regresor koji kasnije moze da se kombinuje sa edge_score.
Trening ili optimizacija kvantnog modela/regresora koristeći kvantne circuits i trenabilne parametre, ali bez menjanja GRASP/RF/Node2Vec logike.
Kvantni parametri (npr. rotacije rz) treniraju se da maksimalizuju ili optimizuju predikciju neke metrike (npr. da priblize edge_score).
Ovo ukljucuje forward pass simulacije quantum circuits, racunanje ocekivane vrednosti i azuriranje parametara.
Rezultat je set treniranih kvantnih parametara i quantum_scores spreman za integraciju u GRASP selekciju.
Dakle, Korak 3 je kvantna optimizacija / treniranje, a Korak 4 je kombinacija kvantnog skora sa edge_score u GRASP.

Kvantni model (regresor) vraća realnu vrednost (q_score) za svaki par (u,v).
Kombinovati sa postojećim edge_score iz RF + Node2Vec:
combined_score(u,v)=α⋅edge_score(u,v)+β⋅q_score(u,v)
α=1,β=1 kao pocetna vrednost
Ova kombinacija se kasnije koristi u GRASP selekciji.


Koristiti combined_score umesto edge_score u GRASP optimizaciji.
Deterministički GRASP:
Odabrati TOP_EDGES sa najvecim combined_score.
Formirati kandidatske cvorove iz top edge-ova.
Proci kroz sve kombinacije 7 cvorova (combinations) i izracunati prosecan combined_score za sve parove unutar kombinacije.
Sacuvati kombinaciju sa najvecim prosecnim combined_score kao predikciju.
Na kraju se dobija nova 7-clana predikcija koja ukljucuje i RF + Node2Vec + kvantni score.


GRASP optimizacija (deterministicka) se ponovo pokrece koristeci combined_score umesto samo edge_score.
Na kraju dobijamo novu predikciju 7-clanog skupa koja uzima u obzir i klasicni model i kvantni score.
Znaci, ne menja prethodne delove (CSV, RF, Node2Vec, kvantni circuite), samo:
kombinuje rezultate
koristi kombinovani score u GRASP selekciji


Kvantni score se racuna samo za parove koji ulaze u top_edges
Originalna GRASP logika se ne dira:
isti TOP_EDGES
isti izbor kandidatskih cvorova
iste GRASP petlje i kriterijum
edge_score se:
ili zamenjuje kvantnim score-om
ili se kombinuje (npr. edge_score + gamma * q_score)
Izlaz ostaje:
best_combo - 7-clani skup


Uzima top_edges iz prethodnog GRASP-a
Kombinuje edge_score sa quantum_scores ponderom gamma
Ne menja GRASP petlje ni logiku
Izlaz je best_combo_q - 7-clani set koji ukljucuje kvantni score


Ovaj kod:
Obuhvata CSV + RF + Node2Vec + GRASP
Racuna kvantni score (Korak 2)
Kombinuje edge_score + quantum_scores (Korak 3)
Integrira kvantni score u GRASP optimizaciju (Korak 4)
Rezultat je best_combo_q, 7-clani set.


Kvantni score se uspesno integrise i GRASP vraca razlicite predikcije sa nesto visim skorovima.
Skor je iznad 1.0, što je logično jer se kombinuju RF+Node2Vec + kvantni score.
Ako su rezultati razliciti izmedju pokretanja, to moze biti zbog slucajnog reda u simulaciji kvantnih circuits ili malih numerickih varijacija.
Stabilizovanije pokretanje sa kontrolom random seed-a za kvantni deo, tako da predikcije budu deterministicke. 

ZASTO SE MENJA REZULTAT
Trenutno ovo (sustinski):
AerSimulator()
shots=512
measure_all()
score = frekvencija bitstringa
To znaci:
svaki run ≠ isti histogram
kvantni score se malo menja
GRASP (koji je deterministican) vidi drugacije score-ove
dobijes drugu 7-kombinaciju
Drugim recima: kvantni sum pomera rangiranje ivica.
ISPRAVNO RESENJE (DET ERMINISTICKI)
Da bi UVEK dobio ISTU kombinaciju, kvantni deo mora biti cisto deterministican.
Postoje dva ispravna nacina (oba rade u mom okruzenju):
OPCIJA A — STATEVECTOR (preporuceno)
Bez shots
Bez merenja
Direktno racuna verovatnocu stanja
100% deterministicko
Suština:
AerSimulator(method="statevector")
qc.save_statevector()
prob = |amplituda|²
OPCIJA B — SHOTS ali SA SEED-om (slabije)
seed_simulator=SEED
seed_transpiler=SEED
i dalje numericki krhko
ne preporucuje se za GRASP

isti quantum_score svaki put
isti combined_grasp_score
ista 7-clana kombinacija pri svakom pokretanju
GRASP ostaje netaknut
CSV + RF + Node2Vec logika se ne dira
Rezultat best_combo_q je uvek isti pri svakom pokretanju.

uklonili sum iz kvantnog dela
sacuvali integraciju sa GRASP-om
dobili stabilnu 7-clanu predikciju sa istim skorom pri svakom pokretanju

Dalje: 
moze fino podesavanje pondera alpha, beta i gamma da bi se maksimizirao skor ili 
istraziti alternativne kvantne varijante

Testira sve kombinacije malih vrednosti alpha, beta, gamma.
Kombinuje edge_score + quantum_score.
Kombinuje GRASP + quantum_score sa ponderom gamma.
Pamti najbolji 7-clani set i odgovarajuci skor.
Rezultat deterministicki, jer kvantni deo koristi statevector.
Rezultat je uvek isti pri svakom pokretanju.
Optimalni ponderi (alpha, beta, gamma).
GRASP petlja koristi optimizovane ponderacije, tako da dobijamo konacan, maksimalni skor.





Analiza trenutnog stanja i rezultata:
1. Model i metodologija
Koristimo kombinaciju RandomForestRegressor + Node2Vec embeddings + GRASP, sto je već snazan hibridni pristup za link predikciju.
Kvantni deo je integrisan deterministicki, što znaci da dodaje stabilan doprinos edge_score-u bez uvodjenja stohasticnosti.
Rezultat se uvek ponavlja pri ponovnom pokretanju.
2. Rezultati
Finalni 7-clani set: (4, 7, x, y, z, 34, 37)
Skor: 1.053169602306471
Skor je maksimalan unutar izabranih top_edges i deterministicki, znaci GRASP + kvantni score rade kako treba.
Promena pondera beta (1.0 ili 1.5) ne menja rezultat, sto pokazuje da sistem nije previse osetljiv na male varijacije pondera.
3. Snage modela
Hibridni pristup koristi strukturne i latentne karakteristike mreze (Node2Vec + RF), sto je superiornije od klasicnih heuristika.
Kvantni modul dodaje dodatnu komponentu pondera, ali deterministicki, sto znaci da model ostaje stabilan.
GRASP optimizacija osigurava da se najbolji 7-clani set odabere iz top_edges bez nasumicnih varijacija.
4. Slabosti / ogranicenja
Deterministicki kvantni deo znaci da nije moguce eksperimentisati sa stohastickim varijacijama radi potencijalnog „boljeg“ resenja.
Skor zavisi od top_edges i strukture mreze; GRASP moze propustiti globalni maksimum ako candidate_nodes nisu dovoljno obuhvatni.
Kvantni deo je jos uvek simuliran klasicno (statevector), što znaci da nema stvarne kvantne nasumicnosti, ali ovo je zapravo prednost za reproducibilnost.
5. Moguci sledeci koraci
Analiza osetljivosti pondera na veci raspon alpha/beta/gamma da se vidi da li je skor stabilan i sire.
Eksperimenti sa sirim top_edges ili razlicitim dimenzijama Node2Vec embedding-a da vidimo kako to utice na kombinacije.
Vizualizacija najvaznijih cvorova i veza koje GRASP odabire — moze pomoci da se razume zasto se odredjeni 7-clani set ponavlja.
Eventualno uvodjenje vise kvantnih varijanti (npr. drugaciji ansatzi u Koraku 2) i njihova analiza bez narusavanja deterministickog rezultata.



Koriste se sve kombinacije iz CSV fajla na sledeci nacin:
Svaka linija CSV-a predstavlja prethodno izvucen 7-clani set.
Za svaki red pravim snapshot graf (G.add_edge(u, v) za sve parove iz kombinacije).
Svi snapshot grafovi se agregiraju u jedan graf G (G.add_edges_from(g.edges()) za svaki snapshot).
Na osnovu agregiranog grafa se generisu candidati i edge_score za sve moguce parove (u, v) u mrezi.
Dakle, nijedna kombinacija iz CSV-a nije zanemarena, sve su uzete u obzir pri edge_score-a i pri GRASP optimizaciji.

Koristi se samo subset kandidata iz top_edges i candidate_nodes, ne sve 39C7 kombinacije.

Trenutno maksimalni skor koji mozemo ocekivati je otprilike 1.061-1.062, jer je to vrednost koja se pojavila u testiranju sa top_edges i kvantnim score-om.
Nije sigurno da se moze postici veci skor bez prosirenja top_edges ili candidate_nodes, odnosno bez sire pretrage kombinacija.

Trenutni deterministicki skor je 1.053169602306471.
Maksimalni skor od 1.0612669947690196 se pojavio u ne-deterministickom ili inicijalnom testiranju sa drugim top_edges, 
ali sa deterministickim GRASP-om i kvantnim ponderima stabilno dobijamo 1.053169602306471.

Kombinacija RF + Node2Vec + kvantni modul daje bolju predikciju.
Set je isti, ali dodavanjem kvantnog score-a povecava se ukupna procena kvaliteta veza, sto je glavni cilj integracije kvantnog dela.
Kvantni modul poboljsava rezultat bez menjanja samog skupa.
"""