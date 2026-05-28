# Metric Learning for Face Recognition (Track 31)

Questo repository contiene il codice e la documentazione per un progetto di **Face Recognition** sviluppato nell'ambito dell'apprendimento di metriche (*Metric Learning*). Il sistema affronta il problema del riconoscimento facciale in uno scenario **Open-Set**, verificando la capacità del modello di generalizzare su soggetti e identità completamente inediti, non inclusi nel set di addestramento.

Il progetto mette a confronto due approcci principali:
1. **Baseline**: Estrazione di feature tramite una spina dorsale (backbone) ResNet-18 pre-addestrata su ImageNet e ottimizzata tramite Cross-Entropy Loss, combinata con una ricerca K-Nearest Neighbors (K-NN).
2. **Metric Learning**: Ottimizzazione geometrica dello spazio latente tramite una **Online Hard Triplet Loss** (*Batch-Hard*) applicata per forzare la coesione intra-classe e la separazione inter-classe.

---

## Architettura e Dettagli Implementativi

- **Backbone**: ResNet-18 (con rimozione del classificatore lineare finale e introduzione di un layer di normalizzazione L2 per proiettare gli embedding su un'ipersfera unitaria).
- **Strategia di Campionamento**: Implementazione di un `PKSampler` personalizzato in PyTorch che seleziona dinamicamente $P=8$ identità distinte e $K=4$ immagini per ciascuna identità per ogni batch (Batch Size effettiva = 32).
- **Hard Negative Mining**: Selezione online all'interno del batch del positivo più distante (*hardest positive*) e del negativo più vicino (*hardest negative*) rispetto all'ancora.
- **Valutazione del Retrieval**: Ricerca dei k-vicini tramite metrica del coseno e calcolo delle metriche di mAP (Mean Average Precision), Rank-1, Rank-5 e Rank-10 accuracy.

---

## Dataset e Suddivisione (Open-Set Split)

Il progetto utilizza un subset estratto dal dataset **CASIA-WebFace**, organizzato in modo rigoroso per l'open-set validation:
- **Training Set**: 160 identità distinte (pari all'80% del totale).
- **Test Set**: 40 identità distinte (pari al 20% del totale), completamente escluse dalla fase di training per testare la generalizzazione pura del sistema di retrieval.

---

## Risultati Sperimentali

Le performance di retrieval sono state valutate nello spazio latente tramite K-NN sia per la configurazione baseline che dopo il fine-tuning metrico:

| Configurazione Modello | mAP Globale | Rank-1 Accuracy | Rank-5 Accuracy | Rank-10 Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Fase 1: Baseline (ResNet-18 + CrossEntropy)** | **0.3872** | **75.67%** | **89.42%** | **93.10%** |
| Fase 2: Metric Learning (Online Triplet Loss) | 0.0406 | 23.28% | 48.20% | 61.62% |

### Nota di Analisi Critica
Il decremento prestazionale osservato nella Fase 2 costituisce un caso di studio emblematico sul **Model Collapse** causato dall'interazione tra batch size ridotte ($N=32$) e strategie *Batch-Hard*. Un batch troppo piccolo limita drasticamente lo spazio di ricerca dei negativi, generando gradienti stocastici rumorosi che degradano la topologia latente pre-esistente della spina dorsale.

---

## Struttura del Repository

```text
├── data/                    # Cartella contenente i subset del dataset
├── docs/
│   └── REPORTS.md           # Report scientifico e teorico dettagliato dei risultati
├── notebooks/
│   └── prog_dl2.ipynb       # Notebook principale contenente l'intera pipeline sperimentale
└── src/                     # Codice sorgente del progetto (moduli, sampler, loss)