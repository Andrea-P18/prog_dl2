# Apprendimento di Metriche per il Riconoscimento Facciale: Un'Analisi sulla Triplet Loss

**Repository:** [Link alla repo]

---

## 1. Abstract
Questo progetto affronta la complessa sfida del riconoscimento facciale in un contesto *Open-Set*, dove le identità presentate in fase di inferenza (test set) sono completamente disgiunte da quelle viste durante l'addestramento. L'obiettivo è apprendere una mappatura da immagini ad alta dimensionalità verso uno spazio latente compatto in cui la distanza del coseno rifletta fedelmente la somiglianza semantica tra le identità. Il progetto mette in contrasto una baseline di estrazione feature (addestrata come classificatore su ResNet-18) con un approccio geometrico di Metric Learning basato sulla *Online Hard Triplet Loss*.

## 2. Fondamenti Teorici
Nel riconoscimento facciale moderno, il task non è classificare, ma imparare una funzione di embedding $f(x) \in \mathbb{R}^d$. 
La **Triplet Loss** si propone di ottimizzare direttamente le distanze relative nello spazio latente analizzando una tripletta composta da un'Ancora ($x_a$), un Positivo della stessa identità ($x_p$) e un Negativo di un'identità diversa ($x_n$):

$$\mathcal{L}_{triplet} = \max(0, d(x_a, x_p) - d(x_a, x_n) + \alpha)$$

dove $\alpha = 0.5$ rappresenta il margine imposto per separare i cluster.
Per massimizzare il segnale dei gradienti, è stata adottata una strategia di **Hard Negative Mining** dinamica (*Batch-Hard*): per ogni ancora all'interno del batch, la rete individua il positivo più distante (hardest positive) e il negativo più vicino (hardest negative), concentrando l'addestramento solo sulle triplette che violano il margine.

## 3. Architettura e Setup Sperimentale
L'intero framework è stato sviluppato in PyTorch.

* **Backbone:** ResNet-18 pre-addestrata (pesi ImageNet), con il layer finale completamente connesso sostituito da una funzione di normalizzazione L2. Questa proiezione forza tutti gli embedding a risiedere su un'ipersfera unitaria, rendendo la distanza Euclidea direttamente proporzionale alla similarità del coseno.
* **Dataset e Split:** È stato estratto un subset dal dataset CASIA-WebFace. Per simulare un ambiente reale (*Open-Set*), il dataset è stato diviso con una proporzione 80/20 basata sulle identità:
    * **Train Set:** 160 identità (16.000 immagini)
    * **Test Set:** 40 identità disgiunte (4.000 immagini)
* **Campionamento (*P-K Sampler*):** Fondamentale per la Triplet Loss, è stato creato un Dataloader personalizzato che garantisce la presenza di $P=8$ classi distinte e $K=4$ campioni per classe in ogni step (Batch Size effettiva = 32).

**Iperparametri:**
* *Fase 1 (Baseline):* Ottimizzatore Adam ($lr=1 \times 10^{-3}$), Cross-Entropy Loss, 15 epoche, LR Scheduler a gradino.
* *Fase 2 (Metric Learning):* Ottimizzatore Adam ($lr=1 \times 10^{-4}$), Hard Triplet Loss ($\alpha=0.5$), 15 epoche. I layer iniziali del backbone sono stati congelati per tentare di stabilizzare i gradienti derivanti dal mining.

## 4. Valutazione del Retrieval
L'inferenza è stata condotta simulando un sistema di ricerca facciale reale: le rappresentazioni estratte sono state passate a un algoritmo K-Nearest Neighbors (K-NN) con metrica *cosine*, calcolando la Mean Average Precision (mAP) e l'accuratezza ai primi Rank.

| Modello / Metodo | mAP Globale | Rank-1 Acc | Rank-5 Acc | Rank-10 Acc |
| :--- | :--- | :--- | :--- | :--- |
| **Fase 1: ResNet-18 (Baseline)** | **0.3872** | **75.67%** | **89.42%** | **93.10%** |
| Fase 2: Fine-Tuning con Triplet Loss | 0.0406 | 23.28% | 48.20% | 61.62% |

### 4.1 Analisi del Comportamento Anomalo
Il crollo drastico delle prestazioni (mAP dal 38% al 4%) rappresenta un tipico "edge case" della topologia spaziale forzata ed è ascrivibile a tre fattori congiunti:
1. **Dinamiche del Batch Size:** Nel Metric Learning, la qualità dell'Hard Mining scala con la grandezza del batch. Un batch size di 32 espone l'ancora a soli 28 negativi possibili. Questi negativi sono spesso facili (easy negatives) e non offrono un gradiente utile, o alternativamente introducono rumore stocastico elevato che distrugge la struttura appresa dalla baseline.
2. **Backbone Congelato:** Aver impedito l'aggiornamento dei primi layer convoluzionali ha limitato la capacità del modello di ri-adattare l'estrazione delle feature geometriche del viso necessarie per assecondare la spinta violenta dell'equazione di margine.
3. **Collasso dello Spazio (*Model Collapse*):** Spesso la combinazione di batch limitati e *Batch-Hard mining* induce la rete a schiacciare tutti gli embedding in un singolo punto per minimizzare banalmente la loss spaziale. 

## 5. Cluster Analysis (t-SNE)
La topologia latente del Test Set è stata analizzata riducendo le dimensioni degli embedding ($d=512$) a uno spazio 2D tramite l'algoritmo t-SNE (con Perplexity=30). 
Coerentemente con i dati di retrieval, la proiezione post-Triplet Loss mostra una marcata perdita di coesione intra-classe. Le identità non viste in fase di training non riescono a formare cluster densi e si osserva un amalgama disperso causato dalla corruzione dei pesi durante il fine-tuning.

## 6. Conclusioni e Lavori Futuri
Il progetto dimostra la complessità dell'ottimizzazione metrica pura rispetto ai gradienti più gentili della classificazione classica. Per il futuro, la roadmap prevede:
* L'implementazione di loss avanzate basate sui margini angolari (come **ArcFace** o **CosFace**), che permettono di addestrare lo spazio latente comportandosi come una Cross-Entropy standard, eliminando la necessità di campionamenti complessi (P-K Sampler) e aggirando i limiti hardware legati alle grandi batch size.
* Aumento del Batch Size (a 256 o superiore) in combinazione con una strategia di mining *Semi-Hard*, che garantisce gradienti positivi continui senza destabilizzare la rete.