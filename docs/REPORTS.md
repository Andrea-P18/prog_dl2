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
L'inferenza è stata condotta simulando un sistema di ricerca facciale reale: le rappresentazioni estratte sono state passate a un algoritmo K-Nearest Neighbors (K-NN) con metrica