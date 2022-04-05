# CDSF-Transferability
The source code of our ICASSP 2022 paper: Exploring Transferability Measures and Domain Selection in Cross-Domain Slot Filling.


* Personal Homepage
* Basic Introduction
* Code Files
* Running Tips
* Citation

## Personal Homepage
  * [Homepage](https://www.lamda.nju.edu.cn/lixc/)

## Basic Introduction
  * During Cross Domain Slot Filling (CDSF), when does a source model help the target task?
  * If a source task contains slot types that the target one does not own (non-targeted slots) or the target has new slot types to identify (zero-shot slots), does the transfer process become better or worse?
  * We implement CDSF on the Snips \[[1](https://dblp.org/rec/journals/corr/abs-1805-10190.html?view=bibtex)\] benchmark which contains 7 domains, and investigate the domain transferability and negative transfer in CDSF.

## Code Files
 * `paths.py`    Data file, log file and other file paths
 * `snips_data.py`    Process and load Snips data, and SnipsDataset (torch.utils.data.Dataset)
 * `plot_data.py`    Plot data information
 * `model.py`    Slotfilling network architectures (torch.nn.Module), including both Coarse and Coach SF models as introduced in our paper
 * `slot_filling.py`    Coarse SF class, including train and test functions
 * `slot_filling_coach.py`    Coach SF class, including train and test functions
 * `crf.py`    Implementation of CRF
 * `text.py` & `utils.py` & `tools.py`    Some helper functions
 * `train_dyn.py`    Train Coarse CDSF via dynamic transfer, i.e, sequentially adding source domains sorted by the shared slot numbers
 * `train_dyn_coach.py`    Train Coach CDSF via dynamic transfer, i.e, sequentially adding source domains sorted by the shared slot numbers
 * `analyze_dyn.py`    Analyze and plot the logged results

## Running Tips
  * Snips Data: the utilized data is the same as and downloaded from [Coach](https://github.com/zliucr/coach)
  * Word Embeddings: we use both word-level \[[2](https://dblp.org/rec/journals/tacl/BojanowskiGJM17.html?view=bibtex)\] and character-level \[[3](https://dblp.org/rec/conf/emnlp/HashimotoXTS17.html?view=bibtex)\] embeddings to obtain 400d vectors for tokens and slot descriptions, the detailed implementations could be found in `text.py` (we use the provided embeddings in torchtext)
  * Package Versions: python==3.7.3, nltk=3.6.2, torch==1.9.0, torchtext==0.10.0
  * Running: directly run `train_dyn.py` or `train_dyn_coach.py`, and then run `analyze_dyn.py` to show the results

## Citation
  * Xin-Chun Li, Yan-Jia Wang, Le Gan, De-Chuan Zhan. Exploring Transferability Measures and Domain Selection in Cross-Domain Slot Filling. In: Proceedings of the 2022 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP'2022), online conference, Singapore, 2022.
  * \[[BibTex](https://dblp.org/pid/246/2947.html)\]
