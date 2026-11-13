# Literature review

Written 2026-11-12 for the midterm report and the paper. Every reference below was checked against the publisher,
arXiv or the dataset repository (bibliographic details in the list at the end). Claims about a paper's results are
limited to what its abstract or main text states.

## 1. Laboratory versus field datasets

Most image-based plant-disease work builds on **PlantVillage**, about 54,000 leaf images of 14 crops photographed one
leaf at a time on a plain background [1]. Mohanty et al. [2] trained convolutional networks on it and reached 99.35%
accuracy on a held-out PlantVillage test set, but only about 31% on images collected from trusted online sources under
different conditions. They concluded that more diverse training data is needed. Ferentinos [3] likewise reported
very high accuracy on a large laboratory-dominated collection together with a marked drop when laboratory-trained
models were applied to field images. Barbedo [4] listed the factors that limit real-world use, among them covariate
shift between training and deployment images and the difficulty of separating symptoms from the background. Noyan [5]
showed that PlantVillage carries capture-related bias: a model using only eight background pixels classified it far
above chance (49% vs 2.6%).

Field and in-the-wild datasets were built in response. **PlantDoc** [6] collected 2,598 web images of 13 species with
cluttered backgrounds. **Tomato-Village** [7] photographed tomato plants in fields in Rajasthan, India, motivated by the
observation that PlantVillage-trained models do not work well in real fields. **PlantWild** [8] is the largest
in-the-wild collection (18,542 images, 89 classes) and stresses small differences between diseases and large variation
within one disease. Regional field datasets for tomato from Pakistan [9] and Bangladesh [10]–[12] are recent and, to our
knowledge, have no published benchmark. A common pattern is that each dataset is evaluated **within itself**; how a
model trained on one source behaves on another is rarely reported.

## 2. Hand-crafted features and classical machine learning

Before deep learning, disease recognition relied on segmentation followed by colour, texture and shape descriptors
[13]. Texture is usually described with grey-level co-occurrence statistics [14] and local binary patterns [15]; colour
with channel moments and histograms. Classifiers are typically support vector machines [16] and random forests [17].
For tomato, Mokhtar et al. [18] combined Gabor wavelet features with SVMs, Hlaing and Zaw [19] modelled SIFT-based
texture statistics together with colour moments, and Tan et al. [20] compared SVM, k-NN and random forests on
hand-crafted features with several CNNs. Singh and Misra [21] combined segmentation with soft-computing classifiers.
These studies report high accuracy, but almost all use PlantVillage images with random image-level splits, so they
measure performance on a single laboratory source. Classical pipelines remain attractive for low-resource settings
because they are cheap and their inputs are named, measurable quantities.

## 3. Robustness and distribution shift

Hendrycks and Dietterich [22] showed that image classifiers degrade under common corruptions such as blur, noise and
brightness change, which motivates controlled perturbation tests. Torralba and Efros [23] showed that every dataset has
its own "signature" and that models trained on one dataset transfer poorly to another. Domain-generalisation work
evaluates by holding out whole domains; Gulrajani and Lopez-Paz [24] showed that, with a careful leave-domain-out
protocol, many specialised methods do not beat plain empirical risk minimisation, and WILDS [25] collected real
distribution shifts for benchmarking. Classical remedies for lighting variation include grey-world colour constancy
[26] and contrast-limited adaptive histogram equalisation (CLAHE) [27]; data augmentation is the usual learning-based
remedy [28].

## 4. Shortcut learning, leakage and duplicates

Geirhos et al. [29] describe shortcut learning: models exploit cues that work on the benchmark but not on the intended
task. In medical imaging, models learned hospital- or device-specific signals [30], [31], and DeGrave et al. [31]
showed that good external-test accuracy alone does not prove that a model uses the pathology. Kapoor and Narayanan [32]
catalogue data leakage as a main cause of irreproducible machine-learning results, including duplicates across training
and test sets. Barz and Denzler [33] found that 3.3% of CIFAR-10 and 10% of CIFAR-100 test images have near-duplicates
in training. Roberts et al. [34] showed that when data are grouped or structured, blocked (grouped) cross-validation is
more appropriate than random splitting for predicting to new data. Label errors are a related problem; confident
learning [35] flags likely mislabelled examples from out-of-sample predicted probabilities.

## 5. Explanation of models and of features

Feature-importance measures explain a fitted model. Random-forest impurity importance is biased towards some feature
types [36], permutation importance can extrapolate to unrealistic inputs when features are correlated [37], and the
importance of a feature depends on which of many equally good models is chosen [38]. Local explainers such as LIME [39]
and SHAP [40] often disagree with each other [41]. Rudin [42] argues that for high-stakes decisions inherently
interpretable models should be preferred to post-hoc explanation. In plant pathology, Toda and Okura [43] visualised a
CNN trained on PlantVillage and found that it attends to lesion colour and texture, but on a single laboratory dataset.

**Concept-based explanation** expresses a model in human terms. TCAV [44] tests a network's sensitivity to
user-defined concepts, and concept bottleneck models [45] predict through a layer of named concepts. With
vision-language models such as CLIP [46], concept scores can be obtained without concept labels: Label-free CBM [47]
and LaBo [48] generate concepts with language models and score them with CLIP. In this project, models are trained and tested only with classical ML; we compare two kinds of learned
feature extractor. The classical one follows the bag-of-visual-words idea [52], [53]: codebooks of colour and
texture prototypes learned with k-means [54], [55]. The deep one uses frozen pretrained networks, OpenCLIP [49] and an
ImageNet ResNet-50 [50], purely as feature extractors.

## 6. Gap addressed by this project

(i) Classical tomato-disease studies report high accuracy on laboratory images with image-level splits; we evaluate on
field photos with grouped splits and quantify how much related images inflate a benchmark. (ii) Field datasets are
benchmarked within themselves; we test across eight image sources with leave-one-source-out evaluation and check
dataset provenance. (iii) Explanations are usually reported for one model on one dataset; we ask which features —
formula-based, learned by k-means, or extracted by a frozen deep network — keep the same effect across sources, and whether models restricted to
them transfer better.

## References

[1] D. P. Hughes and M. Salathé, "An open access repository of images on plant health to enable the development of
mobile disease diagnostics," arXiv:1511.08060, 2015.
[2] S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using deep learning for image-based plant disease detection,"
Frontiers in Plant Science, vol. 7, art. 1419, 2016, doi:10.3389/fpls.2016.01419.
[3] K. P. Ferentinos, "Deep learning models for plant disease detection and diagnosis," Computers and Electronics in
Agriculture, vol. 145, pp. 311–318, 2018.
[4] J. G. A. Barbedo, "Factors influencing the use of deep learning for plant disease recognition," Biosystems
Engineering, vol. 172, pp. 84–91, 2018, doi:10.1016/j.biosystemseng.2018.05.013.
[5] M. A. Noyan, "Uncovering bias in the PlantVillage dataset," arXiv:2206.04374, 2022.
[6] D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra, "PlantDoc: A dataset for visual plant disease
detection," in Proc. ACM IKDD CoDS and COMAD, 2020, pp. 249–253, doi:10.1145/3371158.3371196.
[7] M. Gehlot, R. K. Saxena, and G. C. Gandhi, "'Tomato-Village': a dataset for end-to-end tomato disease detection in
a real-world environment," Multimedia Systems, vol. 29, pp. 3305–3328, 2023, doi:10.1007/s00530-023-01158-y.
[8] T. Wei, Z. Chen, Z. Huang, and X. Yu, "Benchmarking in-the-wild multimodal disease recognition and a versatile
baseline," in Proc. 32nd ACM Int. Conf. Multimedia (MM '24), 2024, pp. 1593–1601, doi:10.1145/3664647.3680599.
[9] M. N. Malik, S. H. Ali, and H. Ali, "Tomato leaf disease classification dataset in Pakistan," Mendeley Data, v2,
2026, doi:10.17632/3mbnb82mxd.2.
[10] P. Bapari, M. Z. R. Zitu, and M. Ummehani, "A high-resolution image dataset of tomato (Solanum lycopersicum) leaves
for multi-class disease detection and classification from Bangladesh," Mendeley Data, v1, 2025,
doi:10.17632/74zvtzx9pr.1.
[11] M. K. Oni and T. T. Prama, "A comprehensive dataset of tomato leaf images for disease analysis in Bangladesh," Data
in Brief, 2025; data: "Tomato leaf image dataset for disease analysis in real-world environment," Mendeley Data, v1,
2024, doi:10.17632/rnbsw72zb5.1.
[12] A. Imtiaz, F. B. I. Swapnil, S. R. Masud, and D. Karmaker, "Tomato leaf dataset: A dataset for multiclass disease detection and classification," Data in
Brief, 2025, doi:10.1016/j.dib.2025.111520; data: Mendeley Data, doi:10.17632/bpfd9cns5g.2.
[13] J. G. A. Barbedo, "Digital image processing techniques for detecting, quantifying and classifying plant
diseases," SpringerPlus, vol. 2, art. 660, 2013, doi:10.1186/2193-1801-2-660.
[14] R. M. Haralick, K. Shanmugam, and I. Dinstein, "Textural features for image classification," IEEE Trans. Systems,
Man, and Cybernetics, vol. SMC-3, no. 6, pp. 610–621, 1973.
[15] T. Ojala, M. Pietikäinen, and T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture
classification with local binary patterns," IEEE Trans. Pattern Anal. Mach. Intell., vol. 24, no. 7, pp. 971–987,
2002.
[16] C. Cortes and V. Vapnik, "Support-vector networks," Machine Learning, vol. 20, pp. 273–297, 1995.
[17] L. Breiman, "Random forests," Machine Learning, vol. 45, pp. 5–32, 2001.
[18] U. Mokhtar, M. A. S. Ali, A. E. Hassanien, and H. Hefny, "Tomato leaves diseases detection approach based on
support vector machines," in Proc. 11th Int. Computer Engineering Conf. (ICENCO), 2015,
doi:10.1109/ICENCO.2015.7416356.
[19] C. S. Hlaing and S. M. M. Zaw, "Tomato plant diseases classification using statistical texture feature and color
feature," in Proc. IEEE/ACIS 17th Int. Conf. Computer and Information Science (ICIS), 2018, pp. 439–444,
doi:10.1109/ICIS.2018.8466483.
[20] L. Tan, J. Lu, and H. Jiang, "Tomato leaf diseases classification based on leaf images: A comparison between
classical machine learning and deep learning methods," AgriEngineering, vol. 3, no. 3, pp. 542–558, 2021,
doi:10.3390/agriengineering3030035.
[21] V. Singh and A. K. Misra, "Detection of plant leaf diseases using image segmentation and soft computing
techniques," Information Processing in Agriculture, vol. 4, no. 1, pp. 41–49, 2017.
[22] D. Hendrycks and T. Dietterich, "Benchmarking neural network robustness to common corruptions and
perturbations," in Proc. ICLR, 2019.
[23] A. Torralba and A. A. Efros, "Unbiased look at dataset bias," in Proc. IEEE CVPR, 2011, pp. 1521–1528.
[24] I. Gulrajani and D. Lopez-Paz, "In search of lost domain generalization," in Proc. ICLR, 2021.
[25] P. W. Koh et al., "WILDS: A benchmark of in-the-wild distribution shifts," in Proc. ICML, PMLR 139, 2021,
pp. 5637–5664.
[26] G. Buchsbaum, "A spatial processor model for object colour perception," J. Franklin Institute, vol. 310, no. 1,
pp. 1–26, 1980.
[27] K. Zuiderveld, "Contrast limited adaptive histogram equalization," in Graphics Gems IV, Academic Press, 1994,
pp. 474–485.
[28] C. Shorten and T. M. Khoshgoftaar, "A survey on image data augmentation for deep learning," J. Big Data, vol. 6,
art. 60, 2019.
[29] R. Geirhos et al., "Shortcut learning in deep neural networks," Nature Machine Intelligence, vol. 2,
pp. 665–673, 2020.
[30] J. R. Zech et al., "Variable generalization performance of a deep learning model to detect pneumonia in chest
radiographs: A cross-sectional study," PLoS Medicine, vol. 15, no. 11, e1002683, 2018.
[31] A. J. DeGrave, J. D. Janizek, and S.-I. Lee, "AI for radiographic COVID-19 detection selects shortcuts over
signal," Nature Machine Intelligence, vol. 3, pp. 610–619, 2021, doi:10.1038/s42256-021-00338-7.
[32] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," Patterns,
vol. 4, no. 9, art. 100804, 2023, doi:10.1016/j.patter.2023.100804.
[33] B. Barz and J. Denzler, "Do we train on test data? Purging CIFAR of near-duplicates," J. Imaging, vol. 6, no. 6,
art. 41, 2020, doi:10.3390/jimaging6060041.
[34] D. R. Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic
structure," Ecography, vol. 40, no. 8, pp. 913–929, 2017, doi:10.1111/ecog.02881.
[35] C. G. Northcutt, L. Jiang, and I. L. Chuang, "Confident learning: Estimating uncertainty in dataset labels,"
J. Artificial Intelligence Research, vol. 70, pp. 1373–1411, 2021.
[36] C. Strobl, A.-L. Boulesteix, A. Zeileis, and T. Hothorn, "Bias in random forest variable importance measures:
Illustrations, sources and a solution," BMC Bioinformatics, vol. 8, art. 25, 2007.
[37] G. Hooker, L. Mentch, and S. Zhou, "Unrestricted permutation forces extrapolation: Variable importance requires
at least one more model, or there is no free variable importance," Statistics and Computing, vol. 31, art. 82, 2021,
doi:10.1007/s11222-021-10057-z.
[38] A. Fisher, C. Rudin, and F. Dominici, "All models are wrong, but many are useful: Learning a variable's importance
by studying an entire class of prediction models simultaneously," J. Machine Learning Research, vol. 20, no. 177,
pp. 1–81, 2019.
[39] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any
classifier," in Proc. ACM SIGKDD, 2016, pp. 1135–1144.
[40] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Proc. NeurIPS, 2017.
[41] S. Krishna et al., "The disagreement problem in explainable machine learning: A practitioner's perspective,"
arXiv:2202.01602, 2022 (TMLR, 2024).
[42] C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable
models instead," Nature Machine Intelligence, vol. 1, pp. 206–215, 2019.
[43] Y. Toda and F. Okura, "How convolutional neural networks diagnose plant disease," Plant Phenomics, vol. 2019,
art. 9237136, 2019, doi:10.34133/2019/9237136.
[44] B. Kim et al., "Interpretability beyond feature attribution: Quantitative testing with concept activation vectors
(TCAV)," in Proc. ICML, PMLR 80, 2018, pp. 2668–2677.
[45] P. W. Koh et al., "Concept bottleneck models," in Proc. ICML, PMLR 119, 2020, pp. 5338–5348.
[46] A. Radford et al., "Learning transferable visual models from natural language supervision," in Proc. ICML,
PMLR 139, 2021, pp. 8748–8763.
[47] T. Oikarinen, S. Das, L. M. Nguyen, and T.-W. Weng, "Label-free concept bottleneck models," in Proc. ICLR, 2023.
[48] Y. Yang, A. Panagopoulou, S. Zhou, D. Jin, C. Callison-Burch, and M. Yatskar, "Language in a bottle: Language
model guided concept bottlenecks for interpretable image classification," in Proc. IEEE/CVF CVPR, 2023,
pp. 19187–19197.
[49] G. Ilharco et al., "OpenCLIP," Zenodo, 2021, doi:10.5281/zenodo.5143773.
[50] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE CVPR, 2016,
pp. 770–778.
[51] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Machine Learning Research, vol. 12,
pp. 2825–2830, 2011.
[52] G. Csurka, C. R. Dance, L. Fan, J. Willamowski, and C. Bray, "Visual categorization with bags of keypoints,"
in Proc. ECCV Workshop on Statistical Learning in Computer Vision, 2004.
[53] M. Varma and A. Zisserman, "A statistical approach to texture classification from single images," Int. J.
Computer Vision, vol. 62, pp. 61–81, 2005.
[54] D. Arthur and S. Vassilvitskii, "k-means++: The advantages of careful seeding," in Proc. ACM-SIAM SODA, 2007,
pp. 1027–1035.
[55] D. Sculley, "Web-scale k-means clustering," in Proc. WWW, 2010, pp. 1177–1178.
