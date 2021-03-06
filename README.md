# text2image-tensorflow
Image generation from text descriptions using Deep Convolutional Generative Adversarial Networks (DC-GANs).

We modify the baseline DC-GAN implementation for our experiments. All our code uses **Python 3** and **TensorFlow**.
We also needed **scikit-learn**, **NumPy** and **SciPy**.
- Baseline DC-GAN code is from - https://github.com/carpedm20/DCGAN-tensorflow 
- Skip-thought-vector Code is from - https://github.com/tensorflow/models/tree/master/research/im2txt
- Pycocotools  is from - https://github.com/cocodataset/cocoapi/tree/master/PythonAPI

We use a modified version of the DC-GAN given in [1]. We add conditioning to the DC-GAN in the form of a sentence embedding of the input text description. In our experiments, we used skip-thought-vectors [2] to generate these sentence embeddings. We also used a class-loss term to modify the loss of the discriminator which penalizes the discriminator if it is unable to identify the sample to be fake given an image from the true data but paired with a text description that does not match that image. This helps the discriminator use the conditioning information properly and check if the generated images are aligned with the text description or not. This model is called the DC-GAN-CLS [3].

The architecture we used.
![Alt Text](assets/ganfinal.png)

We also tried various training schedules for the GAN.
* Update both the generator and discriminator once every iteration. 
* Update the generator twice and discriminator once every iteration.
* Same as above but when discriminator loss approached 0, stop updating the discriminator and only update the generator till situation improves.

Different training schedules did not offer better visuals, however, quantitative evaluation (inception score) proved the third method stated above to be the best.

Some Results for the different models we tried on the CUB-2011 and MS-COCO 2014 datasets.
![Alt Text](assets/ganresult.png)

Some tips
- Add the conditioning vector to the DC-GAN at an intermediate layer by concatenating along channels.
- PCA on the conditioning vector helps keep computational/memory needs low without sacrificing quality. Adding FC layer to reduce dimensionality doesn't work well, leads to difficulty in training.
- If losses stagnate within the first epoch, something is probably wrong.
- Even if losses look good, it may take around 10-20 epochs (training set size of 10K images) before the generated images start to show rudimentary image-like features.


## References

1. A. Radford, L. Metz, and S. Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.

2. R. Kiros, Y. Zhu, R. R. Salakhutdinov, R. Zemel, R. Urtasun, A. Torralba, and S. Fidler. Skip-thought vectors. In Advances in neural information processing systems, pages 3294–3302, 2015.

3. S. Reed, Z. Akata, X. Yan, L. Logeswaran, B. Schiele, and H. Lee. Generative adversarial text to image synthesis. arXiv preprint arXiv:1605.05396, 2016.
