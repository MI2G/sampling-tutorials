# sampling-tutorials : The Bayesian Approach to Inverse Problems in Imaging Science

In these tutorials, you will learn about Bayesian computation for inverse problems in imaging science. We set up an image deconvolution problem and solve the inverse problem by using different sampling algorithms. Because we obtain samples from the posterior distribution we are able to do uncertainty quantification and other advance inferences. Currently, there is a Python notebook using a vanilla [Langevin sampling algorithm](https://hal.science/hal-01267115/document) (MYULA_pytorch.ipynb) and an accelerated algorithm [SK-ROCK](https://pure.hw.ac.uk/ws/portalfiles/portal/41830170/19m1283719.pdf) (using an explicit stabilized method, SKROCK_pytorch.ipynb). We showcase a deblurring problem using a Total Variation (TV) prior.

### Authors
* Dobson, Paul [pdobson@ed.ac.uk](pdobson@ed.ac.uk)
* Kemajou, Mbakam Charlesquin [cmk2000@hw.ac.uk](cmk2000@hw.ac.uk)
* Klatzer, Teresa [t.klatzer@sms.ed.ac.uk](t.klatzer@sms.ed.ac.uk)
* Melidonis, Savvas [sm2041@hw.ac.uk](sm2041@hw.ac.uk)

### Funding

We acknowledge funding from projects [BOLT](https://www.macs.hw.ac.uk/~mp71/bolt.html), [BLOOM](https://www.macs.hw.ac.uk/~mp71/bloom.html) and [LEXCI](https://www.macs.hw.ac.uk/~mp71/lexci.html): This work was supported by the UK Research and Innovation (UKRI) Engineering and Physical Sciences Research Council (EPSRC) grants EP/V006134/1 , EP/V006177/1 and EP/T007346/1, EP/W007673/1 and EP/W007681/1.
