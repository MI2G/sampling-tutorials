# sampling-tutorials : The Bayesian Approach to Inverse Problems in Imaging Science

In these tutorials, you will learn about Bayesian computation for inverse problems in imaging science. We set up an image deconvolution problem and solve the inverse problem by using different sampling algorithms. Because we obtain samples from the posterior distribution we are able to do uncertainty quantification and other advance inferences. There is a Python notebook using a vanilla [Langevin sampling algorithm](https://hal.science/hal-01267115/document) (MYULA_pytorch.ipynb) and an accelerated algorithm [SK-ROCK](https://pure.hw.ac.uk/ws/portalfiles/portal/41830170/19m1283719.pdf) (using an explicit stabilized method, SKROCK_pytorch.ipynb). We showcase a deblurring problem using a Total Variation (TV) prior. Another tutorial features a plug-and-play ULA algorithm [PnP-ULA](https://epubs.siam.org/doi/abs/10.1137/21M1406349?journalCode=sjisbi) using a neural network by [Ryu et al](https://github.com/uclaopt/Provable_Plug_and_Play/). The latest tutorial on Empirical Bayes Estimation enables you to understand how regularization parameters can be estimated in a Bayesian way without requiring ground truth. It's an implementation of the SAPG algorithm originally proposed by [Vidal et al](https://epubs.siam.org/doi/pdf/10.1137/20M1339829).

### Authors
* Dobson, Paul [pdobson@ed.ac.uk](pdobson@ed.ac.uk)
* Kemajou, Mbakam Charlesquin [cmk2000@hw.ac.uk](cmk2000@hw.ac.uk)
* Klatzer, Teresa [t.klatzer@sms.ed.ac.uk](t.klatzer@sms.ed.ac.uk)
* Melidonis, Savvas [sm2041@hw.ac.uk](sm2041@hw.ac.uk)

### Funding

We acknowledge funding from projects [BOLT](https://www.macs.hw.ac.uk/~mp71/bolt.html), [BLOOM](https://www.macs.hw.ac.uk/~mp71/bloom.html) and [LEXCI](https://www.macs.hw.ac.uk/~mp71/lexci.html): This work was supported by the UK Research and Innovation (UKRI) Engineering and Physical Sciences Research Council (EPSRC) grants EP/V006134/1 , EP/V006177/1 and EP/T007346/1, EP/W007673/1 and EP/W007681/1.

### Optional: Install sampling_tools as a package

We provide code so that you can install the `sampling_tools` module as a package, and use it in your own code.
`sampling_tools` can be easily installed with `pip`. After cloning the repository, run the following commands:

```bash
$ cd sampling_tutorials
$ pip install .
```

This is optional, you can run the tutorial notebooks without installing `sampling_tools`.
The package can then be imported in Python as `import sampling_tools as st`. 
