import numpy as np


def training_sample_distribution(sample_size):
    sample = np.random.uniform(size=sample_size)  # Initialize sample
    isNormal = np.random.uniform() < 0.50

    if isNormal:
        sample = np.random.normal(loc=np.random.uniform(-20, 20),
                                  scale=np.random.uniform(0.5, 10), size=sample_size)
    if not isNormal:
        np.random.normal(size=sample_size)
        """
        We will examine the following distribution types:
        1) Student-t distribution/Cauchy distribution
        2) Exponential/Erlang distribution
        3) Uniform distribution
        4) Bimodal distributions
        5) Log-normal distributions
        6) Mixed variance normal distributions
        """
        distribution_type = np.random.choice([1, 2, 3, 4, 5, 6])
        if distribution_type == 1:  # Student-t distribution with df ranging from 1 (Cauchy) to 15
            df = np.round(np.random.uniform(1, 15))
            sample = np.random.standard_t(df=df, size=sample_size)

            sample = np.random.standard_t(df=df, size=sample_size)
        elif distribution_type == 2:  # Erlang's distribution with df ranging from 1 (Exponential), to 7
            orientation = np.random.choice([-1, 1])
            shape = np.round(np.random.uniform(1, 7))
            rate = np.random.uniform(0.5, 4)
            sample = orientation * np.random.gamma(shape=shape, scale=1.0 / rate, size=sample_size)
        elif distribution_type == 3:  # Uniform distribution
            span = np.random.uniform(1, 20)
            a = np.random.uniform(-15, 15)
            b = a + span
            sample = np.random.uniform(a, b, size=sample_size)
        elif distribution_type == 4:  # Multimodal distribution
            mu1 = np.random.uniform(-6, -2)
            mu2 = np.random.uniform(2, 6)
            sigma1 = np.random.uniform(0.25, 2)
            sigma2 = np.random.uniform(0.25, 2)
            p = np.random.uniform(0.35, 0.65)
            comp = np.random.binomial(1, p, size=sample_size)
            x1 = np.random.normal(mu1, sigma1, size=sample_size)
            x2 = np.random.normal(mu2, sigma2, size=sample_size)
            sample = np.where(comp, x1, x2)
        elif distribution_type == 5:  # Log-normal distribution
            orientation = np.random.choice([-1, 1])
            sigma = np.random.uniform(0.3, 1.25)
            sample = orientation * np.random.lognormal(mean=0, sigma=sigma, size=sample_size)
        else:
            sigma1 = np.random.uniform(0.5, 1.5)
            sigma2 = np.random.uniform(3, 7)
            p = np.random.uniform(0.10, 0.3)
            comp = np.random.binomial(1, p, size=sample_size)
            x1 = np.random.normal(0, sigma1, size=sample_size)
            x2 = np.random.normal(0, sigma2, size=sample_size)
            sample = np.where(comp, x2, x1)

    return sample, isNormal


def validation_sample_distribution(sample_size):
    sample = np.random.uniform(size=sample_size)  # Initialize sample
    isNormal = np.random.uniform() < 0.5

    if isNormal:
        sample = np.random.normal(loc=np.random.uniform(-20, 20),
                                  scale=np.random.uniform(0.5, 10), size=sample_size)
    if not isNormal:

        np.random.normal(size=sample_size)
        """
        We will examine the following distribution types:
        1) Student-t distribution/Cauchy distribution
        2) Exponential/Erlang distribution
        3) Uniform distribution
        4) Bimodal distributions
        5) Laplace distributions (UNSEEN)
        6) Beta distributions (UNSEEN)
        7) Mixed variance normal distributions
        """
        distribution_type = np.random.choice([1, 2, 3, 4, 5, 6, 7])
        if distribution_type == 1:  # Student-t distribution with df ranging from 1 (Cauchy) to 15
            df = np.round(np.random.uniform(1, 15))
            sample = np.random.standard_t(df=df, size=sample_size)
        elif distribution_type == 2:  # Erlang's distribution with df ranging from 1 (Exponential), to 7
            orientation = np.random.choice([-1, 1])
            shape = np.round(np.random.uniform(1, 7))
            rate = np.random.uniform(0.5, 4)
            sample = orientation * np.random.gamma(shape=shape, scale=1.0 / rate, size=sample_size)
        elif distribution_type == 3:  # Uniform distribution
            span = np.random.uniform(1, 20)
            a = np.random.uniform(-15, 15)
            b = a + span
            sample = np.random.uniform(a, b, size=sample_size)
        elif distribution_type == 4:  # Multimodal distribution
            mu1 = np.random.uniform(-6, -2)
            mu2 = np.random.uniform(2, 6)
            sigma1 = np.random.uniform(0.25, 2)
            sigma2 = np.random.uniform(0.25, 2)
            p = np.random.uniform(0.35, 0.65)
            comp = np.random.binomial(1, p, size=sample_size)
            x1 = np.random.normal(mu1, sigma1, size=sample_size)
            x2 = np.random.normal(mu2, sigma2, size=sample_size)
            sample = np.where(comp, x1, x2)
        elif distribution_type == 5:
            scale = np.random.uniform(1, 5)
            sample = np.random.laplace(scale=scale, size=sample_size)
        elif distribution_type == 6:
            a = np.random.choice([0.5, 1.0, 1.5])
            b = np.random.choice([0.5, 1.0, 1.5])
            sample = np.random.beta(a=a, b=b, size=sample_size)
        else:
            sigma1 = np.random.uniform(0.5, 2)  # Slightly different from the training set...
            sigma2 = np.random.uniform(5, 10)
            p = np.random.uniform(0.10, 0.3)
            comp = np.random.binomial(1, p, size=sample_size)
            x1 = np.random.normal(0, sigma1, size=sample_size)
            x2 = np.random.normal(0, sigma2, size=sample_size)
            sample = np.where(comp, x2, x1)

    return sample, isNormal
