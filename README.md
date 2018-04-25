# Solving Schrodinger's Equation for the Double Slit Experiment
This was created as the final project for the class CS 555: Numerical Partial Differential Equations
at the University of Illinois at Urbana-Champaign.


## The Equation

The problem consists of solving the following equation:
$$ i \hbar \frac{\partial}{\partial t} \phi(\mathbb{r}, t) = [\frac{-\hbar^2}{2m} \Del + V(\mathbb{r, t}) ]\phi(\mathbb{r}, t)$$


IThe domain of the problem is:

$$ x \in [-1, 1]^2$$


## Testing the Method

To test that the method performs as expected we first test the error and convergence of the problem of a particle in a box. This corresponds to the potential, $$V(\mathbb{r},t) $$ being 0 within the domain and infinity outside of the domain. This reduces the problem to one with homogenous Dirichlet boundary conditions.

The solution to this is

$$ \cos(\frac{x\pi}{2})\cos(\frac{y\pi}{2})e^{it\frac{\hbar \pi^2}{8m}}$$


## Double Slit Experiment

Gmesh was used to generate the mesh of the domain. The problem consists of a plane wave heading in the direction of two closely spaced slits. To test that the solution is correct we will measure the interference pattern at the right hand side is expected.

