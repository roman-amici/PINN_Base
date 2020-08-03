# PINN_Base

A framework for building custom Physics Informed Neural Networks [PINNs](https://github.com/maziarraissi/PINNs).

This was built in the course of my own research on PINNs as an alternative to copying and pasting code for each new idea I had. As a result, the base template is built for extending PINNs rather than as a interface for solving particular differential equations.

Currently, it supports Tensorflow 1.15. It was my original intention to have maintain feature parity for a Tensorflow 2.0 version as well however it is very far behind the 1.0 version and may be discontinued in the future.

## Installation

To install as a python module, from the git root-directory call

```shell
pip install .
```

Requirements include Tensorflow 1.15, as well as the usual suspects from the scipy stack.

### Why TF 1.15?

PINNs remain difficult to train with standard first-order methods. The best option remains L-BFGS. The world has moved on to TF 2.0. In that transition, they unfortunately left some things behind, most notably the ScipyOptimizerInterface which was the only native support for L-BFGS that tensorflow had. There are ways to use L-BFGS in TF 2.0 (as well as pytorch), however, none of them are particularly elegant making migration a less than compelling prospect.

## Extension Philosophy

As it exists, PINN_Base implements a Multi-Layer Perceptron for regression problems. In order to do anything interesting, like solve a differential equation, you will have to extend it by inheriting from PINN_Base. The conventional wisdom when writing a library is to "prefer composition over inheritance". So this design could be criticized on those grounds. The rational for favoring inheritance over composition was to avoid locking myself down to a single interface for each of the components.

For instance, Keras implements a composable library with separate components for the forward-pass, the loss function, and the optimizer. As a result the loss function has a very specific interface which takes only Y and Y_hat. This makes it cumbersome (though not impossible) to implement a PINN using Keras since one also needs the input, X, in order to compute grad(Y,X). Inheritance, makes it easier to override portions of the interface as needed while still reusing code. This is especially important in a research context where it is intentionally left unknown what the typical use case will end up being.

## Usage

See examples.
TODO: Add Examples
