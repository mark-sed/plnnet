/**
 * Simple neural network.
 */
:- use_module(library(clpfd)).

/**
 * Sigmoid function.
 * @param X1 value for which sigmoid will be counted
 * @param true for derivation of sigmoid
 *        false for sigmoid
 * @param Y output value
 */
sigmoid(X, true, Y) :- Y is X * (1 - X).
sigmoid(X, false, Y) :- Y is 1/(1 + (exp(-X))).

/**
 * Activtion function for whole matrix.
 * @param Fun activation function
 * @param Deriv if it should be normal function or derivation
 * @param [XX|XXS] input matrix
 * @param [YY|YYS] output matrix
 */
activation(_, _, [], []).
activation(Fun, Deriv, [XX|XXS], [YY|YYS]) :- call(activelem, Fun, Deriv, XX, YY), 
                                              activation(Fun, Deriv, XXS, YYS).

activelem(_, _, [], []).
activelem(Fun, Deriv, [X|XS], [Y|YS]) :- call(Fun, X, Deriv, Y), 
                                         activelem(Fun, Deriv, XS, YS).

/**
 * Dot product of 2 lists.
 * @param [X|XS] first list
 * @param [Y|YS] second list
 * @param V result of dot product
 */
dot1D([], [], 0).
dot1D([X|XS], [Y|YS], V) :- Prod is X * Y, 
                            dot1D(XS, YS, R),
                            V is Prod + R.