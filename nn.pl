/**
 * Simple neural network.
 */
:- use_module(library(clpfd)).
:- op(500, xfy, =+>).
:- op(500, xfy, =->).
:- op(500, xfy, =*>).

% Operators for matrix arithmetics
(X, Y) =+> P :- matAdd(X, Y, P).
(X, Y) =-> P :- matSub(X, Y, P).
(X, Y) =*> P :- matMul(X, Y, P).

inputs([[0.0, 1.0, 0.0],
          [0.0, 1.0, 1.0],
          [0.0, 0.0, 1.0],
          [1.0, 0.0, 0.0],
          [1.0, 0.0, 1.0],
          [1.0, 1.0, 1.0],
          [0.0, 0.0, 0.0],
          [1.0, 1.0, 1.0]]).

outputs([[0.0], 
         [0.0], 
         [0.0], 
         [1.0], 
         [1.0], 
         [1.0],
         [0.0],
         [1.0]]).

weights([[0.5], [0.5], [0.5]]).

test([[0.0, 1.0, 0.0]]).

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
activation(_, _, [], []) :- !.
activation(Fun, Deriv, [XX|XXS], [YY|YYS]) :- call(activelem, Fun, Deriv, XX, YY), 
                                              activation(Fun, Deriv, XXS, YYS).

activelem(_, _, [], []) :- !.
activelem(Fun, Deriv, [X|XS], [Y|YS]) :- call(Fun, X, Deriv, Y), 
                                         activelem(Fun, Deriv, XS, YS).

/**
 * Forward function
 * @param Fun activation function
 * @param Data input data
 * @param Weights weight matrix
 * @param F forward step
 */
forward(Fun, Data, Weights, Frwd) :- dot(Data, Weights, DotD),
                                     activation(Fun, false, DotD, Frwd).

/**
 * Backward function
 * @param Fun activation function
 * @param Frwd data from forward function
 * @param Out output data
 * @param Inp input data
 * @param Weights trained weights
 * @param W adjusted weights 
 */
backward(Fun, Frwd, Out, Inp, Weights, W) :- (Out, Frwd) =-> Err,  % Get error
                                             activation(Fun, true, Frwd, AFrwd),
                                             (Err, AFrwd) =*> Delta,
                                             transpose(Inp, TInp),
                                             dot(TInp, Delta, DDelta),
                                             (Weights, DDelta) =+> W, !.

/**
 * Train function
 * @param Fun activation function
 * @param Inp input data
 * @param Out expected output data
 * @param Weights weights matrix
 * @param W trained weights
 */
train(Fun, Inp, Out, Weights, W) :- forward(Fun, Inp, Weights, Frwd),
                                    backward(Fun, Frwd, Out, Inp, Weights, W).

/**
 * Training function
 * Trains NNet in passed in amount of epochs
 * @param Fun activation function
 * @param Inp input data for training
 * @param Out output data for training
 * @param Weights current weights
 * @param Epochs amount of epochs to train on
 * @param W returned newly trained weights
 */

training(Fun, Inp, Out, Weights, Epochs, [W|TW]) :- Epochs > 0,
                                                    E is Epochs-1,
                                                    train(Fun, Inp, Out, Weights, W),
                                                    training(Fun, Inp, Out, W, E, TW).
training(_, _, _, _, _, _) :- !.
/**
 * Prediction function
 * @param Fun activation function that should be used
 * @param Data input data
 * @param Weights trained weights
 * @param Pred made prediction
 */
predict(Fun, Data, Weights, Pred) :- dot(Data, Weights, DotD),
                                     activation(Fun, false, DotD, Pred).

run(Inp, E) :- inputs(I),
               outputs(O),
               weights(W),
               training(sigmoid, I, O, W, E, NW),
               length(NW, L),
               L1 is L-1,
               nth0(L1, NW, NWl), !,
               predict(sigmoid, Inp, NWl, P),
               nth0(0, Inp, Expcl),
               nth0(0, Expcl, Expc), 
               write("Expected = ["), write(Expc), write("]"), nl,
               write("Predicted = "), write(P), nl, nl.

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

/**
 * Dot product of 2 matrixes
 * @param [XX|XXS] matrix MxN
 * @param YYS matrix NxM
 * @param V matrix of results
 */
dot([], _, [[]]).
dot(XXS, YYS, V) :- dotAlign(YYS, 0, TYYS),  % Rotate array columns
                    dotInner(XXS, TYYS, V).

dotInner([XX], YYS, [V]) :- !, dotInner2(XX, YYS, V).
dotInner([XX|XXS], YYS, [D|T]) :- dotInner2(XX, YYS, D), 
                                  dotInner(XXS, YYS, T).

dotInner2(_, [], []) :- !.
dotInner2(XX, [YY|YYS], [D|T]) :- dot1D(XX, YY, D), 
                                  dotInner2(XX, YYS, T).

dotAlign(X, N, [Y|T]) :- dotAlignM(X, N, Y),  % Call nth element extractions for all possible ns
                         N2 is N+1,
                         !,  % Cut on boundry check fail
                         nth0(0, X, X0),
                         length(X0, L),
                         N2 =< L,
                         dotAlign(X, N2, T).
dotAlign(_,_,[]).  % To succeed after N2 > L

% Extracts all nth element from passed in arrays
dotAlignM([], _, []).
dotAlignM([XS|XXS], N, [X|T]) :- nth0(N, XS, X), 
                                 dotAlignM(XXS, N, T).

/**
 * Applies operation to 2 vectors 
 * (nth element in X with nth element in Y)
 */
vectOp(Op, X, Y, Z) :- maplist(Op, X, Y, Z).

/**
 * Adds 2 matrixes
 * (nth element in X to nth element in Y)
 */
matAdd([], [], [[]]).
matAdd([XX], [YY], [ZZ]) :- !, vectOp(plusf, XX, YY, ZZ).
matAdd([XX|XXS], [YY|YYS], [ZZ|ZZS]) :- vectOp(plusf, XX, YY, ZZ),
                                        matAdd(XXS, YYS, ZZS).

plusf(X, Y, Z) :- Z is X + Y.  % plus from std lib doesn't work with floats

/**
 * Subtracts 2 matrixes
 * (nth element in X to nth element in Y)
 */
matSub([], [], [[]]).
matSub([XX], [YY], [ZZ]) :- !, vectOp(minus, XX, YY, ZZ).
matSub([XX|XXS], [YY|YYS], [ZZ|ZZS]) :- vectOp(minus, XX, YY, ZZ),
                                        matSub(XXS, YYS, ZZS).

minus(X, Y, Z) :- Z is X - Y.

/**
 * Multipliers 2 matrixes
 * (nth element in X to nth element in Y)
 */
matMul([], [], [[]]).
matMul([XX], [YY], [ZZ]) :- !, vectOp(mul, XX, YY, ZZ).
matMul([XX|XXS], [YY|YYS], [ZZ|ZZS]) :- vectOp(mul, XX, YY, ZZ),
                                        matMul(XXS, YYS, ZZS).

mul(X, Y, Z) :- Z is X * Y.
