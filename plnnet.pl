/**
 * Program showcasing nn module
 * @author Marek Sedláček (xsedla1b)
 * @email xsedla1b@fit.vutbr.cz
 * @date April 2021
 */ 

% Starts program by calling main function
:- initialization(main). 

:- use_module(input2).
:- use_module(nn).

/**
 * Runs tests with expected value printed (for default project input files)
 * @param E number of epochs for training (does not saves the weights)
 */
runTests(E) :- getInputs(I),
               getOutputs(O),
               getWeights(W),
               training(sigmoid, I, O, W, E, NW),
               length(NW, L),
               L1 is L-1,
               nth0(L1, NW, NWl), !,
               loadData('test', TestData),
               runSubTest(NWl, TestData).

% Subfunction for runTests
runSubTest(_, []) :- !.
runSubTest(Weights, [X|XS]) :- predict(sigmoid, [X], Weights, P),
                               nth0(0, X, Expcl),
                               write(X), write(': '),
                               write('Expected = ['), write(Expcl), write('] - '),
                               write('Predicted = '), write(P), nl,
                               runSubTest(Weights, XS).
                

/**
 * Trains NN and runs prediction
 * @param Inp input data for prediction
 * @param E number of epochs for training
 * @note does not save trained weights
 */
run(Inp, E) :- getInputs(I),
               getOutputs(O),
               getWeights(W),
               training(sigmoid, I, O, W, E, NW),
               length(NW, L),
               L1 is L-1,
               nth0(L1, NW, NWl), !,
               predict(sigmoid, Inp, NWl, P),
               write('Predicted = '), write(P), nl, nl.

/**
 * Main function. 
 * Interactive program for training and predictions
 */
main :- loadTrainData('train'), 
        loadOutputData('outputs'),
        loadWeights('weights'),
        write('Input number of epochs for training: '), nl,
        loadInput(Epochs),
        write('Training...'), nl,
        getInputs(I),
        getOutputs(O),
        getWeights(W),
        training(sigmoid, I, O, W, Epochs, NW),
        length(NW, L),
        L1 is L-1,
        nth0(L1, NW, NWl), !,
        asserta(trainedWeights(NWl)),
        write('Input (eg.: 1.0 0.0 0.0): '), nl,
        loadInput(Inp), !,
        predict(sigmoid, [Inp], NWl, P),
        write('Predicted = '), write(P), nl, nl.
main :- write('ERROR occurred!'), 
        halt(1).
