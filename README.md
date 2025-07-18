
# Costs in prediction modelling for health


## Paper details


## Summary of notebooks


### Notebook 1: 

This notebook:
- Generates some simulated data (example labels and score predictions) by generating scores from two normal distributions, one for each class.
- Generates a calibrated version of the scores, using the PDFs of the normal distributions used to generate them.
- Plots ROC curve for the generated ranking.
- Plots cost lines for each point on the ROC curve, showing how this produces the lower envelope (the standard definition of a cost curve)
- Plots lower envelope cost curve from ROC curve, without using the predicted scores.
- Plots Brier curve and cost curve together, showing the decomposition of Brier score into refinement and calibration loss.
- Show that the Brier curve for calibrated scores equals the lower envelope cost curve.
- Plots decision curve for this ROC curve
- Show that 'upper envelope' decision curve can be plotted to show the best net benefit possible for this ROC curve, which would be seen if the
scores were calibrated.
- Show that calibrated probabilities do result in the 'upper envelope' decision curve.
- Plot Brier curves including the 'treat all' and 'treat none' lines, which we refer to as 'predict all as positive' and 'predict all as negative'.


### Notebook 2: Cost curve illustration

This notebook uses a toy example model output with 9 examples, to demonstrate cost curves and Brier curves, including:
- How cost lines in cost space each correspond to a particular point in ROC space (point-line duality).
- The typical definition of a cost curve as the lower envelope of all cost lines for a given ROC curve (independent of model scores).
- Brier curves as the cost curve when using the probabilistic threshold choice method.


### Notebook 3: Isometrics in ROC space

This notebook shows isometrics for net benefit and Brier curves in ROC space, demonstrating:
- How the cost affects the gradient of these isometrics and which points in ROC space have equal net benefit or loss.
- The gradient of these isometrics is the same for net benefit and loss, for a given cost proportion, such that these measures will
always choose the same point on the ROC curve as the 'best' (for a given cost proportion).
- When the cost prooprtion is high the net benefit isometrics become much closer than for loss, showing the big change in the range of possible 
net benefit values across cost proportions. In contrast, loss (for Brier curves) does not change much across cost proportions.


### Notebook 4: 

This notebook shows:

- What happens for decision curves and Brier curves when the classes are swapped, i.e. when 
a positive example becomes a negative example, and vice versa, and the scores are flipped so that a score of A becomes 1-A.

- Rescaling scores for decision curves so that the cost value of a positive and negative misclassification sums to 2 (like for Brier curves).



