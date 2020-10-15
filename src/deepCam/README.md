### TODO
- [x] Set up W&B
- [X] Run tests on different splits
- [X] Get visualisation script as function that can be easily imported and ran from other scripts
- [X] Run splits on ALLHIST (e.g. 60000 samples)
- [X] Get visualisations

## Call 16th of October
#Here's some things we should/could get done:
- [x] Investate preempt queue
- [ ] Implement multi-node training
- [X] Visualization for 4k 80p
- [X] Fix colorbar
- [ ] Create confusion matrices
- [x] Model checkpointing (including the optimizer and the LR scheduler)

#Experiments:
- [ ] Train e.g. on 64k and test on blocks with different time displacement from training data (graph time displacement vs. IoU)
- [ ] Train on all of Allhist and predict on HAPPI15
- [ ] Dial down fp loss factor especially for cyclones (e.g. make a graph that shows IoU, fp rate, fn rate vs. fp loss factor
- [ ] Add channels, e.g. pressure (vii) and/or temperature (viii, ix) (for tropical cyclones)
- [ ] Sensitivity analysis for channels

#Other ideas:
- [ ] Add a feature somewhat deep in the model with the time of the year
