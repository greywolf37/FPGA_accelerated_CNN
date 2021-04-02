# Deliverables

Your final report, along with the code submission, presentation and reported FOM, will help us grade your final project.

## Deadlines

* Presentation in class on 4/28 (tentative)
* Final report due on 5/11 (tentative) at 11:59 PM
* Source code on GitHub Classroom is due by 5/11 (tentative) 11:59 PM

## Presentation

Your final presentation will consist of a synchronous presentation in class on 4/28 (tentative) with your webcam on. The slide deck should contain the following slides:

* Title slide
* Baseline Design (Software)
* Baseline Design (Hardware)
* Design Space Exploration (Optimizations)
* Preliminary Results

The *baseline design* should relate back to the non-optimized version specified in the milestones. The *design space exploration and optimizations* refers to your optimization flow and the specific optimizations you carried out. The *preliminary results* should contain a CPU baseline and results from your current, preliminary implementation.

Each presentation should take no longer than five minutes. We will have to cut you off if you go over five minutes. 

### Presentation Order

TBD closer to the date.

### Template

You can find the presentation template on Canvas in the *Final Project* module.

### Rubric

Your presentation grade will be decided by rubric with the following equally weighted categories:

* Poise
* Length of presentation
* Organization
* Visuals
* Data Presentation
* Mechanics

## Report

Your final report should contain no more than 10 pages, excluding references. The report should have the following sections:

* Abstract
* Introduction
* Methodology
* Evaluation
* Conclusion
* Appendix (no more than 4 pages)

You must use the [ACM Master Article Template](https://www.acm.org/publications/proceedings-template). We suggest using [Overleaf](https://www.overleaf.com/).

## Code

Your code should be committed and pushed to github. In order to validate your results, we will run the following commands with `benfranklin` being replaced with your name:

```bash
git clone https://github.com/penn-ese680-002/project-benfranklin.git 
cd project-benfranklin
aws_ppi
run_script benchmark.ppi # this is ran in aws_ppi
```

**Make sure that you can succeed at running the listed four commands. We will be unable to verify your FOM if the command fails.**

`benchmark.ppi` should be an aws_ppi command script that runs your benchmark script and reports the runtimes, accuracies and FOM of your design. `benchmark.ppi` should be stored in the root directory of your repository. An example aws_ppi command script is shown below. More details can be found [here](https://cmd2.readthedocs.io/en/latest/features/scripting.html#command-scripts).


```bash
launch fpga_build
start_job ...
```
