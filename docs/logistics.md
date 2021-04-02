# Logistics

The final project will consist of the following gradable items:

* Figure of Merit (FOM) (40 points)
    * Design will be ranked based on FOM
    * Grade will be curved based on ranking
* Milestone Completion (20 points)
* Report (20 points)
* Presentation (20 points)
* Credit Usage 

## Figure of Merit (40 points)

The figure of merit is defined as follows 

$$
FOM := Runtime
$$

$Runtime$ is the number of seconds to complete 16 inferences at batch size = {1, 16}

### Resource Constraints

*Your design must use less than 1024 DSPs.*

## Milestones (10 points)

In order to award partial credit, your design will consist of the following milestones:

* Hardware kernel completion (without optimization, software emulation only) (4pts)
* Setup C++ extension with hardware kernel and software emulation (4pts)
* Run your Python benchmarking script in software emulation (4pts)
* Baseline design (non-optimized version) verified in software and hardware (4pts)
* Software optimizations (4pts)
    - Use all three compute units and/or out of order command queue (non-exhaustive list)

*Your report should prove that you have achieved these milestones.*

## [Report (20 points)](deliverables.md)

## [Presentation (20 points)](deliverables.md)

## Credit Usage

Each group will be given a credit limit of **300 USD** for the project. The project will **automatically terminate** once you've exhausted all credits. Credit usage (sum of both partners) starts 3/24. The top 5 groups with minimal credit usage will receive 5, 4, 3, 2, 1 extra credit points. Feel free to utilize the AWS Anomaly Detecetion outlined in Piazza post [@146](https://piazza.com/class/kjui9qnwfkz9?cid=146) to help better monitor your spending. Check your AWS Credit usage using [Cost Explorer](https://console.aws.amazon.com/cost-management/home?region=us-east-1#/) often. **We are not responsible for managing your expenses or detecting non-terminated instances.**
