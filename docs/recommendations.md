# Recommendations

As this may be some of your first full-system FPGA projects, we provide the following recommendations below. We write these recommendations remembering the times we've been burned by not following them. They may seem cumbersome, but they are useful.

## Hardware Recommendations

* Create a low-performing working algorithm right away
* Always implement optimizations one at a time with verification inbetween
* Always test in software emulation before testing in hardware emulation or hardware
* Have your testing script verify over less data when testing in hardware emulation
* Carefully look over logs and the HLS guidance for further optimizations and when debugging
* Implement most optimizations found in lab 3 and 4

## Software Recommendations

* Create a low-performing working solution right away
* Create changes one at a time with verification in-between
* Script your verification and benchmarking flow as early as possible
    - Scripting makes it easier to do parameter sweeps

## Overall Recommendations

* Carefully heed all warnings given by aws_ppi
* Do not use `aws_ppi ssh` if possible, watchdog timers will only reactivate when you leave aws_ppi with ctrl+d
* Do not have multiple processes of aws_ppi running at the same time
* Check your EC2 console whenever finished for the day to ensure no instances are running
* **Get started on the project right away,** full system integration takes time, you cannot start this a week before the deadline


Finally, as a reminder, your teaching staff is available for any questions you might have. However, we will adopt the tradition to be more hands-off during the final project. Piazza posts will be answered in batch twice or once per day from the release of the project. You will need to rely on your peers to answer most non-logistical questions, keeping in mind that sharing code is academic misconduct.
